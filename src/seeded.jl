"""
    seedsolver(prob, alg; kw...) -> s

Solver hook for algorithms that find one root from a guess. Built once from the normalized `prob`
and the solve's keywords, where everything shared between wavevectors is set up.
"""
function seedsolver(prob, alg; kw...)
    applicable(pointsolver, prob, alg) ||
        throw(ArgumentError("$(_algname(alg)) does not find a root from a seed"))
    return NearestRoot(pointsolver(remake(prob; target = Region(nothing)), alg; kw...))
end

struct NearestRoot{S}
    point::S
end
function (s::NearestRoot)(k, ω0)
    r = s.point(k)
    ωs = r.omega
    retcode = isempty(ωs) ? ReturnCode.Failure : r.converged ? ReturnCode.Success : ReturnCode.Saturated
    i = isempty(ωs) ? 0 : argmin(j -> abs(ωs[j] - ω0), eachindex(ωs))
    omega = iszero(i) ? eltype(ωs)(NaN, NaN) : ωs[i]
    resid = isnothing(r.resid) ? nothing : iszero(i) ? eltype(r.resid)(NaN) : r.resid[i]
    return (; omega, resid, r.nevals, retcode)
end

function _solve(prob::DispersionProblem{<:Seed, <:Wavevector}, alg; kw...)
    t0 = time_ns()
    r = seedsolver(prob, alg; kw...)(prob.k, prob.target.omega0)
    return DispersionSolution(r.omega, r.resid, SolveStats(r.nevals, (time_ns() - t0) / 1.0e9), r.retcode, prob, alg)
end

_solve(prob::DispersionProblem{<:Seed}, alg; kw...) = _follow(prob, alg; kw...)

# A seed along several wavevectors, by an algorithm that finds every root at each.
function _follow(prob, alg; kw...)
    applicable(pointsolver, prob, alg) || throw(ArgumentError(
        "$(_algname(alg)) finds one root at one wavevector: to follow a seed along several, wrap it in `Continuation`"
    ))
    survey = _solve(remake(prob; target = Region(nothing)), alg; kw...)
    (; omega0, at) = prob.target
    t = track(survey, omega0, @something(at, first(wavevectors(prob.k))))
    return DispersionSolution(t.omega, t.resid, t.stats, t.retcode, prob, alg)
end

"""
    refine(prob, guesses, alg; gate = Inf, kw...)
    refine(sol, alg; gate = Inf, kw...)

The roots of `prob` nearest `guesses`, one at each of its wavevectors and shaped like them, each
solved by `alg` on its own from its guess: a [`DispersionSolution`](@ref), `NaN` where a guess is
not finite or its solve fails. `prob`'s target is not used, so its plasma can be another model of
the one the guesses came from.
Points are independent, so a poor guess can converge to another branch's root: one that moves further than `gate · Ω_ref` is `NaN`.

`refine(sol, alg)` re-solves every root of `sol` on the plasma of `sol.prob` and returns the same
kind of solution: for [`SurveySolution`](@ref), roots that fail are dropped, and those that reach the same root are one.

`kw` are `solve` keywords of `alg`, over those stored on the problem.
"""
refine(prob::DispersionProblem, guesses, alg; gate = Inf, kw...) =
    _refine(_refiner(prob, alg, kw)..., guesses, alg, gate)

function _refiner(prob, alg, kw)
    prob = remake(prob; kwargs = merge(prob.kwargs, values(kw)))
    p, u = normalized(prob)
    return seedsolver(p, alg; p.kwargs...), prob, p, u
end

function _refine(seeded, prob, p, u, guesses, alg, gate)
    t0 = time_ns()
    ks = wavevectors(p.k)
    gs = _asarray(guesses) ./ u.omega
    size(gs) == size(ks) || throw(DimensionMismatch("$(size(gs)) guesses for $(size(ks)) wavevectors"))
    T = realtype(p)
    omega = fill(Complex{T}(NaN, NaN), size(ks))
    nevals = zeros(Int, size(ks))
    todo = filter(i -> isfinite(gs[i]), eachindex(ks))
    # The first solve runs alone: it fixes whether the solver reports residuals.
    r1 = isempty(todo) ? nothing : seeded(ks[first(todo)], gs[first(todo)])
    resid = isnothing(r1) || isnothing(r1.resid) ? nothing : fill(T(NaN), size(ks))
    function store!(i, r)
        nevals[i] = r.nevals
        successful_retcode(r.retcode) && abs(r.omega - gs[i]) ≤ gate || return
        omega[i] = r.omega
        isnothing(resid) || (resid[i] = r.resid)
        return
    end
    isnothing(r1) || store!(first(todo), r1)
    _single_blas() do
        Threads.@threads for i in todo[2:end]
            store!(i, seeded(ks[i], gs[i]))
        end
    end
    nfound = count(isfinite, omega)
    code = iszero(nfound) ? ReturnCode.Failure : nfound == length(todo) ? ReturnCode.Success : ReturnCode.Partial
    stats = SolveStats(sum(nevals), (time_ns() - t0) / 1.0e9)
    return _restore(DispersionSolution(_scalarize(omega), _scalarize(resid), stats, code, p, alg), prob, u)
end

refine(sol::DispersionSolution, alg; gate = Inf, kw...) = _refine(_refiner(sol.prob, alg, kw)..., sol.omega, alg, gate)
function refine(sol::BranchSolution, alg; gate = Inf, kw...)
    seeded, prob, p, u = _refiner(sol.prob, alg, kw)
    refined = map(b -> _refine(seeded, prob, p, u, b.omega, alg, gate), sol.branches)
    kept = filter(s -> any(isfinite, s.omega), refined)
    code = isempty(kept) ? ReturnCode.Failure :
        sol.retcode == ReturnCode.Saturated ? sol.retcode :
        all(successful_retcode, refined) ? ReturnCode.Success : ReturnCode.Partial
    branches = [DispersionBranch(s.omega, s.k, s.resid, s.omega_ref) for s in kept]
    stats = sum(s -> s.stats, refined; init = SolveStats(0, 0.0))
    return BranchSolution(branches, stats, code, prob, alg)
end

# Guesses that reach one root, to the accuracy a seeded solve stops at, are that one root.
_same(a, b) = abs(a - b) ≤ sqrt(eps(real(typeof(a)))) * max(abs(a), 1)

function refine(sol::SurveySolution, alg; gate = Inf, kw...)
    t0 = time_ns()
    seeded, prob, p, u = _refiner(sol.prob, alg, kw)
    ks = wavevectors(p.k)
    win = window(p.target)
    T = _rootfloat(sol)
    function cell(c)
        omega, resid, nevals, reports, complete = Complex{T}[], T[], 0, false, true
        for z in sol.roots[c]
            g = z / u.omega
            r = seeded(ks[c], g)
            nevals += r.nevals
            if !(successful_retcode(r.retcode) && abs(r.omega - g) ≤ gate && r.omega in win)
                complete &= !(z in sol.prob.target)
            elseif !any(w -> _same(w, r.omega), omega)
                push!(omega, r.omega)
                push!(resid, something(r.resid, T(NaN)))
                reports |= !isnothing(r.resid)
            end
        end
        return (; omega = omega .* u.omega, resid, nevals, reports, complete)
    end
    out = Vector{Any}(undef, length(ks))
    _single_blas() do
        Threads.@threads for i in eachindex(out)
            out[i] = cell(i)
        end
    end
    roots = reshape([o.omega for o in out], size(ks))
    resids = any(o -> o.reports, out) ? reshape([o.resid for o in out], size(ks)) : fill(nothing, size(ks))
    code = all(isempty, roots) ? ReturnCode.Failure :
        sol.retcode == ReturnCode.Saturated ? sol.retcode :
        all(o -> o.complete, out) ? ReturnCode.Success : ReturnCode.Partial
    stats = SolveStats(sum(o -> o.nevals, out), (time_ns() - t0) / 1.0e9)
    return SurveySolution(roots, resids, sol.k, sol.omega_ref, stats, code, prob, alg)
end
