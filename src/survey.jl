"""
    pointsolver(prob, alg; kw...) -> s

Solver hook for algorithms that find every root at one wavevector.
"""
function pointsolver end

function _solve(prob::DispersionProblem{<:Region}, alg; kw...)
    t0 = time_ns()
    _check_pointsolver(prob, alg)
    ks, zv, rv, nev, conv = _pointroots(pointsolver(prob, alg; kw...), prob)
    stats = SolveStats(sum(nev), (time_ns() - t0) / 1.0e9)
    # `converged = false` in the discovery process may miss some roots ⇒ `Saturated`.
    code = all(isempty, zv) ? ReturnCode.Failure : all(conv) ? ReturnCode.Success : ReturnCode.Saturated
    return SurveySolution(zv, rv, _scalarize(ks), true, stats, code, prob, alg)
end

# `applicable` ignores keywords, so a solver still rejects the keywords it does not take.
_check_pointsolver(prob, alg) = applicable(pointsolver, prob, alg) ||
    throw(ArgumentError("$(_algname(alg)) does not find every root at a wavevector, which a `Region` target needs"))

# The first point fixes the containers' element types for inference; the rest run threaded.
function _pointroots(point, prob)
    T = realtype(prob)
    win = window(prob.target)
    ks = wavevectors(prob.k)
    r1 = point(first(ks))
    zv = similar(ks, Vector{Complex{T}})
    rv = similar(ks, typeof(r1.resid))
    nev = similar(ks, Int)
    conv = similar(ks, Bool)
    function store!(i, r)
        keep = findall(z -> z in win, r.omega)
        return (zv[i], rv[i], nev[i], conv[i]) = (r.omega[keep], _pick(r.resid, keep), r.nevals, r.converged)
    end
    i1 = firstindex(ks)
    store!(i1, r1)
    _single_blas() do
        Threads.@threads for i in eachindex(ks)
            i == i1 || store!(i, point(ks[i]))
        end
    end
    return ks, zv, rv, nev, conv
end

_pick(::Nothing, keep) = nothing
_pick(x, keep) = x[keep]

# Point solvers run on parallel threads; BLAS threads on top would oversubscribe.
function _single_blas(f)
    n = BLAS.get_num_threads()
    n == 1 && return f()
    BLAS.set_num_threads(1)
    try
        return f()
    finally
        BLAS.set_num_threads(n)
    end
end

# m=0 sweeps flow through as 0-dim arrays; collapse to scalars at construction
_scalarize(x::AbstractArray{<:Any, 0}) = x[]
_scalarize(x) = x

_gather(vals, sheet, nan) = map((v, i) -> iszero(i) ? nan : oftype(nan, v[i]), vals, sheet)
_gather_resid(::AbstractArray{Nothing}, sheet, T) = nothing
_gather_resid(resids, sheet, T) = _scalarize(_gather(resids, sheet, T(NaN)))

# Roots in `Ω_ref`, the unit of `gate`; a sheet keeps only its roots inside the box.
_linkroots(sol) = map(z -> z ./ sol.omega_ref, sol.roots)
function _cut!(sheet, sol)
    for c in eachindex(sheet, sol.roots)
        iszero(sheet[c]) || sol.roots[c][sheet[c]] in sol.prob.target || (sheet[c] = 0)
    end
    return sheet
end
_rootfloat(sol) = real(eltype(eltype(sol.roots)))

"""
    link(sol::SurveySolution; gate)::BranchSolution

Join the roots of a survey into branches continuous along `k`. `gate`,
in `Ω_ref`, is the furthest a branch may sit from its extrapolation at the next wavevector: by
default ⅛ of the box diagonal, unbounded without a box. A branch ends where it leaves the box.
"""
function link(sol::SurveySolution; gate = _linkscale(sol.prob.target) / sol.omega_ref)
    sheets = linksheets(_linkroots(sol); gate)
    filter!(s -> any(!iszero, _cut!(s, sol)), sheets)
    T = _rootfloat(sol)
    branches = map(sheets) do sheet
        ωs = _scalarize(_gather(sol.roots, sheet, Complex{T}(NaN, NaN)))
        DispersionBranch(ωs, sol.k, _gather_resid(sol.resids, sheet, T), sol.omega_ref)
    end
    code = isempty(branches) ? ReturnCode.Failure : sol.retcode
    return BranchSolution(branches, sol.stats, code, sol.prob, sol.alg)
end

"""
    track(sol::SurveySolution, omega0, k0; gate = Inf)

The branch through the root nearest `omega0` at the wavevector of `sol` nearest `k0`, linked
outward over the roots of the survey `sol`, as a [`DispersionSolution`](@ref) shaped like the
wavevectors and ending where it leaves the box. `gate` is [`link`](@ref)'s, unbounded by default.
"""
function track(sol::SurveySolution, omega0::Number, k0; gate = Inf)
    ks = _asarray(sol.k)
    cell = argmin(c -> norm(ks[c] - k0), CartesianIndices(ks))
    isempty(sol.roots[cell]) && throw(ArgumentError("no root at the wavevector nearest `k0`"))
    seed = (cell, argmin(abs.(sol.roots[cell] .- omega0)))
    sheet = _cut!(only(linksheets(_linkroots(sol); gate, seed)), sol)
    T = _rootfloat(sol)
    omega = _scalarize(_gather(sol.roots, sheet, Complex{T}(NaN, NaN)))
    code = sol.retcode == ReturnCode.Saturated ? sol.retcode :
        all(!iszero, sheet) ? ReturnCode.Success : ReturnCode.Partial
    prob = remake(sol.prob; target = Seed(omega0, k0))
    return DispersionSolution(omega, sol.k, _gather_resid(sol.resids, sheet, T), sol.omega_ref, sol.stats, code, prob, sol.alg)
end

_asarray(x::AbstractArray) = x
_asarray(x) = fill(x)
