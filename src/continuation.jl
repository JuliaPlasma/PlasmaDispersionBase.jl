"""
    Continuation(base; order = 3, reltol = 1e-2, abstol = 1e-10, maxsubdiv = 10)

Track one root branch along an ordered `k` path, a `Seed` target: each step extrapolates a
degree-`order` polynomial in arclength through the last accepted roots, and the seeded algorithm
`base` corrects it. A seed anchored mid-path is tracked outward both ways.

A step is accepted within `reltol·|ω| + abstol` [`Ω_ref`] of its prediction; otherwise the k-interval
is bisected and retried, up to `maxsubdiv` levels. An interval still unresolved is `NaN`, and the
solution `ReturnCode.Partial`.
"""
struct Continuation{B, T}
    base::B
    order::Int
    reltol::T
    abstol::T
    maxsubdiv::Int
end
Continuation(base; order = 3, reltol = 1.0e-2, abstol = 1.0e-10, maxsubdiv = 10) =
    Continuation(base, order, promote(reltol, abstol)..., maxsubdiv)

# At one wavevector, a continuation is its base's solve.
seedsolver(prob, alg::Continuation; kw...) = seedsolver(prob, alg.base; kw...)

function _follow(prob, alg::Continuation; kw...)
    t0 = time_ns()
    sz = size(wavevectors(prob.k))
    length(sz) ≤ 1 || throw(ArgumentError("Continuation follows a path: higher-dimensional sweeps are not supported"))
    ks = vec(wavevectors(prob.k))
    (; omega0, at) = prob.target
    j = isnothing(at) ? firstindex(ks) : argmin(i -> norm(at - ks[i]), eachindex(ks))
    W = complex(float(typeof(omega0)))
    path = _Path(seedsolver(prob, alg.base; kw...), alg, W(omega0), ks[j], typeof(norm(ks[j]))[], W[], 0)
    omega = similar(ks, W)
    ω, r = _advance!(path, ks[j])
    resid = similar(ks, typeof(r))
    omega[j], resid[j] = ω, r
    for i in (j + 1):lastindex(ks)
        omega[i], resid[i] = _advance!(path, ks[i])
    end
    # Backward from the anchor, its accepted root the only history.
    empty!(path.s); empty!(path.w)
    isfinite(omega[j]) && _push!(path, ks[j], omega[j])
    for i in (j - 1):-1:firstindex(ks)
        omega[i], resid[i] = _advance!(path, ks[i])
    end
    code = all(isfinite, omega) ? ReturnCode.Success : ReturnCode.Partial
    stats = SolveStats(path.nevals, (time_ns() - t0) / 1.0e9)
    return DispersionSolution(_scalarize(reshape(omega, sz)), _resids(reshape(resid, sz)), stats, code, prob, alg)
end

# Residuals as the solver reports them: `nothing` from a solver without one.
_resids(::AbstractArray{Nothing}) = nothing
_resids(r) = _scalarize(r)

# The last `order + 1` accepted roots `w`, oldest first, at pairwise distinct arclengths `s` along
# the path travelled; `k` is the newest one's wavevector.
mutable struct _Path{S, A, W, K, T}
    solver::S
    alg::A
    seed::W
    k::K
    s::Vector{T}
    w::Vector{W}
    nevals::Int
end

# Step to `kt` as `(ω, resid)`, bisecting the interval from the last accepted point on rejection.
function _advance!(path, kt, depth = 0)
    guess = _predict(path, kt)
    r = path.solver(kt, guess)
    path.nevals += r.nevals
    if _accept(path, r.omega, guess)
        _push!(path, kt, r.omega)
        return r.omega, r.resid
    end
    unresolved = (oftype(guess, complex(NaN, NaN)), isnothing(r.resid) ? nothing : oftype(r.resid, NaN))
    stuck = isempty(path.w) || iszero(norm(kt - path.k))
    (stuck || depth ≥ path.alg.maxsubdiv) && return unresolved
    ωm, _ = _advance!(path, _kmid(path.k, kt), depth + 1)
    isfinite(ωm) || return unresolved
    return _advance!(path, kt, depth + 1)
end

function _push!(path, k, ω)
    d = norm(k - path.k)
    # Re-refining the same k replaces its node: the extrapolant needs distinct arclengths.
    if !isempty(path.w) && iszero(d)
        path.w[end] = ω
        return path
    end
    push!(path.s, isempty(path.s) ? zero(d) : last(path.s) + d)
    push!(path.w, ω)
    path.k = k
    if length(path.w) > path.alg.order + 1
        popfirst!(path.s)
        popfirst!(path.w)
    end
    return path
end

# Lagrange extrapolation in arclength through the retained nodes; the degree grows with the history.
function _predict(path, kt)
    (; s, w) = path
    isempty(w) && return path.seed
    st = last(s) + norm(kt - path.k)
    v = zero(eltype(w))
    for i in eachindex(s)
        l = one(st)
        for j in eachindex(s)
            j == i || (l *= (st - s[j]) / (s[i] - s[j]))
        end
        v += l * w[i]
    end
    return v
end

# Any accepted node gives an extrapolant (a constant from one node) whose error gauges the step.
function _accept(path, ω, guess)
    isfinite(ω) || return false
    isempty(path.w) && return true
    scale = max(abs(last(path.w)), abs(ω))
    return abs(ω - guess) ≤ path.alg.reltol * scale + path.alg.abstol
end

_kmid(a::Wavevector, b::Wavevector) = Wavevector((a.k_perp + b.k_perp) / 2, (a.k_para + b.k_para) / 2)
