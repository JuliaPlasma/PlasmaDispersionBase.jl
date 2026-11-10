struct SolveStats{T}
    nevals::Int # number of evaluations
    time::T # seconds
end

Base.:+(a::SolveStats, b::SolveStats) = SolveStats(a.nevals + b.nevals, a.time + b.time)

module ReturnCode
"""
    ReturnCode.T

Return codes are notes given by the solvers to indicate the state of the solution.

- `Success`   — every requested root converged.
- `Partial`   — some requested points have no root: `NaN` on a branch, dropped from a refined survey.
- `Saturated` — a survey's root search stopped short of its tolerance, so it may have **missed roots**.
- `MaxIters`  — the polisher hit its iteration cap without reaching tolerance.
- `Failure`   — no root found at all.
"""
@enum T Success Partial Saturated MaxIters Failure
end

"""
    successful_retcode(x)::Bool

True when `x` reports a fully converged solve.
"""
successful_retcode(c::ReturnCode.T) = c === ReturnCode.Success
successful_retcode(sol) = successful_retcode(sol.retcode)

# Fields `omega`, `k`, `resid`, `omega_ref`, with `omega` the type `W`.
abstract type AbstractBranch{W} end

Base.length(b::AbstractBranch) = length(b.omega)
Base.iterate(b::AbstractBranch, args...) = iterate(b.omega, args...)
Base.getindex(b::AbstractBranch, args...) = getindex(b.omega, args...)
Base.eltype(::Type{<:AbstractBranch{W}}) where {W} = eltype(W)
Base.lastindex(b::AbstractBranch) = lastindex(b.omega)

"""
    DispersionSolution(omega, resid, stats, retcode, prob, alg)

One branch with its solve: of a `Seed` problem, of `track`, or of `refine`. Solvers construct it on
their normalized `prob`; `k` and `omega_ref` follow from it. `resid` is `NaN` at non-converged
entries.
"""
struct DispersionSolution{W, K, R, U, S, Pr, A} <: AbstractBranch{W}
    omega::W
    k::K
    resid::R
    omega_ref::U
    stats::S
    retcode::ReturnCode.T
    prob::Pr
    alg::A
end
DispersionSolution(omega, resid, stats, retcode, prob, alg) =
    DispersionSolution(omega, _scalarize(wavevectors(prob.k)), resid, true, stats, retcode, prob, alg)

"""
    SurveySolution

Result of a `Region` `solve`: the roots found at each wavevector.
"""
struct SurveySolution{W, E, Z, R, K, U, S, Pr, A}
    omega::W
    resid::E
    roots::Z    # with those of `window(prob.target)` past the box, which linking needs
    resids::R
    k::K
    omega_ref::U
    stats::S
    retcode::ReturnCode.T
    prob::Pr
    alg::A
end
function SurveySolution(roots, resids, k, omega_ref, stats, retcode, prob, alg)
    omega, resid = _inbox(roots, resids, prob.target)
    return SurveySolution(_scalarize(omega), _scalarize(resid), roots, resids, k, omega_ref, stats, retcode, prob, alg)
end
Base.propertynames(::SurveySolution) = (:omega, :resid, :k, :omega_ref, :stats, :retcode, :prob, :alg)

_inbox(roots, resids, ::Region{Nothing}) = roots, _inbox_resids(resids, roots, Returns(true))
# `findall(in(target), z)` would take `target` for a collection.
function _inbox(roots, resids, target)
    keep(z) = z in target
    return map(z -> filter(keep, z), roots), _inbox_resids(resids, roots, keep)
end
_inbox_resids(::AbstractArray{Nothing}, roots, keep) = nothing
_inbox_resids(resids, roots, keep) = map((r, z) -> r[findall(keep, z)], resids, roots)

"""
    BranchSolution

The branches of a linked survey, a vector of [`DispersionBranch`](@ref)es: `sol[i]` is the `i`-th
branch, `filter(isgrowing, sol)` keeps the unstable ones. Each branch spans the full
`k` grid (missing points are `NaN`).
"""
struct BranchSolution{BR, S, Pr, A} <: AbstractVector{BR}
    branches::Vector{BR}
    stats::S
    retcode::ReturnCode.T
    prob::Pr
    alg::A
end

# The branches side by side: `omega` and `resid` shaped `(size(k)..., nbranch)`.
@inline function Base.getproperty(s::BranchSolution{BR}, name::Symbol) where {BR}
    name === :k && return _scalarize(wavevectors(getfield(s, :prob).k))
    bs = getfield(s, :branches)
    if name === :omega
        return isempty(bs) ? zeros(eltype(BR), size(wavevectors(getfield(s, :prob).k))..., 0) : stack(b -> b.omega, bs)
    elseif name === :resid
        return isempty(bs) || isnothing(first(bs).resid) ? nothing : stack(b -> b.resid, bs)
    end
    return getfield(s, name)
end
Base.propertynames(s::BranchSolution) = (fieldnames(typeof(s))..., :omega, :k, :resid)

Base.filter(pred, s::BranchSolution) = BranchSolution(filter(pred, s.branches), s.stats, s.retcode, s.prob, s.alg)
Base.size(s::BranchSolution) = size(s.branches)
Base.getindex(s::BranchSolution, args...) = s.branches[args...]

_show_stats(io, st::SolveStats) = print(io, st.nevals, " evals, ", round(st.time; sigdigits = 3), " s")

function Base.show(io::IO, s::SurveySolution)
    print(io, "SurveySolution(retcode=", s.retcode, ", ", sum(z -> count(in(s.prob.target), z), s.roots), " roots at ", length(s.roots), " wavevectors, ")
    _show_stats(io, s.stats)
    return print(io, ")")
end

function Base.show(io::IO, s::BranchSolution)
    print(io, "BranchSolution(retcode=", s.retcode, ", ", length(s.branches), " branches, ")
    _show_stats(io, s.stats)
    return print(io, ")")
end
Base.show(io::IO, ::MIME"text/plain", sol::BranchSolution) = show(io, sol)

function Base.show(io::IO, sol::DispersionSolution)
    print(io, "DispersionSolution(retcode=", sol.retcode, ", omega=", sol.omega, ", ")
    _show_stats(io, sol.stats)
    return print(io, ")")
end

"""
    DispersionBranch

One root branch over the wavevectors of a survey. `resid` is `nothing` for solvers without one.
"""
struct DispersionBranch{W, K, R, U} <: AbstractBranch{W}
    omega::W
    k::K
    resid::R
    omega_ref::U
end

"""
    isgrowing(branch, margin = 1e-3)

Whether `branch` grows anywhere faster than `margin · Ω_ref`.
"""
isgrowing(b::AbstractBranch, margin = 1.0e-3) =
    maximum((imag(ω) for ω in b if isfinite(ω)); init = (-Inf)) > margin * b.omega_ref

_restore(sol::DispersionSolution, prob, u) = DispersionSolution(
    sol.omega .* u.omega, _scale_k(sol.k, u.k), sol.resid, sol.omega_ref * u.omega, sol.stats, sol.retcode, prob, sol.alg
)
_restore(sol::SurveySolution, prob, u) = SurveySolution(
    map(z -> z .* u.omega, sol.roots), sol.resids, _scale_k(sol.k, u.k), sol.omega_ref * u.omega, sol.stats, sol.retcode, prob, sol.alg
)
