"""
    Seed(omega0, at = nothing)

One root near `omega0`, or the branch through it along the problem's `k`; `at` anchors it to a
wavevector of `k`, the first by default.
"""
struct Seed{W, K}
    omega0::W
    at::K
end
Seed(omega0) = Seed(omega0, nothing)

"""
    Region(box)

Every root inside `box = (lowerleft, upperright)` of the complex ω plane, or anywhere for
`box = nothing` (only for algorithms that find all roots, e.g. matrix eigenvalue solvers).
"""
struct Region{B}
    box::B
end

Base.iterate(r::Region, state = 1) = iterate(r.box, state)
Base.getindex(r::Region, i) = r.box[i]

Base.in(z::Number, r::Region) = _in_box(r.box, z)
_in_box(::Nothing, z) = true
_in_box((ll, ur)::Tuple, z) = real(ll) ≤ real(z) ≤ real(ur) && imag(ll) ≤ imag(z) ≤ imag(ur)

# A survey's linking scale: the default `gate`, and how far past every edge of the box roots are linked.
_linkscale(r::Region) = _linkscale(r.box)
_linkscale(::Nothing) = Inf
_linkscale((ll, ur)::Tuple) = abs(ur - ll) / 8

"""
    window(target::Region)

The ω window a survey keeps roots in: the box widened by the default linking `gate`, ⅛ of its diagonal,
on every side. A branch leaving the box keeps its own roots just outside, rather than a neighbour's
being linked in their place; points outside the box itself are then `NaN`. Point solvers should
report the roots they find in the window, not only in the box.
"""
window(r::Region) = Region(_window(r.box))
_window(::Nothing) = nothing
function _window(box::Tuple)
    pad = _linkscale(box) * (1 + im)
    return (box[1] - pad, box[2] + pad)
end

"""
    DispersionProblem(plasma, target, k; ref, kwargs...)
    DispersionProblem(plasma, k; kwargs...)

Roots `ω` of the dispersion relation of `plasma` at wavevectors `k`.

Units follow `plasma`: with a physical `Plasma`, `target`, `k` and results are in rad/s and m⁻¹, and
solvers run on the form normalized by `ref`, see [`scales`](@ref). With a [`NormalizedPlasma`](@ref) or
a collection of [`NormalizedPopulation`](@ref)s, they are in `Ω_ref` and `Ω_ref/c`.

`target` selects which roots:

- [`Seed`](@ref)`(ω, at)` or a number `ω`: one root, or one branch along `k`.
- [`Region`](@ref)`((lowerleft, upperright))` or a tuple: every root in the ω box. Omitting
  `target` means every root.

`k` is a `Wavevector`, an array of them, or a sweep (`AngleSweep`, `CartesianSweep`).

`kwargs` are stored keywords of `solve`, as in SciML: `solve(prob, alg; kw...)` passes
`prob.kwargs` merged with `kw` to the solver, which rejects any it does not take.
"""
struct DispersionProblem{T, K, P, R, O}
    plasma::P
    target::T
    k::K
    ref::R
    kwargs::O
end

function DispersionProblem(plasma, target, k; ref = _default_ref(plasma), kwargs...)
    isnothing(ref) || plasma isa Plasma ||
        throw(ArgumentError("`ref` normalizes a physical `Plasma`; this plasma is already normalized"))
    return DispersionProblem(_plasma(plasma), _target(target), k, ref, NamedTuple(kwargs))
end
DispersionProblem(plasma, k; kw...) = DispersionProblem(plasma, Region(nothing), k; kw...)

# Without B0 there is no gyrofrequency to normalize by.
function _default_ref(plasma::Plasma)
    s = first(plasma.populations)
    return iszero(norm(plasma.B0)) ? abs(plasma_frequency(s.particle, s.n)) : s.particle
end
_default_ref(_) = nothing

# Solvers see one normalized type, whichever form the user passed.
_plasma(p::Plasma) = p
_plasma(p) = NormalizedPlasma(p)

_target(x) = x
_target(omega::Number) = Seed(omega)
_target(box::Tuple) = Region(box)

"""
    remake(prob::DispersionProblem; plasma, target, k, ref, kwargs)

`prob` with the given fields replaced.
"""
remake(prob::DispersionProblem; plasma = prob.plasma, target = prob.target, k = prob.k, ref = prob.ref, kwargs = prob.kwargs) =
    DispersionProblem(plasma, target, k; ref, kwargs...)

realtype(t::Seed) = promote_type(realtype(t.omega0), realtype(t.at))
realtype(r::Region) = realtype(r.box)
realtype(p::DispersionProblem) = promote_type(realtype(p.target), realtype(p.k))

"""
    normalized(prob::DispersionProblem) -> (prob, units)

The problem solvers run, with `plasma` a [`NormalizedPlasma`](@ref) and `target` and `k` in `Ω_ref`
and `Ω_ref/c`, and the `units = (; omega, k)` its ω and k multiply by to give `prob`'s. For
evaluating a solver's functions outside `solve`, at a problem's ω and k.
"""
normalized(prob::DispersionProblem) = prob, (omega = true, k = true) # the identity for every number type, NaN included
function normalized(prob::DispersionProblem{<:Any, <:Any, <:Plasma})
    plasma = NormalizedPlasma(prob.plasma; prob.ref)
    T = realtype(prob)
    Ω = plasma.scales.Omega_ref
    u = (omega = T(Ω), k = T(Ω / C_SI))
    p = remake(prob; plasma, target = _scale_target(prob.target, u), k = _scale_k(prob.k, u.k, /), ref = nothing)
    return p, u
end

_scale_target(t::Seed, u) = Seed(t.omega0 / u.omega, _scale_k(t.at, u.k, /))
_scale_target(t::Region{Nothing}, u) = t
_scale_target(t::Region, u) = Region(t.box ./ u.omega)

_algname(alg) = nameof(typeof(alg))

"""
    solve(prob::DispersionProblem, alg; kw...)

Solve `prob` on its normalized form and report in its units.

`sol.prob` is `prob` with `kw` merged into its stored keywords.
"""
function CommonSolve.solve(prob::DispersionProblem, alg; kwargs...)
    p, u = normalized(prob)
    kw = merge(p.kwargs, values(kwargs))
    return _restore(_solve(p, alg; kw...), remake(prob; kwargs = kw), u)
end

# Several solvers can be loaded together, so none owns a default.
CommonSolve.solve(::DispersionProblem; kwargs...) =
    throw(ArgumentError("pass an algorithm of a solver package: `solve(prob, alg)`"))
