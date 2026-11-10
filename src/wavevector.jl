"""
    Wavevector(k_perp, k_para)
    Wavevector(; k, theta)

One wavevector `𝐤 = (k_perp, 0, k_para)` in the frame with `ẑ ∥ B`, or from `|k|` and radian angle
`θ` to `B₀`, as in [`AngleSweep`](@ref).
"""
struct Wavevector{T}
    k_perp::T
    k_para::T
end
Wavevector(k_perp, k_para) = Wavevector(promote(k_perp, k_para)...)
Wavevector(; k, theta) = Wavevector(k .* sincos(theta)...)

Base.eltype(::Type{Wavevector{T}}) where {T} = T
Base.convert(::Type{Wavevector{T}}, k::Wavevector) where {T} = Wavevector{T}(k.k_perp, k.k_para)
Base.broadcastable(k::Wavevector) = Ref(k)

@inline Base.abs2(k::Wavevector) = k.k_para^2 + k.k_perp^2
@inline Base.angle(k::Wavevector) = atan(k.k_perp, k.k_para)  # propagation angle to B0

Base.:-(a::Wavevector, b::Wavevector) = Wavevector(a.k_perp - b.k_perp, a.k_para - b.k_para)
LinearAlgebra.norm(k::Wavevector) = hypot(k.k_perp, k.k_para)

abstract type ParameterGeometry end

"""
    AngleSweep(k, theta)
    AngleSweep(; k, theta)

Polar k-space geometry: `|k|` and radian angle `θ` to `B₀`, so that `𝐤 = (k sinθ, k cosθ)`.
Each axis is a number or a vector of values.
"""
struct AngleSweep{K, T} <: ParameterGeometry
    k::K
    theta::T
end
AngleSweep(; k, theta) = AngleSweep(k, theta)

"""
    CartesianSweep(k_perp, k_para)
    CartesianSweep(; k_perp = false, k_para)

Cartesian k-space geometry. Each axis is a number or a vector of values.
"""
struct CartesianSweep{P, Z} <: ParameterGeometry
    k_perp::P
    k_para::Z
end
# k_perp=false: Bool is the weakest promoting Number, so k_para's type wins.
CartesianSweep(; k_perp = false, k_para) = CartesianSweep(k_perp, k_para)

_axes(g::AngleSweep) = (g.k, g.theta)
_axes(g::CartesianSweep) = (g.k_perp, g.k_para)
_wavevector(::AngleSweep, k, θ) = Wavevector(; k, theta = θ)
_wavevector(::CartesianSweep, k_perp, k_para) = Wavevector(k_perp, k_para)

_isswept(::AbstractVector) = true
_isswept(::Number) = false
_isswept(x) = throw(ArgumentError("sweep axis must be a number or a vector of values, got $(typeof(x))"))

"""
    wavevectors(k) -> AbstractArray{<:Wavevector}

Every wavevector of `k`, shaped like its swept axes: 0-dimensional for one `Wavevector`, `k` itself
for an array of them (other collections are collected), and one dimension per vector axis of a sweep.
"""
wavevectors(k::Wavevector) = fill(k)
wavevectors(ks::AbstractArray{<:Wavevector}) = ks
wavevectors(ks) = _collected(collect(ks))
_collected(ks::AbstractArray{<:Wavevector}) = ks
function wavevectors(g::ParameterGeometry)
    a, b = _axes(g)
    full = [_wavevector(g, x, y) for x in _values(a), y in _values(b)]
    return reshape(full, map(length, filter(_isswept, (a, b))))
end
_values(x) = _isswept(x) ? x : (x,)

"""
    realtype(x)

Real float type carried by a problem, target or wavevectors: the element type solvers work in, so
user number types (`Float32`, `BigFloat`, …) flow into solution containers.
"""
realtype(x) = float(real(eltype(x)))
realtype(::Nothing) = Union{}
realtype(x::Tuple) = promote_type(map(realtype, x)...)
realtype(ks::AbstractArray{<:Wavevector}) = float(eltype(eltype(ks)))
realtype(g::ParameterGeometry) = realtype(_axes(g))

_scale_k(::Nothing, s, op = *) = nothing
_scale_k(k::Wavevector, s, op = *) = Wavevector(op(k.k_perp, s), op(k.k_para, s))
_scale_k(g::AngleSweep, s, op = *) = AngleSweep(op.(g.k, s), g.theta)
_scale_k(g::CartesianSweep, s, op = *) = CartesianSweep(op.(g.k_perp, s), op.(g.k_para, s))
_scale_k(ks, s, op = *) = map(k -> _scale_k(k, s, op), ks)
