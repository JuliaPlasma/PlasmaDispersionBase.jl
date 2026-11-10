"""
    NormalizedPopulation(Omega, Pi2, vdf, c = 1)

One population in normalized units: `Omega = Ω_s/Ω_ref` (signed), `Pi2 = (ω_ps/Ω_ref)²` (negative for a
negative-density population), and `vdf` with speeds in units of `c`, the speed of light in `vdf`'s own
speed unit. Each solver converts `vdf` to its own representation.
"""
struct NormalizedPopulation{T, V, C}
    Omega::T
    Pi2::T
    vdf::V
    c::C
end
NormalizedPopulation(Omega, Pi2, vdf, c = 1) = NormalizedPopulation(promote(Omega, Pi2)..., vdf, c)

"""
    NormalizedPlasma(populations...)
    NormalizedPlasma(plasma::Plasma; ref)

Dimensionless plasma: a collection of [`NormalizedPopulation`](@ref), or a physical `Plasma` with
frequencies normalized to `Ω_ref`, see [`scales`](@ref).
"""
struct NormalizedPlasma{S, C}
    populations::S
    scales::C
end
NormalizedPlasma(populations) = NormalizedPlasma(populations, nothing)
NormalizedPlasma(populations::NormalizedPopulation...) = NormalizedPlasma(populations, nothing)
NormalizedPlasma(p::NormalizedPlasma) = p

function NormalizedPlasma(plasma::Plasma; ref = _default_ref(plasma))
    sc = scales(plasma; ref)
    sps = map(plasma.populations, sc.populations) do s, ss
        NormalizedPopulation(ss.wc, sign(s.n) * ss.wp^2, s.vdf, C_SI)
    end
    return NormalizedPlasma(sps, sc)
end

"""
    scales(plasma::Plasma; ref)

Dimensionless scales of a physical plasma: speeds in `c`, frequencies in `Ω_ref`, lengths in
`c/Ω_ref`. `ref` is a `Particle`, whose gyrofrequency is `Ω_ref`, or a frequency [rad/s]; by default
the first population's particle, or its plasma frequency when `B0 = 0`.

- `vA`, `B0` [T], `Omega_ref` [rad/s]
- `populations`: per population `(; wc, wp, d, vth_perp, vth_para, rho)` with signed
  `wc = Ω_s/Ω_ref`, magnitude `wp = ω_ps/Ω_ref`, inertial length `d = 1/wp`, and, for PlasmaPopulations
  models, thermal speeds and gyroradius `rho = vth_perp/|wc|` (else `nothing`)
"""
function scales(plasma::Plasma; ref = _default_ref(plasma))
    B0 = norm(plasma.B0)
    Ωref = ref isa Particle ? abs(gyrofrequency(ref, B0)) : si_frequency(ref)
    iszero(Ωref) && throw(ArgumentError("Ω_ref = 0: with B0 = 0, give `ref` as a frequency [rad/s]"))
    sp = map(plasma.populations) do s
        wc = gyrofrequency(s.particle, B0) / Ωref
        wp = plasma_frequency(s.particle, s.n) / Ωref
        vth_perp, vth_para = _vth_perp_para(s.vdf)
        rho = isnothing(vth_perp) ? nothing : vth_perp / abs(wc)
        (; wc, wp, d = 1 / wp, vth_perp, vth_para, rho)
    end
    return (; vA = alfven_speed(plasma) / C_SI, B0, Omega_ref = Ωref, populations = sp)
end

_vth_perp_para(d::AbstractVDF) = vth_perp_para(d) ./ C_SI
_vth_perp_para(_) = (nothing, nothing)
