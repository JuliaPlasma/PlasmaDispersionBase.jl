module PlasmaDispersionBaseMakieExt

using PlasmaDispersionBase: dispersion_diagram!, wavevectors, SurveySolution, DispersionSolution
import PlasmaDispersionBase
using Makie

_curves(sol::DispersionSolution) = (sol,)
_curves(sols) = sols

# |k| for the x-axis — works for a k∥ sweep (m=1) or a (|k|,θ) sweep (m=2) alike.
_kmag(k) = sqrt(abs2(k))

struct FigureAxesPlots
    figure::Makie.Figure
    axes::Tuple{Makie.Axis,Makie.Axis}
    plots::Vector{Makie.AbstractPlot}
end
Base.show(io::IO, m::MIME, d::FigureAxesPlots) = show(io, m, d.figure)
Base.show(io::IO, m::MIME"text/plain", d::FigureAxesPlots) = show(io, m, d.figure)
Base.showable(m::MIME{M}, fg::FigureAxesPlots) where {M} = showable(m, fg.figure)
Base.display(d::FigureAxesPlots) = display(d.figure)
Base.iterate(d::FigureAxesPlots, s=1) = s > 3 ? nothing : (getfield(d, s), s + 1)

function PlasmaDispersionBase.dispersion_diagram!((axr, axi), sol; k_unit = 1, omega_unit = 1, kw...)
    plots = Makie.AbstractPlot[]
    for root in _curves(sol)
        k = vec(map(_kmag, wavevectors(root.k))) ./ k_unit
        ω = vec(collect(root.omega)) ./ omega_unit
        p = sortperm(k)
        k, ωr, ωi = k[p], real.(ω)[p], imag.(ω)[p]
        push!(plots, Makie.scatterlines!(axr, k, ωr; linewidth=2, markersize=4, kw...))
        push!(plots, Makie.scatterlines!(axi, k, ωi; linewidth=2, markersize=4, kw...))
    end
    return plots
end

# Gray, so branches drawn on top stand out.
function PlasmaDispersionBase.dispersion_diagram!((axr, axi), sol::SurveySolution; k_unit = 1, omega_unit = 1, color=(:gray, 0.6), kw...)
    k = reduce(vcat, (fill(_kmag(kv), length(ω)) for (kv, ω) in zip(vec(wavevectors(sol.k)), vec(sol.omega))); init = Float64[]) ./ k_unit
    ω = reduce(vcat, vec(sol.omega); init = ComplexF64[]) ./ omega_unit
    return Makie.AbstractPlot[
        Makie.scatter!(axr, k, real.(ω); color, markersize=4, kw...),
        Makie.scatter!(axi, k, imag.(ω); color, markersize=4, kw...),
    ]
end

"""
    dispersion_diagram(sol; k_unit = 1, omega_unit = 1, figure = (;), xlabel = "|k|", title = "")

Two axes side by side: `Re ω(k)` and `Im ω(k)` of each branch of `sol`, in units of `k_unit`
and `omega_unit`. `sol` is a `BranchSolution`, a seeded `DispersionSolution`, or a vector of them;
a `SurveySolution` draws its roots in the box as unlinked points.

`fig, (axr, axi), plots = dispersion_diagram(sol)`.
"""
function PlasmaDispersionBase.dispersion_diagram(sol; figure=(;), xlabel="|k|", title="", kw...)
    fig = Makie.Figure(; figure...)
    axr = Makie.Axis(fig[1, 1]; xlabel, ylabel="Re ω")
    axi = Makie.Axis(fig[1, 2]; xlabel, ylabel="Im ω")
    isempty(title) || Makie.Label(fig[0, :], title; fontsize=16, font=:bold)
    Makie.linkxaxes!(axr, axi)
    plots = dispersion_diagram!((axr, axi), sol; kw...)
    Makie.colsize!(fig.layout, 1, Makie.Aspect(1, 1.0))
    Makie.colsize!(fig.layout, 2, Makie.Aspect(1, 1.0))
    Makie.resize_to_layout!(fig)
    return FigureAxesPlots(fig, (axr, axi), plots)
end

end
