module PlasmaDispersionBase

using LinearAlgebra: LinearAlgebra, BLAS, norm
using PlasmaPopulations
using PlasmaPopulations: C_SI, gyrofrequency, plasma_frequency, alfven_speed, si_frequency
using CommonSolve
import CommonSolve: solve

eval(Expr(:export, filter!(n -> Base.isexported(PlasmaPopulations, n), names(PlasmaPopulations))...))
export Wavevector, AngleSweep, CartesianSweep
export NormalizedPopulation, NormalizedPlasma, scales
export DispersionProblem, Seed, Region, ReturnCode, solve
export isgrowing, link, track, refine, dispersion_diagram, dispersion_diagram!

VERSION >= v"1.11.0-DEV.469" && eval(Meta.parse("""public ParameterGeometry, wavevectors, remake,
    DispersionSolution, SurveySolution, BranchSolution, DispersionBranch, SolveStats, successful_retcode,
    window, pointsolver, seedsolver, normalized, realtype, muller"""))

include("wavevector.jl")
include("plasma.jl")
include("problem.jl")
include("solution.jl")
include("linking.jl")
include("survey.jl")
include("seeded.jl")
include("muller.jl")

function dispersion_diagram! end
function dispersion_diagram end

end
