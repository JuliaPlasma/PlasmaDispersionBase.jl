module PlasmaDispersionBase

using LinearAlgebra: LinearAlgebra, BLAS, norm
using PlasmaPopulations: Plasma, Particle, AbstractVDF, C_SI, gyrofrequency, plasma_frequency, alfven_speed,
    si_frequency, vth_perp_para
using CommonSolve
import CommonSolve: solve

export Wavevector, AngleSweep, CartesianSweep
export NormalizedPopulation, NormalizedPlasma, scales
export DispersionProblem, Seed, Region, ReturnCode, solve
export isgrowing, link, track, refine, dispersion_diagram, dispersion_diagram!

VERSION >= v"1.11.0-DEV.469" && eval(Meta.parse("""public ParameterGeometry, wavevectors, remake,
    DispersionSolution, SurveySolution, BranchSolution, DispersionBranch, SolveStats, successful_retcode,
    window, pointsolver, seedsolver, normalized, realtype"""))

include("wavevector.jl")
include("plasma.jl")
include("problem.jl")
include("solution.jl")
include("linking.jl")
include("survey.jl")
include("seeded.jl")

function dispersion_diagram! end
function dispersion_diagram end

end
