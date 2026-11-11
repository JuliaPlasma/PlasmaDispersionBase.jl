# PlasmaDispersionBase.jl

Common interface of linear plasma dispersion solvers: wavevectors, problems, solutions, unit
normalization and branch linking. Solvers are separate packages that plug in their algorithms:

| package | algorithms |
|---|---|
| [VlasovMaxwellDispersion.jl](https://github.com/JuliaSpacePhysics/VlasovMaxwellDispersion.jl) | root finding on the exact hot-plasma `det 𝒟`: `Muller`, `AAA`, `GRPF` |
| [PlasmaBO.jl](https://github.com/JuliaSpacePhysics/PlasmaBO.jl) | every root at once as matrix eigenvalues: `BOHH`, `BOPBK`, `BOFluid` |

Plasmas are described with [PlasmaPopulations.jl](https://github.com/JuliaPlasma/PlasmaPopulations.jl), re-exported here.

```julia
using VlasovMaxwellDispersion   # or PlasmaBO; both re-export this package's API
using PlasmaPopulations: gyrofrequency

plasma = Plasma(Population(Proton(), BiMaxwellian; T = 100.0, n = 5e6), Population(Electron(), BiMaxwellian; T = 50.0, n = 5e6); B0 = 5e-9)
Ωp = gyrofrequency(Proton(), plasma.B0)
ks = AngleSweep(; k = range(1e-6, 1e-5, 20), theta = deg2rad(30))
alg = AAA()   # or PlasmaBO's BOHH()

survey = solve(DispersionProblem(plasma, ((0.0 - 0.3im) * Ωp, (1.1 + 0.05im) * Ωp), ks), alg)  # roots at each k
branches = link(survey)                                                                        # joined along k
filter(isgrowing, branches)
```

## API

| name | meaning |
|---|---|
| `Wavevector(k_perp, k_para)`, `Wavevector(; k, theta)` | one wavevector `[k_perp, 0, k_para]`, `ẑ ∥ B₀`, `k_perp` signed |
| `AngleSweep(; k, theta)`, `CartesianSweep(; k_perp = 0, k_para)` | polar `𝐤 = [k sinθ, 0, k cosθ]` or Cartesian grid; each axis a number or a vector |
| `DispersionProblem(plasma, target, k; ref, kw...)` | roots at wavevectors `k`: a `Wavevector`, an array of them, or a sweep; `kw` are stored `solve` keywords |
| `Seed(ω, at)`, a number | `target`: one branch through `ω` at wavevector `at` (the first by default) |
| `Region((lowerleft, upperright))`, a tuple | `target`: every root in the ω box, or every root when omitted; `ω in region` tests the box |
| `solve(prob, alg; kw...)` | `DispersionSolution` for a `Seed`; `SurveySolution` for a `Region`; `sol.prob` keeps `kw` |
| `link(survey; gate)::BranchSolution` | the survey's roots joined into branches along `k`; `gate` [`Ω_ref`] bounds a branch's distance from its extrapolation, by default ⅛ of the box diagonal |
| `NormalizedPlasma(plasma; ref)`, `NormalizedPopulation(Omega, Pi2, vdf, c = 1)` | dimensionless plasma: `Ω_s/Ω_ref`, `(ω_ps/Ω_ref)²`, speeds of `vdf` in units of `c` |
| `scales(plasma; ref)` | the dimensionless scales of that normalization |
| `isgrowing(branch, margin = 1e-3)` | growth above `margin·Ω_ref`, so the threshold follows `ref`: pass `margin` or `ref` when the first population is not the scale of interest |
| `track(survey, ω0, k0; gate = Inf)` | the branch through the survey's root nearest `ω0` at its wavevector nearest `k0`, a `DispersionSolution` |
| `refine(prob, ωs, alg; gate = Inf, kw...)` | a guess at each wavevector of `prob`, `ωs` shaped like them, each solved by `alg`: a `DispersionSolution`, `NaN` where a guess is not finite, its solve fails, or the root moves further than `gate·Ω_ref`. `kw` are `solve` keywords of `alg`, over those stored on `prob` |
| `refine(sol, alg; gate = Inf, kw...)` | the same from every root of `sol`, at any stage: a survey before linking, its branches, or a seeded solution; e.g. eigenvalue roots polished on the exact determinant. |
| `Continuation(base; order = 3, reltol = 1e-2, abstol = 1e-10, maxsubdiv = 10)` | algorithm for a `Seed` along a path with seeded algorithm `base`: predicts each root by extrapolating the last `order + 1` along `k`, and bisects the step where `base` lands further than `reltol·\|ω\| + abstol` [`Ω_ref`] from it, up to `maxsubdiv` times, else `NaN`; a seed anchored mid-path is followed both ways. |
| `ReturnCode.T` | `Success`, `Partial`, `Saturated`, `MaxIters`, `Failure` |
| `dispersion_diagram(sol; k_unit, omega_unit)`, `dispersion_diagram!` | `Re ω(k)` and `Im ω(k)` of linked branches or of seeded solutions, or of a survey's roots in its box as unlinked points; with Makie loaded |
| below | public, not exported |
| `wavevectors(k)` | array of the wavevectors of `k`, shaped like its swept axes |
| `remake(prob; plasma, target, k, ref, kwargs)` | `prob` with fields replaced |

### Units

With a physical `Plasma`, `ω`, boxes and seeds are in rad/s and `k` in m⁻¹, and so are results.
Solvers run on the form normalized by `ref`: `Ω_ref` is the gyrofrequency of `ref`, a `Particle`,
or a frequency [rad/s] (default: the first population's particle, or its plasma frequency when
`B0 = 0`), and `k` is in `Ω_ref/c`.
With a `NormalizedPlasma`, a collection of `NormalizedPopulation`s or a single one, inputs and results
are in those normalized units.

### Solutions

All solution types share `retcode`, `stats`, `prob`, `alg` fields.

`SurveySolution`: `omega`, the roots inside the box as a vector at each wavevector (the vector itself
at a single `Wavevector`), `resid` alike, `k`, `omega_ref`.

A branch has fields `omega` (one root, or an array shaped like the wavevectors), `k`, `resid` (the
solver's residual, or `nothing`) and `omega_ref` (`Ω_ref` in problem units), and iterates `omega`.
Points where a branch is absent are `NaN`.
`DispersionSolution` is one branch and `BranchSolution` is a vector of `DispersionBranch`es  (its `omega` and `resid` are the branches', shaped `(size(k)..., nbranch)`).

## Writing a solver

Hooks are called on the normalized problem; the problem's `plasma` is a `NormalizedPlasma`. Each of
its `populations` carries `Omega`, `Pi2` and the `vdf` as the user gave it, with speeds in the unit in which the speed of light is the population's `c` (`C_SI` from a physical `Plasma`); the solver converts `vdf` to its own representation and divides its speeds by `c`.

Hooks get the problem's keywords merged with the `solve` call's, and reject the ones they don't take.
An algorithm implements one of them; each returns a callable built once per solve from the normalized `prob`, and called from several threads at once.

| hook | for | gives |
|---|---|---|
| `seedsolver(prob, alg; kw...) -> s`, `s(k, ω0) = (; omega, resid, nevals, retcode)` | one root from a guess | `Seed` at one `Wavevector`, `refine` |
| `pointsolver(prob, alg; kw...) -> s`, `s(k) = (; omega, resid, nevals, converged)` | every root at one wavevector | `Region` surveys |

A `pointsolver` searches at least `window(prob.target)`, wider than the box: roots outside the window are dropped.

Public, not exported, for solvers:

| name | meaning |
|---|---|
| `normalized(prob)` | `(prob, units)`: the problem solvers run, and the `(; omega, k)` its ω and k multiply by to give `prob`'s; for a solver's functions outside `solve` (a residual, a matrix) |
| `realtype(prob)` | the real float type of `prob`'s numbers, for a solver's containers |
| `successful_retcode(sol)` | whether a solve fully converged |
| `muller(f, x0; atol = 1e-10, maxiter = 100)`, `muller(f, x0, x1, x2; …)` -> (root, nevals, retcode) | Derivative-free root-finding algorithm of complex `f` by David Muller in 1956, from a guess or three starting points; `root` is `NaN` unless converged |
| `window(target)` | the ω window a survey keeps roots in for linking: the box widened by ⅛ of its diagonal per side |
