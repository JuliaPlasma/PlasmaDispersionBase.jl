using PlasmaDispersionBase
using PlasmaDispersionBase: wavevectors, linksheets, ReturnCode
using LinearAlgebra: BLAS, norm
using PlasmaPopulations: Plasma, Population, Proton, Electron, BiMaxwellian, C_SI, gyrofrequency, plasma_frequency
using Test

@testset "Wavevector vocabulary" begin
    @test abs2(Wavevector(3.0, 4)) == 25
    @test angle(Wavevector(1.0, 1.0)) ≈ π / 4
    @test norm(Wavevector(4.0, 6.0) - Wavevector(1.0, 2.0)) == 5
    k = Wavevector(; k = 2.0, theta = π / 6)
    @test k.k_perp ≈ 1 && k.k_para ≈ √3
    @test CartesianSweep(; k_para = [1.0]).k_perp == 0

    @test size(wavevectors(Wavevector(1.0, 2.0))) == ()
    @test size(wavevectors(AngleSweep(; k = [1.0, 2.0, 3.0], theta = [0.0, π / 2]))) == (3, 2)
    ks = wavevectors(AngleSweep(; k = [1.0, 2.0], theta = π / 2))
    @test size(ks) == (2,) && ks[2].k_perp ≈ 2 && abs(ks[2].k_para) < 1.0e-15
    @test wavevectors(CartesianSweep(; k_perp = 1.0, k_para = [2.0, 3.0])) == [Wavevector(1.0, 2.0), Wavevector(1.0, 3.0)]
    @test_throws ArgumentError wavevectors(AngleSweep(; k = (1.0, 2.0), theta = 0.0))
end

@testset "linksheets" begin
    # One constant root is a single m=2 sheet
    sheets = linksheets(fill([1.0 + 0.0im], 5, 5); gate = 0.5)
    @test length(sheets) == 1 && all(==(1), only(sheets))

    # Far values are distinct sheets
    sheets = linksheets([[1.0 + 0.0im, 5.0 + 0.0im] for _ in 1:5]; gate = 0.5)
    @test length(sheets) == 2
    @test all(sh -> all(!iszero, sh), sheets)
    @test sort([first(sh) for sh in sheets]) == [1, 2]

    # Crossing branches, shuffled per cell: a seed follows its own line through the crossing
    ks = range(0.05, 2; length = 20)
    vals = [isodd(i) ? ComplexF64[k, 2 - k] : ComplexF64[2 - k, k] for (i, k) in enumerate(ks)]
    sheet = only(linksheets(vals; gate = Inf, seed = (CartesianIndex(1), 1)))
    @test [vals[i][sheet[i]] for i in eachindex(ks)] ≈ ks

    # A fast branch among many static roots is not cut
    vals = [[ComplexF64(k); 10 .+ 0.001 .* (1:50)] for k in ks]
    @test all(==(1), only(linksheets(vals; gate = Inf, seed = (CartesianIndex(10), 1))))

    # A growing root meets its damped twin at k₀ and the pair splits into two real roots.
    k₀, ks = 0.51, 0.0:0.025:1.0
    pair(k) = 9.1 + (k - k₀) .+ 0.5 .* (k < k₀ ? (im, -im) : (-1, 1)) .* sqrt(abs(k - k₀))
    vals = [ComplexF64[pair(k)..., fill(9.0 - 0.003im, 4)...] for k in ks]
    for cell in (firstindex(ks), lastindex(ks))
        @test all(<=(2), only(linksheets(vals; gate = Inf, seed = (CartesianIndex(cell), 1))))
    end

    # Crossing stable lines stay whole whichever cell's rounding makes it the seed
    ks = range(0, 1; length = 9)
    for cell in eachindex(ks), line in 1:2
        vals = [ComplexF64[2.7 + 1.4k^2, 3.16] for k in ks]
        vals[cell][line] += 1.0e-20im
        sheets = linksheets(vals; gate = 0.5)
        @test length(sheets) == 2
        @test all(sh -> allequal(sh) && !iszero(first(sh)), sheets)
    end
end

# A solver whose roots at k̃ are ω̃ = k̃∥ and ω̃ = 3 − k̃∥ + 0.1i, in normalized units, box or not
struct LineRoots end
PlasmaDispersionBase.pointsolver(prob, ::LineRoots) =
    k -> (; omega = ComplexF64[k.k_para, 3 - k.k_para + 0.1im], resid = nothing, nevals = 1, converged = true)

# A seeded solver: the root of LineRoots nearest the guess, failing beyond `reach` away
struct NearestLine end
PlasmaDispersionBase.seedsolver(prob, ::NearestLine; reach = 1) = function (k, ω0)
    ω = complex(argmin(z -> abs(z - ω0), (k.k_para, 3 - k.k_para + 0.1im)))
    ok = abs(ω - ω0) < reach
    return (; omega = ok ? ω : complex(ω0), resid = nothing, nevals = 1, retcode = ok ? ReturnCode.Success : ReturnCode.Failure)
end

# The root of {k̃∥, 1.2} nearest the guess: a static root beside a moving one
struct LineOrStatic end
PlasmaDispersionBase.seedsolver(prob, ::LineOrStatic) =
    (k, ω0) -> (; omega = complex(argmin(z -> abs(z - ω0), (k.k_para, 1.2))), resid = nothing, nevals = 1, retcode = ReturnCode.Success)

# A branch leaving the box through a mode just inside its edge
struct EdgeRoots end
PlasmaDispersionBase.pointsolver(prob, ::EdgeRoots) =
    k -> (; omega = ComplexF64[k.k_para + 0.05im, 1.0 - 0.02im], resid = nothing, nevals = 1, converged = true)

@testset "A branch leaving the box ends at its edge" begin
    kz = collect(0.5:0.05:1.5)
    survey = solve(DispersionProblem(NormalizedPopulation(1.0, 1.0, nothing), (0.0 - 0.1im, 1.0 + 0.1im), CartesianSweep(; k_para = kz)), EdgeRoots())
    # The survey's own roots are those inside the box
    @test survey.omega == [k ≤ 1 ? [k + 0.05im, 1.0 - 0.02im] : [1.0 - 0.02im] for k in kz]
    sol = link(survey)
    @test length(sol) == 2
    leaving = only(filter(isgrowing, sol))
    @test isequal(leaving.omega, [k ≤ 1 ? k + 0.05im : complex(NaN, NaN) for k in kz])
    @test only(filter(!isgrowing, sol)).omega == fill(1.0 - 0.02im, length(kz))
end

@testset "Survey and tracking through a solver" begin
    p = Proton()
    B0 = 1.0e-8
    plasma = Plasma(Population(p, BiMaxwellian; T = 1.0, n = 1.0e6), Population(Electron(), BiMaxwellian; T = 1.0, n = 1.0e6); B0)
    Ω = gyrofrequency(p, B0)
    kn = Ω / C_SI
    geom = CartesianSweep(; k_para = collect(range(0.1, 1.2; length = 12)) .* kn)

    # Physical units in and out: ω = c k∥
    sol = link(solve(DispersionProblem(plasma, (0.0 - 1im, 2.0 + 1im) .* Ω, geom), LineRoots()))
    @test sol.retcode == ReturnCode.Success
    @test length(sol) == 2
    b = only(filter(b -> all(isfinite, b.omega), sol))
    @test b.omega ≈ C_SI .* geom.k_para
    @test b.k == wavevectors(geom)
    @test b.omega_ref ≈ Ω && isnothing(b.resid)
    @test count(isgrowing, sol) == 1

    # Every root, no box; at a single wavevector `omega` is the vector of roots
    sol = solve(DispersionProblem(NormalizedPlasma(), Wavevector(0.0, 1.0)), LineRoots())
    @test sort(sol.omega; by = real) == [1.0, 2.0 + 0.1im]
    @test sol.k == Wavevector(0.0, 1.0) && isnothing(sol.resid)

    # Solvers see a NormalizedPlasma whichever normalized form was passed
    s = NormalizedPopulation(1.0, 1.0, nothing)
    @test all(p -> DispersionProblem(p, Wavevector(0.0, 1.0)).plasma == NormalizedPlasma(s), (s, (s,), NormalizedPlasma(s)))

    # Tracking a seed anchored mid-path through a survey's roots
    survey = solve(DispersionProblem(plasma, geom), LineRoots())
    sol = track(survey, 2.0 * Ω, Wavevector(0.0, kn))
    @test sol.retcode == ReturnCode.Success && sol.prob.target isa Seed
    @test sol.omega ≈ (3 .- geom.k_para ./ kn .+ 0.1im) .* Ω
    @test sol.k == wavevectors(geom) && sol.omega_ref ≈ Ω && isgrowing(sol)

    # A seed is the same branch without the survey, and at one wavevector the nearest root
    seeded = solve(DispersionProblem(plasma, Seed(2.0 * Ω, Wavevector(0.0, kn)), geom), LineRoots())
    @test seeded.omega == sol.omega && seeded.retcode == ReturnCode.Success && seeded.k == sol.k
    @test solve(DispersionProblem(plasma, 0.9 * Ω, Wavevector(0.0, kn)), LineRoots()).omega ≈ Ω
    @test solve(DispersionProblem(plasma, 1.2 * Ω, Wavevector(0.0, kn)), NearestLine()).omega ≈ Ω
    @test solve(DispersionProblem(plasma, 1.2 * Ω, Wavevector(0.0, kn)), Continuation(NearestLine())).omega ≈ Ω

    # Continuation follows the same branch from a mid-path anchor; `solve` keywords reach `base`
    cont = solve(DispersionProblem(plasma, Seed(2.0 * Ω, Wavevector(0.0, kn)), geom), Continuation(NearestLine()))
    @test cont.omega ≈ sol.omega && cont.retcode == ReturnCode.Success && cont.k == sol.k && isnothing(cont.resid)
    lost = solve(DispersionProblem(plasma, Seed(2.0 * Ω, Wavevector(0.0, kn)), geom), Continuation(NearestLine()); reach = 1.0e-9)
    @test lost.retcode == ReturnCode.Partial && all(isnan, lost.omega)
    # The first step is checked too: from one node the static root is nearer than the moving one
    ks = Wavevector.(0.0, kn .* [1.0, 1.5, 2.0])
    @test solve(DispersionProblem(plasma, Seed(1.0 * Ω), ks), Continuation(LineOrStatic())).omega ≈ [1.0, 1.5, 2.0] .* Ω

    # A guess at every wavevector, in problem units, refined by the single-wavevector solve:
    # a missing guess and a failed solve are both NaN
    guesses = sol.omega .* (1 + 1.0e-3)
    guesses[2] = NaN
    guesses[3] = 10Ω
    polished = refine(DispersionProblem(plasma, geom), guesses, NearestLine())
    @test polished.retcode == ReturnCode.Partial
    @test isnan(polished.omega[2]) && isnan(polished.omega[3])
    @test polished.omega[4:end] ≈ sol.omega[4:end]
    @test polished.k == wavevectors(geom) && polished.omega_ref ≈ Ω
    @test_throws DimensionMismatch refine(DispersionProblem(plasma, geom), guesses[1:2], NearestLine())

    # Refining with another algorithm at either stage: roots before linking, where a relinked
    # gate needs no new solve, or branches, which keep their shape
    @test size(survey.omega) == (12,) && survey.k == wavevectors(geom)
    refined = refine(survey, NearestLine())
    @test refined.retcode == ReturnCode.Success && refined.omega ≈ survey.omega
    @test length(link(refined)) == 2 && length(link(refined; gate = 1.0e-6)) > 2
    branches = link(survey)
    @test size(branches.omega) == (12, 2)
    @test refine(branches, NearestLine()).omega ≈ branches.omega
    # Guesses that reach the same root are one root
    doubled = PlasmaDispersionBase.SurveySolution(map(z -> [z; z .+ 0.01Ω], survey.roots), survey.resids, survey.k, survey.omega_ref, survey.stats, survey.retcode, survey.prob, survey.alg)
    @test refine(doubled, NearestLine()).omega ≈ survey.omega
    # A root that moves past the gate is dropped
    off = refine(sol.prob, sol.omega .+ 0.5Ω, NearestLine(); gate = 0.1)
    @test off.retcode == ReturnCode.Failure && all(isnan, off.omega)
    # Keywords stored on the solution's problem reach the refining algorithm, under the call's own
    stored = PlasmaDispersionBase.DispersionSolution(sol.omega, sol.resid, sol.stats, sol.retcode, PlasmaDispersionBase.remake(sol.prob; kwargs = (; reach = 0)), sol.alg)
    @test refine(stored, NearestLine()).retcode == ReturnCode.Failure
    @test refine(stored, NearestLine(); reach = 1).omega ≈ sol.omega

    # Problem keywords are solve keywords: the solver rejects those it does not take
    @test_throws MethodError solve(DispersionProblem(plasma, geom; mode = :L), LineRoots())

    # No default algorithm; an algorithm without a hook for the target says so
    @test_throws ArgumentError solve(DispersionProblem(plasma, geom))
    @test_throws ArgumentError solve(DispersionProblem(plasma, geom), :not_an_algorithm)
    @test_throws ArgumentError solve(DispersionProblem(plasma, 1.0, geom), NearestLine())
    @test_throws ArgumentError solve(DispersionProblem(plasma, geom), NearestLine())
end

# Roots that report the BLAS threads each point ran with; `hook` runs first at every point but the first
struct BlasSeen{F}
    hook::F
end
PlasmaDispersionBase.pointsolver(prob, a::BlasSeen) = function (k)
    k.k_para == 1 || a.hook()
    return (; omega = ComplexF64[BLAS.get_num_threads()], resid = nothing, nevals = 1, converged = true)
end

@testset "BLAS runs single-threaded only while point solves run in parallel" begin
    n0 = BLAS.get_num_threads()
    BLAS.set_num_threads(4)
    x = Threads.threadpoolsize() == 1 ? 4 : 1
    seen = [[4], [x], [x]]
    survey(alg) = solve(DispersionProblem(NormalizedPlasma(), CartesianSweep(; k_para = [1.0, 2.0, 3.0])), alg).omega
    @test survey(BlasSeen(Returns(nothing))) == seen
    # A survey that outlasts a concurrent one stays single-threaded
    entered, release = Base.Event(), Base.Event()
    a = Threads.@spawn survey(BlasSeen(() -> (notify(entered); wait(release))))
    wait(entered)
    @test survey(BlasSeen(() -> (notify(release); wait(a)))) == [[x], [x], [x]]
    @test fetch(a) == seen && BLAS.get_num_threads() == 4
    BLAS.set_num_threads(n0)
end

@testset "Unmagnetized plasma normalizes by a plasma frequency" begin
    p = Proton()
    plasma = Plasma(Population(p, BiMaxwellian; T = 1.0, n = 1.0e6); B0 = 0.0)
    @test scales(plasma).Omega_ref ≈ plasma_frequency(p, 1.0e6)
    @test_throws ArgumentError scales(plasma; ref = p)
end

@testset "Aqua" begin
    using Aqua
    Aqua.test_all(PlasmaDispersionBase)
end
