using Test
include("../src/AtractorsQGP.jl")
using .AtractorsQGP
using StaticArrays
using Random

@testset "AtractorsQGP refactor API" begin
    model = BRSSSModel()
    u0 = [800.0, 0.5]
    tspan = (0.22, 0.5)

    sol = solve_hydro(model, u0, tspan; saveat=0.02)
    @test !isempty(sol.t)

    ics = generate_initial_conditions(5; T_range=(700.0, 900.0), A_range=(-1.0, 1.0))
    @test eltype(ics) == SVector{2,Float64}
    @test all(ic -> 700.0 * FM_PER_MEV <= ic[1] <= 900.0 * FM_PER_MEV, ics)

    ics_fm = generate_initial_conditions(5; T_range=(3.0, 4.0), temperature_unit=:fm, rng=MersenneTwister(123))
    @test all(ic -> 3.0 <= ic[1] <= 4.0, ics_fm)

    sols = generate_trajectories(model, ics, tspan; saveat=0.02, parallel=:serial)
    @test length(sols) == 5
    @test eltype(sols) != Any

    dataset = build_dataset(sols)
    @test size(dataset, 2) == 3

    dataset_mev = build_dataset(sols; temperature_unit=:MeV)
    @test size(dataset_mev) == size(dataset)
    @test dataset_mev[1, 2] ≈ dataset[1, 2] * MEV_PER_FM
    @test_throws ArgumentError build_dataset(sols; temperature_unit=:Kelvin)

    split_input = [
        0.2 1000.0 1.0
        0.3  900.0 0.5
        0.2  850.0 0.2
        0.3  800.0 0.1
    ]
    split_ranges = AtractorsQGP._split_trajectories(split_input)
    @test split_ranges == [1:2, 3:4]

    bad_solutions = [
        (t=[0.2, 0.21], u=[SVector(1000.0, 0.5), SVector(NaN, 0.4)]),
        (t=[0.2], u=[SVector(950.0, -0.2)]),
    ]
    cleaned = build_dataset(bad_solutions)
    @test size(cleaned, 1) == 2
    @test all(isfinite, cleaned)

    lle = run_LLE(model, u0, tspan; saveat=0.02)
    @test isfinite(lle)

    dim = estimate_dimension(dataset[:, 2:3])
    @test dim > 0


    mktempdir() do tmp
        csv_path = joinpath(tmp, "dataset.csv")
        h5_path = joinpath(tmp, "dataset.h5")
        jls_path = joinpath(tmp, "dataset.jls")

        save_dataset(csv_path, dataset)
        save_dataset(h5_path, dataset)
        save_dataset(jls_path, dataset)

        @test size(load_dataset(csv_path)) == size(dataset)
        @test size(load_dataset(h5_path)) == size(dataset)
        @test size(load_dataset(jls_path)) == size(dataset)
    end
end
