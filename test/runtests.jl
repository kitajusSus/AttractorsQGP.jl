using Test
include("../src/AtractorsQGP.jl")
using .AtractorsQGP
using StaticArrays

@testset "AtractorsQGP refactor API" begin
    model = BRSSSModel()
    u0 = [800.0, 0.5]
    tspan = (0.22, 0.5)

    sol = solve_hydro(model, u0, tspan; saveat=0.02)
    @test !isempty(sol.t)

    ics = generate_initial_conditions(5; T_range=(700.0, 900.0), A_range=(-1.0, 1.0))
    @test eltype(ics) == SVector{2,Float64}

    sols = generate_trajectories(model, ics, tspan; saveat=0.02, parallel=:serial)
    @test length(sols) == 5
    @test eltype(sols) != Any

    dataset = build_dataset(sols)
    @test size(dataset, 2) == 3

    lle = run_LLE(model, u0, tspan; saveat=0.02)
    @test isfinite(lle)

    dim = estimate_dimension(dataset[:, 2:3])
    @test dim > 0
end
