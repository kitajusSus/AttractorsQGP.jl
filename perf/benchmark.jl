using BenchmarkTools

# include(".../src/AtractorsQGP.jl")
using .AtractorsQGP

println("Running baseline benchmark...")
println("model")
model = BRSSSModel()
println("initial conditions")
ICs = generate_initial_conditions(256; T_range=(400.0, 2500.0), A_range=(-8.0, 20.0))
τspan = (0.22, 1.0)
println("trajectories")
@btime generate_trajectories($model, $ICs, $τspan; saveat=0.02)
println("trajectories (allocated)")
@show @allocated generate_trajectories(model, ICs, τspan; saveat=0.02)
