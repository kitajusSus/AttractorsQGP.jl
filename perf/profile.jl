using Profile

include("../src/AtractorsQGP.jl")
using .AtractorsQGP

model = BRSSSModel()
ICs = generate_initial_conditions(256)
τspan = (0.22, 1.0)

Profile.clear()
@profile generate_trajectories(model, ICs, τspan; saveat=0.02)
Profile.print()
