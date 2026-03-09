module AtractorsQGP


abstract type AbstractHydroModel end
"""
Hydrodynamic transport parameters.
"""
struct HydroParams{T<:Real}
    eta_over_s::T
    tau_pi::T
    lambda1::T
end

include("models/brsss.jl")
include("models/mis.jl")


include("equations/bjorken.jl")

include("solver/hydro_solver.jl")

include("simulation/initial_conditions.jl")
include("simulation/trajectories.jl")

include("analysis/lle.jl")
include("analysis/pca.jl")
include("analysis/dimension.jl")
include("analysis/plots.jl")

include("io/data_io.jl")

export HydroParams, AbstractHydroModel, BRSSSModel, MISModel
export solve_hydro, generate_initial_conditions, generate_trajectories
export build_dataset, run_LLE, run_pca, estimate_dimension
export save_dataset_csv, load_dataset_csv, save_dataset_h5, load_dataset_h5
export save_dataset_jls, load_dataset_jls, save_dataset, load_dataset
export set_publication_theme, resolve_def, get_data
export plot_phase_space_grid, plot_thermodynamics_evolution, plot_pca_summary
export run_pipeline





"""
Run full simulation
"""
function run_pipeline(
    model::AbstractHydroModel;
    n_points::Integer=200,
    tspan::Tuple{<:Real,<:Real}=(0.22, 1.0),
    T_range::Tuple{<:Real,<:Real}=(400.0, 2500.0),
    A_range::Tuple{<:Real,<:Real}=(-8.0, 20.0),
    saveat::Real=0.01,
)
    ics = generate_initial_conditions(n_points; T_range=T_range, A_range=A_range)
    solutions = generate_trajectories(model, ics, tspan; saveat=saveat)
    dataset = build_dataset(solutions)
    lle = run_LLE(model, ics[1], tspan; saveat=saveat)
    dim = estimate_dimension(dataset[:, 2:3])
    return (solutions=solutions, dataset=dataset, lle=lle, dimension=dim)
end

end
