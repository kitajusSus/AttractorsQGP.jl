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

include("constants/units.jl")

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
export HBARC_MEV_FM, MEV_PER_FM, FM_PER_MEV, to_temperature_unit, temperature_to_fm
export solve_hydro, generate_initial_conditions, generate_trajectories
export build_dataset,run_pca, run_pca_kernel, run_pca_per_time, estimate_dimension
export explained_variance_ratio_from_svd,
       normalize_minmax,
       run_pca,
       run_pca_kernel,
       get_tau_slice,
       run_pca_for_tau,
       run_pca_per_time,
       run_evolution_pca_workflow
export lle
export save_dataset_csv, load_dataset_csv, save_dataset_h5, load_dataset_h5
export save_dataset_jls, load_dataset_jls, save_dataset, load_dataset
export set_publication_theme, resolve_def, get_data
export plot_phase_space_grid, plot_thermodynamics_evolution, plot_pca_summary, plot_pca_evr_over_time
export run_main





"""
Run full simulation generating important data
"""
function run_main(
    model::AbstractHydroModel;
    n_points::Integer=1000,
    tspan::Tuple{<:Real,<:Real}=(0.22, 1.2),
    T_range::Tuple{<:Real,<:Real}=(400.0, 2500.0),
    A_range::Tuple{<:Real,<:Real}=(-8.0, 20.0),
    saveat::Union{Real, AbstractVector{<:Real}, Nothing}=0.01,
    seed::Integer=5,
)
    ics = generate_initial_conditions(n_points; T_range=T_range, A_range=A_range, seed=seed)
    solutions = generate_trajectories(model, ics, tspan; saveat=saveat)
    dataset = build_dataset(solutions)
    return (solutions=solutions, dataset=dataset)
end

end
