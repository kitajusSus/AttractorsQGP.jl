module AttractorsQGP

using LinearAlgebra
using Statistics
using Random
using StaticArrays
using OrdinaryDiffEqRosenbrock
using DifferentialEquations
using ProgressMeter
using CSV
using DataFrames
using HDF5
using Serialization
using Interpolations
using Distances
using NearestNeighbors
using KrylovKit
using SparseArrays
using LinearMaps
using MultivariateStats
using Polynomials
using Symbolics
using Latexify
using Lux
using Optimisers
using Zygote
using ForwardDiff
using ComponentArrays
using ColorSchemes
using CairoMakie
using GLMakie
using LaTeXStrings

abstract type AbstractHydroModel end

struct HydroParams{T <: Real}
    eta_over_s::T
    tau_pi::T
    lambda1::T
end

include("constants/units.jl")
include("models/brsss.jl")
include("models/mis.jl")
include("models/hjsw.jl")

include("equations/bjorken.jl")
include("solver/hydro_solver.jl")

include("simulation/initial_conditions.jl")
include("simulation/trajectories.jl")

include("analysis/pca.jl")
include("analysis/dimension.jl")
include("analysis/intrinsic_dimension.jl")
include("analysis/scale_dimensions.jl")
include("analysis/exponent.jl")
include("analysis/fit_polynomials.jl")
include("analysis/lle.jl")
include("analysis/lpca.jl")
include("analysis/plots.jl")

include("pinns/network.jl")
include("pinns/losses.jl")
include("pinns/training.jl")
include("pinns/pinn_solver.jl")
include("pinns/jacobian_analysis.jl")
include("pinns/data_jacobian_analysis.jl")

include("io/data_io.jl")
include("io/attractor.jl")
include("io/citation.jl")

export HydroParams, AbstractHydroModel, BRSSSModel, MISModel, HJSWModel, HJSWwModel
export HBARC_MEV_FM, MEV_PER_FM, FM_PER_MEV, to_temperature_unit, temperature_to_fm
export solve_hydro, generate_initial_conditions, generate_trajectories, build_dataset, run_main

export explained_variance_ratio_from_svd, normalize_minmax, run_pca, run_pca_kernel, get_tau_slice, run_pca_for_tau, run_pca_per_time, run_evolution_pca_workflow
export estimate_effective_dimension, estimate_dimension, scan_dimension_from_data
export estimate_lid, estimate_twonn, scan_intrinsic_dimensions, spectral_dimension, estimate_pinn_dimension
export estimate_scale_dimension, run_LLE
export compute_polynomial_lle

export lle, run_lle_per_time, run_lle_for_selected_taus, lle_spectrum, lle_spectrum_over_k, spectrum_statistics, scan_lle_spectrum
export dims, normalize_max, apply_normalization, swiss_roll, compute_lpca, dynamic_lpca_analysis, compute_lpca_entropy, compute_stable_lpca_collapse, compute_lpca_principal_angles
export transform_to_dimensionless, rescale_temperature, create_mixed_scaled, prepare_dataset_variants, load_hydro_dataset
export evaluate_k_pair, evaluate_k_dependency, compare_normalization_methods, test_coordinate_invariance
export analyze_dimension_distribution, analyze_tolerance_sensitivity, compute_parameterization_k_sweep
export PointwiseDimensionSlice, compute_pointwise_dimensions, compute_pointwise_pr
export compute_soft_weighted_dimension, scan_soft_weighted_dimension
export compute_local_pr_dimension, scan_local_pr_dimension

export PINNConfig, build_pinn_network, normalize_pinn_input, denormalize_pinn_output, pinn_predict
export train_pinn, PINNResult
export predict_trajectories_pinn, build_pinn_dataset, compare_pinn_ode, pinn_attractor_analysis
export pinn_jacobian, pinn_jacobian_full, d_eff_from_singular_values, pinn_deff_at, pinn_deff_scan, pinn_jacobian_scan, pinn_hydrodynamisation_time
export sample_ic_ensemble, fixed_transport_ic_ensemble, pinn_dimensionality_workflow
export normalize_generic, build_generic_network, compute_data_jacobian

export save_dataset_csv, load_dataset_csv, save_dataset_h5, load_dataset_h5, save_dataset_jls, load_dataset_jls, save_dataset, load_dataset
export attractor_data, build_attractor_interpolant, get_attractor_line, temperature_grid, interpolate_attractor_state, get_attractor_line_for_frame
export citation

export set_publication_theme, PLOT_KEYS, resolve_def, get_data, get_limits
export plot_phase_space_grid, plot_phase_space_grid_3d, plot_thermodynamics_evolution, plot_phase_space_evolution, plot_phase_space_evolution_3d, plot_attractor, plot_attractor_Aw_T
export plot_pca_summary, plot_pca_evr_over_time, plot_pca_bar_variance
export plot_twonn, plot_lid_dimension, plot_map_lpca, plot_local_pca, plot_local_pca_regularizations, plot_phase_space_lpca_dims, plot_phase_space_local_pca_dimensions
export plot_k_dependency_bands, plot_normalization_multipanel, plot_normalization_direct_overlay, plot_pr_normalization_direct_overlay
export plot_coordinate_invariance, plot_coordinate_invariance_single
export plot_dimension_distribution, plot_dimension_mean_single, plot_dimension_populations_single, plot_tolerance_sensitivity
export plot_colored_phase_space_slice_2d, plot_colored_phase_space_grid_2d, plot_colored_phase_space_grid_hjsw_projections, plot_colored_phase_space_slice_hjsw_3d, plot_colored_phase_space_grid_3d
export plot_parameterization_focus_tripanel, plot_parameterization_focus_2x2
export plot_soft_weighted_dimension, plot_soft_k_dependency, plot_soft_phase_space_slice_2d, plot_soft_phase_space_grid_2d, plot_soft_phase_space_slice_3d, plot_soft_phase_space_grid_3d
export plot_local_pr_dimension, plot_pr_k_dependency, plot_pr_phase_space_slice_2d, plot_pr_phase_space_grid_2d, plot_pr_phase_space_slice_3d, plot_pr_phase_space_grid_3d, plot_unified_lpca_comparison
export plot_lle_dim, plot_simulation_lle, plot_lle_spectrum_statistics, plot_lle_spectrum_statistics_grid, plot_lle_σ_spectrum, plot_lle_embedding, plot_lle_grid, plot_lle_grid_2d, plot_lle_grid_3d, plot_lle_results_for_taus, plot_lle_spectrum_analysis, plot_lle_spectrum_scan_analysis
export plot_pinn_deff_evolution, animate_pca_evolution

function run_main(
        model::AbstractHydroModel;
        n_points::Integer = 5000,
        tspan::Tuple{<:Real, <:Real} = (0.2, 1.2),
        T_range::Tuple{<:Real, <:Real} = (200.0, 1400.0),
        A_range::Tuple{<:Real, <:Real} = (-1.0, 7.0),
        temperature_unit::Symbol = :MeV,
        saveat::Union{Real, AbstractVector{<:Real}, Nothing} = 0.01,
        seed::Integer = 5,
        output_file::Union{String, Nothing} = nothing,
    )
    ics = generate_initial_conditions(n_points; T_range = T_range, A_range = A_range, temperature_unit = temperature_unit, seed = seed)
    solutions = generate_trajectories(model, ics, tspan; saveat = saveat)
    dataset = build_dataset(solutions)
    if output_file !== nothing
        save_dataset(output_file, dataset)
    end
    return (solutions = solutions, dataset = dataset)
end

function run_main(
        model::HJSWModel;
        n_points::Integer = 5000,
        tspan::Tuple{<:Real, <:Real} = (0.2, 1.2),
        T_range::Tuple{<:Real, <:Real} = (200.0, 1400.0),
        A_range::Tuple{<:Real, <:Real} = (-1.0, 10.0),
        B_range::Tuple{<:Real, <:Real} = (-2.0, 2.0),
        temperature_unit::Symbol = :MeV,
        include_w::Bool = false,
        saveat::Union{Real, AbstractVector{<:Real}, Nothing} = 0.01,
        seed::Integer = 5,
        output_file::Union{String, Nothing} = nothing,
    )

    ics = generate_initial_conditions(
        model,
        n_points;
        T_range = T_range,
        A_range = A_range,
        B_range = B_range,
        temperature_unit = temperature_unit,
        seed = seed,
    )
    solutions = generate_trajectories(model, ics, tspan; saveat = saveat)
    dataset = build_dataset(solutions; temperature_unit = temperature_unit, include_w = include_w)
    if output_file !== nothing
        save_dataset(output_file, dataset)
    end
    return (solutions = solutions, dataset = dataset)
end

function run_main(
        model::HJSWwModel;
        n_points::Integer = 5000,
        tspan::Tuple{<:Real, <:Real} = (0.2, 1.2),
        w_range::Union{Tuple{<:Real, <:Real}, Nothing} = (0.1, 5.0),
        T_range::Union{Tuple{<:Real, <:Real}, Nothing} = nothing,
        A_range::Tuple{<:Real, <:Real} = (-1.0, 10.0),
        B_range::Tuple{<:Real, <:Real} = (-2.0, 2.0),
        temperature_unit::Symbol = :MeV,
        saveat::Union{Real, AbstractVector{<:Real}, Nothing} = 0.01,
        seed::Integer = 5,
        output_file::Union{String, Nothing} = nothing,
    )

    ics = generate_initial_conditions(
        model,
        n_points;
        w_range = w_range,
        T_range = T_range,
        t0 = tspan[1],
        A_range = A_range,
        B_range = B_range,
        temperature_unit = temperature_unit,
        seed = seed,
    )
    solutions = generate_trajectories(model, ics, tspan; saveat = saveat)
    dataset = build_dataset(solutions; has_temperature = false)
    if output_file !== nothing
        save_dataset(output_file, dataset)
    end
    return (solutions = solutions, dataset = dataset)
end

end


