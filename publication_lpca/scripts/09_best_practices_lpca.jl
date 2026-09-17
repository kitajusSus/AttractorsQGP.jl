"""
    Script 09: Best Practices in Local PCA for Attractors in QGP
    Generates publication figures comparing:
    1. Discrete dimension coloring dims() [separate PDF/PNG]
    2. Continuous dimension coloring pr() (Participation Ratio) [separate PDF/PNG]
    Across multiple data standardization/normalization schemes (:none, :max, :minmax, :zscore, dimensionless).
    All multi-panel grids evaluated at identical proper time moments tau in [0.25, 2.5] fm/c.
    All mean dimension evolution trajectories truncated at tau = 6.0 fm/c.
"""

using AttractorsQGP
using CairoMakie
using Printf
using Statistics

if !isdefined(Main, :PublicationLPCA)
    include(joinpath(@__DIR__, "..", "src", "PublicationLPCA.jl"))
end
using .PublicationLPCA

function run_best_practices_lpca_experiment(;
    mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
    hjsw_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
    output_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "best_practices_lpca"),
    results_directory::AbstractString = joinpath(@__DIR__, "..", "results")
)
    mkpath(output_directory)
    mkpath(results_directory)
    CairoMakie.activate!()

    println("=== Starting 09_best_practices_lpca Experiment ===")

    # Synchronized 9-panel time grid starting strictly at initial moment tau = 0.20 fm/c
    grid_taus = Float64[0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80, 1.00, 2.50]
    tau_evolution_grid = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 6.0]

    k_eval = 24
    tol_eval = 0.01
    colormap_choice = cgrad(:managua100, 10, categorical = true)

    # Normalization configurations to evaluate
    mis_norm_configs = [
        (name = "physical", method = :none, return_norm = false, xlabel = L"T\,[\mathrm{fm}^{-1}]", ylabel = L"\mathcal{A}", dataset_type = :physical),
        (name = "max", method = :max, return_norm = true, xlabel = L"T / T_{\mathrm{max}}", ylabel = L"\mathcal{A} / \mathcal{A}_{\mathrm{max}}", dataset_type = :physical),
        (name = "minmax", method = :minmax, return_norm = true, xlabel = L"T_{\mathrm{minmax}}", ylabel = L"\mathcal{A}_{\mathrm{minmax}}", dataset_type = :physical),
        (name = "zscore", method = :zscore, return_norm = true, xlabel = L"T_z", ylabel = L"A_z", dataset_type = :physical),
        (name = "dimensionless", method = :max, return_norm = false, xlabel = L"w = \tau T", ylabel = L"\mathcal{A}", dataset_type = :dimensionless)
    ]

    hjsw_norm_configs = [
        (name = "physical", method = :none, return_norm = false, xlabel = L"T\,[\mathrm{MeV}]", ylabel = L"\mathcal{A}", zlabel = L"\mathcal{B}", dataset_type = :physical),
        (name = "max", method = :max, return_norm = true, xlabel = L"T / T_{\mathrm{max}}", ylabel = L"\mathcal{A} / \mathcal{A}_{\mathrm{max}}", zlabel = L"\mathcal{B} / \mathcal{B}_{\mathrm{max}}", dataset_type = :physical),
        (name = "minmax", method = :minmax, return_norm = true, xlabel = L"T_{\mathrm{minmax}}", ylabel = L"\mathcal{A}_{\mathrm{minmax}}", zlabel = L"\mathcal{B}_{\mathrm{minmax}}", dataset_type = :physical),
        (name = "zscore", method = :zscore, return_norm = true, xlabel = L"T_z", ylabel = L"A_z", zlabel = L"B_z", dataset_type = :physical),
        (name = "dimensionless", method = :max, return_norm = false, xlabel = L"w = \tau T", ylabel = L"\mathcal{A}", zlabel = L"\mathcal{B}", dataset_type = :dimensionless)
    ]

    # =========================================================================
    # Part 1: Conformal MIS Model (2D Phase Space)
    # =========================================================================
    println("\n[1/3] Processing Conformal MIS (2D Phase Space)...")
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)

    for cfg in mis_norm_configs
        ds = cfg.dataset_type == :dimensionless ? mis_variants.dimensionless : mis_variants.physical
        println("  - Generating MIS [$(cfg.name)] Grids: dims() and pr()...")

        # 1A. Separate File for dims()
        fig_dims = plot_colored_phase_space_grid_2d(
            ds,
            grid_taus;
            feature_indices = [2, 3],
            x_label = cfg.xlabel,
            y_label = cfg.ylabel,
            k_neighbor = k_eval,
            tolerance = tol_eval,
            normalize_method = cfg.method,
            return_normalized = cfg.return_norm,
            colormap = colormap_choice
        )
        save(joinpath(output_directory, "grid_mis_$(cfg.name)_dims.pdf"), fig_dims)

        # 1B. Separate File for pr()
        fig_pr = plot_pr_phase_space_grid_2d(
            ds,
            grid_taus;
            feature_indices = [2, 3],
            x_label = cfg.xlabel,
            y_label = cfg.ylabel,
            colormap = colormap_choice,
            color_limits = (1.0, 2.0),
            normalize_method = cfg.method,
            return_normalized = cfg.return_norm,
            k = k_eval
        )
        save(joinpath(output_directory, "grid_mis_$(cfg.name)_pr.pdf"), fig_pr)
    end

    # 1C. Mean local dimension vs tau comparison for MIS
    println("  - Generating MIS Mean Dimension vs tau (tau <= 6 fm/c)...")
    norm_methods_eval = [:none, :max, :minmax, :zscore]
    fig_mis_ev_dims = plot_normalization_direct_overlay(
        mis_raw,
        norm_methods_eval,
        k_eval,
        tau_evolution_grid;
        feature_indices = [2, 3],
        tolerance = tol_eval,
        tau_max = 6.0
    )
    save(joinpath(output_directory, "mean_dimension_vs_tau_mis_dims.pdf"), fig_mis_ev_dims)

    fig_mis_ev_pr = plot_pr_normalization_direct_overlay(
        mis_raw,
        norm_methods_eval,
        k_eval,
        tau_evolution_grid;
        feature_indices = [2, 3],
        tau_max = 6.0
    )
    save(joinpath(output_directory, "mean_dimension_vs_tau_mis_pr.pdf"), fig_mis_ev_pr)

    # =========================================================================
    # Part 2: HJSW Model (3D Phase Space with Fixed Camera & No Overlap)
    # =========================================================================
    println("\n[2/3] Processing HJSW Model (3D Phase Space)...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)

    for cfg in hjsw_norm_configs
        ds = cfg.dataset_type == :dimensionless ? hjsw_variants.dimensionless : hjsw_raw
        println("  - Generating HJSW 3D [$(cfg.name)] Grids: dims() and pr()...")

        # 2A. Separate File for dims()
        fig_hjsw_dims = plot_colored_phase_space_grid_3d(
            ds,
            grid_taus;
            feature_indices = [2, 3, 4],
            x_label = cfg.xlabel,
            y_label = cfg.ylabel,
            z_label = cfg.zlabel,
            k_neighbor = k_eval,
            tolerance = tol_eval,
            normalize_method = cfg.method,
            return_normalized = cfg.return_norm,
            colormap = colormap_choice,
            azimuth = 1.2,
            elevation = 0.20,
            xreversed = true
        )
        save(joinpath(output_directory, "grid_hjsw_$(cfg.name)_dims.pdf"), fig_hjsw_dims)

        # 2B. Separate File for pr()
        fig_hjsw_pr = plot_pr_phase_space_grid_3d(
            ds,
            grid_taus;
            feature_indices = [2, 3, 4],
            x_label = cfg.xlabel,
            y_label = cfg.ylabel,
            z_label = cfg.zlabel,
            colormap = colormap_choice,
            color_limits = (1.0, 3.0),
            normalize_method = cfg.method,
            return_normalized = cfg.return_norm,
            k = k_eval,
            azimuth = 1.2,
            elevation = 0.20,
            xreversed = true
        )
        save(joinpath(output_directory, "grid_hjsw_$(cfg.name)_pr.pdf"), fig_hjsw_pr)
    end

    # 2C. Mean local dimension vs tau comparison for HJSW
    println("  - Generating HJSW Mean Dimension vs tau (tau <= 6 fm/c)...")
    fig_hjsw_ev_dims = plot_normalization_direct_overlay(
        hjsw_raw,
        norm_methods_eval,
        k_eval,
        tau_evolution_grid;
        feature_indices = [2, 3, 4],
        tolerance = tol_eval,
        tau_max = 6.0
    )
    save(joinpath(output_directory, "mean_dimension_vs_tau_hjsw_dims.pdf"), fig_hjsw_ev_dims)

    fig_hjsw_ev_pr = plot_pr_normalization_direct_overlay(
        hjsw_raw,
        norm_methods_eval,
        k_eval,
        tau_evolution_grid;
        feature_indices = [2, 3, 4],
        tau_max = 6.0
    )
    save(joinpath(output_directory, "mean_dimension_vs_tau_hjsw_pr.pdf"), fig_hjsw_ev_pr)

    println("\n=== 09_best_practices_lpca Completed Successfully! ===")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_best_practices_lpca_experiment()
end
