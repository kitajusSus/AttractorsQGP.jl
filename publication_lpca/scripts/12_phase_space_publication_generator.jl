"""
    Script 12: Dedicated Phase Space Generator for Publication
    Produces separated directories for MIS and HJSW:
    - publication_lpca/plots/phase_space/mis/
    - publication_lpca/plots/phase_space/hjsw/

    Generates:
    1. Standalone single-slice phase space plots for each tau in [0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80, 1.00, 2.50]:
       - Discrete local dimension dims()
       - Continuous participation ratio dimension pr()
       - Across standardizations: zscore, minmax, max, physical, dimensionless
       - Title on top tau = ... fm/c in large, clear font (titlesize = 24)
       - Subsampled: 10,000 points calculated, 5,000 points plotted
    2. Comprehensive 3x3 grids with wide subplot gaps (55px) and enlarged tau titles
    3. Complete LaTeX and Markdown documentation (README.md and FIGURES_INFO.tex) in each folder.
"""

using AttractorsQGP
using CairoMakie
using LaTeXStrings
using Printf
using Statistics
includet("write_functions_12.jl")
if !isdefined(Main, :PublicationLPCA)
    include(joinpath(@__DIR__, "..", "src", "PublicationLPCA.jl"))
end
using .PublicationLPCA
function run_phase_space_generator(;
        mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
        hjsw_dataset_path::AbstractString = isfile(joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_10000_points.h5")) ?
            joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_10000_points.h5") :
            joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
        base_output_dir::AbstractString = joinpath(@__DIR__, "..", "plots", "phase_space")
    )
    mis_dir = joinpath(base_output_dir, "mis")
    hjsw_dir = joinpath(base_output_dir, "hjsw")
    mkpath(mis_dir)
    mkpath(hjsw_dir)
    CairoMakie.activate!()

    println("=== Starting 12_phase_space_publication_generator ===")

    grid_taus = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 2.5]
    k_eval = 8
    tol_eval = 0.01
    set_publication_theme()

    colormap_choice = lpca_style

    mis_configs = [
        (name = "zscore", method = :zscore, return_norm = true, xlabel = L"T_z", ylabel = L"\mathcal{A}_z", dataset_type = :physical),
        (name = "physical", method = :none, return_norm = false, xlabel = L"T\,[\mathrm{fm}^{-1}]", ylabel = L"\mathcal{A}", dataset_type = :physical),
    ]

    hjsw_configs = [
        (name = "zscore", method = :zscore, return_norm = true, xlabel = L"T_z", ylabel = L"\mathcal{A}_z", zlabel = L"\mathcal{B}_z", dataset_type = :physical),
        (name = "physical", method = :none, return_norm = false, xlabel = L"T\,[\mathrm{MeV}]", ylabel = L"\mathcal{A}", zlabel = L"\mathcal{B}", dataset_type = :physical),
    ]

    println("\n[1/2] Processing Conformal MIS Phase Space (plots/phase_space/mis/)...")
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)

    for cfg in mis_configs
        ds = cfg.dataset_type == :dimensionless ? mis_variants.dimensionless : mis_variants.physical
        println("  - Generating MIS [$(cfg.name)]...")
        fig_grid_dims = plot_colored_phase_space_grid_2d(
            ds, grid_taus;
            feature_indices = [2, 3], x_label = cfg.xlabel, y_label = cfg.ylabel,
            k_neighbor = k_eval, tolerance = tol_eval, normalize_method = cfg.method,
            return_normalized = cfg.return_norm, colormap = colormap_choice,
            n_calc = 10_000, n_plot = 5_000
        )
        save(joinpath(mis_dir, "grid_mis_$(cfg.name)_dims.pdf"), fig_grid_dims)

        fig_grid_pr = plot_pr_phase_space_grid_2d(
            ds, grid_taus;
            feature_indices = [2, 3], x_label = cfg.xlabel, y_label = cfg.ylabel,
            colormap = colormap_choice, color_limits = (1.0, 2.0),
            normalize_method = cfg.method, return_normalized = cfg.return_norm,
            k = k_eval, n_calc = 10_000, n_plot = 5_000
        )
        save(joinpath(mis_dir, "grid_mis_$(cfg.name)_pr.pdf"), fig_grid_pr)

        fig_grid_hist = plot_pr_dimension_histogram_grid(
            ds, grid_taus;
            feature_indices = [2, 3], k_neighbor = k_eval,
            normalize_method = cfg.method, colormap = colormap_choice,
            color_limits = (1.0, 2.0), n_calc = 10_000
        )
        save(joinpath(mis_dir, "grid_mis_$(cfg.name)_pr_hist.pdf"), fig_grid_hist)

        for tau in grid_taus
            tau_str = @sprintf("%.2f", tau)
            slice_data = compute_pointwise_dimensions(
                ds, tau;
                feature_indices = [2, 3], k_neighbor = k_eval, tolerance = tol_eval,
                normalize_method = cfg.method, return_normalized = cfg.return_norm,
                max_points = 10_000
            )

            fig_slice_dims = plot_colored_phase_space_slice_2d(
                slice_data;
                x_col_idx = 1, y_col_idx = 2,
                x_label = cfg.xlabel, y_label = cfg.ylabel,
                colormap = colormap_choice, n_plot = 5_000
            )
            save(joinpath(mis_dir, "mis_$(cfg.name)_slice_tau_$(tau_str)_dims.pdf"), fig_slice_dims)

            fig_slice_pr = plot_pr_phase_space_slice_2d(
                ds, tau;
                feature_indices = [2, 3], x_label = cfg.xlabel, y_label = cfg.ylabel,
                colormap = colormap_choice, color_limits = (1.0, 2.0),
                normalize_method = cfg.method, return_normalized = cfg.return_norm,
                k = k_eval, n_calc = 10_000, n_plot = 5_000
            )
            save(joinpath(mis_dir, "mis_$(cfg.name)_slice_tau_$(tau_str)_pr.pdf"), fig_slice_pr)

            fig_slice_hist = plot_pr_dimension_histogram(
                ds, tau;
                feature_indices = [2, 3], k_neighbor = k_eval,
                normalize_method = cfg.method, colormap = colormap_choice,
                color_limits = (1.0, 2.0), n_calc = 10_000
            )
            save(joinpath(mis_dir, "mis_$(cfg.name)_slice_tau_$(tau_str)_pr_hist.pdf"), fig_slice_hist)
        end
    end

    write_mis_docs(mis_dir, grid_taus, mis_configs)

    # Part 2: HJSW Model (3D Phase Space)
    println("\n[2/2] Processing HJSW Phase Space (plots/phase_space/hjsw/)...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)

    for cfg in hjsw_configs
        ds = cfg.dataset_type == :dimensionless ? hjsw_variants.dimensionless : hjsw_variants.physical
        println("  - Generating HJSW [$(cfg.name)]...")

        fig_grid_dims = plot_colored_phase_space_grid_3d(
            ds, grid_taus;
            feature_indices = [2, 3, 4], x_label = cfg.xlabel, y_label = cfg.ylabel, z_label = cfg.zlabel,
            k_neighbor = k_eval, tolerance = tol_eval, normalize_method = cfg.method,
            return_normalized = cfg.return_norm, colormap = colormap_choice,
            n_calc = 10_000, n_plot = 5_000
        )
        save(joinpath(hjsw_dir, "grid_hjsw_$(cfg.name)_dims.pdf"), fig_grid_dims)

        fig_grid_pr = plot_pr_phase_space_grid_3d(
            ds, grid_taus;
            feature_indices = [2, 3, 4], x_label = cfg.xlabel, y_label = cfg.ylabel, z_label = cfg.zlabel,
            colormap = colormap_choice, color_limits = (1.0, 3.0),
            normalize_method = cfg.method, return_normalized = cfg.return_norm,
            k = k_eval, n_calc = 10_000, n_plot = 5_000
        )
        save(joinpath(hjsw_dir, "grid_hjsw_$(cfg.name)_pr.pdf"), fig_grid_pr)

        fig_grid_hist = plot_pr_dimension_histogram_grid(
            ds, grid_taus;
            feature_indices = [2, 3, 4], k_neighbor = k_eval,
            normalize_method = cfg.method, colormap = colormap_choice,
            color_limits = (1.0, 3.0), n_calc = 10_000
        )
        save(joinpath(hjsw_dir, "grid_hjsw_$(cfg.name)_pr_hist.pdf"), fig_grid_hist)

        for tau in grid_taus
            tau_str = @sprintf("%.2f", tau)
            slice_data = compute_pointwise_dimensions(
                ds, tau;
                feature_indices = [2, 3, 4], k_neighbor = k_eval, tolerance = tol_eval,
                normalize_method = cfg.method, return_normalized = cfg.return_norm,
                max_points = 10_000
            )

            fig_slice_dims = plot_colored_phase_space_slice_hjsw_3d(
                slice_data;
                x_label = cfg.xlabel, y_label = cfg.ylabel, z_label = cfg.zlabel,
                colormap = colormap_choice, n_plot = 5_000
            )
            save(joinpath(hjsw_dir, "hjsw_$(cfg.name)_slice_tau_$(tau_str)_dims.pdf"), fig_slice_dims)

            fig_slice_pr = plot_pr_phase_space_slice_3d(
                ds, tau;
                feature_indices = [2, 3, 4], x_label = cfg.xlabel, y_label = cfg.ylabel, z_label = cfg.zlabel,
                colormap = colormap_choice, color_limits = (1.0, 3.0),
                normalize_method = cfg.method, return_normalized = cfg.return_norm,
                k = k_eval, n_calc = 10_000, n_plot = 5_000
            )
            save(joinpath(hjsw_dir, "hjsw_$(cfg.name)_slice_tau_$(tau_str)_pr.pdf"), fig_slice_pr)

            fig_slice_hist = plot_pr_dimension_histogram(
                ds, tau;
                feature_indices = [2, 3, 4], k_neighbor = k_eval,
                normalize_method = cfg.method, colormap = colormap_choice,
                color_limits = (1.0, 3.0), n_calc = 10_000
            )
            save(joinpath(hjsw_dir, "hjsw_$(cfg.name)_slice_tau_$(tau_str)_pr_hist.pdf"), fig_slice_hist)
        end
    end

    write_hjsw_docs(hjsw_dir, grid_taus, hjsw_configs)

    return println("=== 12_phase_space_publication_generator Completed Successfully! ===")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_phase_space_generator()
end
