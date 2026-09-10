"""
    Script 08: Comprehensive Participation Ratio (PR) Dimensionality Analysis
    Calculates parameter-free local and ensemble Participation Ratio dimensions:
        d_PR = (Tr C)^2 / Tr(C^2) = (sum lambda_i)^2 / sum (lambda_i^2)
    and compares them directly with Soft-Weighted LPCA and Hard LPCA across proper time tau.

    Resolves dimensionality evolution:
    - Conformal MIS: 2D initial state (T, A) collapsing to 1D universal attractor
    - HJSW Model: 3D initial state (T, A, B) relaxing to 2D intermediate sheet, then collapsing to 1D attractor
"""

using AttractorsQGP
using CairoMakie
using LaTeXStrings
using Printf
using Statistics
using CSV
using DataFrames

function run_participation_ratio_experiment(;
    mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
    hjsw_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
    output_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "participation_ratio"),
    results_directory::AbstractString = joinpath(@__DIR__, "..", "results")
)
    mkpath(output_directory)
    mkpath(results_directory)
    CairoMakie.activate!()

    println("=================================================================")
    println("   STARTING EXPERIMENT 08: COMPREHENSIVE PARTICIPATION RATIO     ")
    println("=================================================================")

    tau_grid = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.5, 5.0, 7.5, 10.0]
    k_fixed = 20
    k_test_values = [10, 20, 40, 80, 160]
    grid_taus = [0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.5, 2.5]
    slice_taus = [0.25, 0.65, 2.50]

    # =========================================================================
    # Part 1: Conformal MIS Model (Initial 2D: T, A)
    # =========================================================================
    println("\n[1/4] Processing Conformal MIS Model (Initial 2D Phase Space)...")
    mis_raw = load_hydro_dataset(mis_dataset_path)

    # 1A. Direct PR vs Soft LPCA comparison
    println("  - Computing PR vs Soft LPCA...")
    mis_pr_scan = scan_local_pr_dimension(mis_raw, tau_grid; feature_indices = [2, 3], normalize_method = :max, k = k_fixed)
    mis_soft_scan = scan_soft_weighted_dimension(mis_raw, tau_grid; feature_indices = [2, 3], normalize_method = :max, k = k_fixed, tol = 0.02, delta = 0.005)

    fig_mis_comp = plot_local_pr_dimension(mis_pr_scan; soft_scan = mis_soft_scan)
    save(joinpath(output_directory, "pr_vs_soft_comparison_mis.pdf"), fig_mis_comp)

    # 1B. K-Dependency Sweep
    println("  - Computing K-dependency sweep for MIS...")
    fig_mis_k = plot_pr_k_dependency(mis_raw, k_test_values, tau_grid; feature_indices = [2, 3], normalize_method = :max)
    save(joinpath(output_directory, "pr_k_dependency_mis.pdf"), fig_mis_k)

    # 1C. Phase Space Slices at tau = 0.25, 0.65, 2.50
    println("  - Generating 2D phase space slices for MIS...")
    for tau_sample in slice_taus
        fig_s = plot_pr_phase_space_slice_2d(
            mis_raw, tau_sample;
            feature_indices = [2, 3],
            x_label = L"T\,[\mathrm{fm}^{-1}]",
            y_label = L"\mathcal{A}",
            color_limits = (1.0, 2.0),
            k = k_fixed
        )
        tau_tag = @sprintf("%.2f", tau_sample)
        save(joinpath(output_directory, "pr_phase_space_slice_mis_tau_$(tau_tag).pdf"), fig_s)
    end

    # 1D. 9-Panel Phase Space Grid
    println("  - Generating 9-panel phase space grid for MIS...")
    fig_mis_grid = plot_pr_phase_space_grid_2d(
        mis_raw, grid_taus;
        feature_indices = [2, 3],
        x_label = L"T\,[\mathrm{fm}^{-1}]",
        y_label = L"\mathcal{A}",
        color_limits = (1.0, 2.0),
        k = k_fixed
    )
    save(joinpath(output_directory, "pr_phase_space_grid_mis.pdf"), fig_mis_grid)

    # =========================================================================
    # Part 2: HJSW Model (Initial 3D: T, A, B)
    # =========================================================================
    println("\n[2/4] Processing HJSW Model (Initial 3D Phase Space)...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)

    # 2A. Direct PR vs Soft LPCA comparison
    println("  - Computing PR vs Soft LPCA for HJSW...")
    hjsw_pr_scan = scan_local_pr_dimension(hjsw_raw, tau_grid; feature_indices = [2, 3, 4], normalize_method = :max, k = k_fixed)
    hjsw_soft_scan = scan_soft_weighted_dimension(hjsw_raw, tau_grid; feature_indices = [2, 3, 4], normalize_method = :max, k = k_fixed, tol = 0.02, delta = 0.005)

    fig_hjsw_comp = plot_local_pr_dimension(hjsw_pr_scan; soft_scan = hjsw_soft_scan)
    save(joinpath(output_directory, "pr_vs_soft_comparison_hjsw.pdf"), fig_hjsw_comp)

    # 2B. K-Dependency Sweep
    println("  - Computing K-dependency sweep for HJSW...")
    fig_hjsw_k = plot_pr_k_dependency(hjsw_raw, k_test_values, tau_grid; feature_indices = [2, 3, 4], normalize_method = :max)
    save(joinpath(output_directory, "pr_k_dependency_hjsw.pdf"), fig_hjsw_k)

    # 2C. 3D Phase Space Slices at tau = 0.25, 0.65, 2.50
    println("  - Generating 3D phase space slices for HJSW...")
    for tau_sample in slice_taus
        fig_s3d = plot_pr_phase_space_slice_3d(
            hjsw_raw, tau_sample;
            feature_indices = [2, 3, 4],
            color_limits = (1.0, 3.0),
            k = k_fixed
        )
        tau_tag = @sprintf("%.2f", tau_sample)
        save(joinpath(output_directory, "pr_phase_space_slice_hjsw_3d_tau_$(tau_tag).pdf"), fig_s3d)
    end

    # 2D. 2D Projection Grid on (A, B) plane
    println("  - Generating 2D projection grid on (A, B) for HJSW...")
    fig_hjsw_grid_ab = plot_pr_phase_space_grid_2d(
        hjsw_raw, grid_taus;
        feature_indices = [3, 4],
        x_label = L"\mathcal{A}",
        y_label = L"\mathcal{B}",
        color_limits = (1.0, 3.0),
        k = k_fixed
    )
    save(joinpath(output_directory, "pr_phase_space_grid_hjsw_projections_AB.pdf"), fig_hjsw_grid_ab)

    # =========================================================================
    # Part 3: Unified 3-Method Comparison (Side-by-Side MIS vs HJSW)
    # =========================================================================
    println("\n[3/4] Generating Unified 3-Method Comparison (Hard vs Soft vs PR)...")
    fig_unified = plot_unified_lpca_comparison(mis_raw, hjsw_raw, tau_grid; k = k_fixed)
    save(joinpath(output_directory, "pr_unified_3methods_comparison.pdf"), fig_unified)

    # =========================================================================
    # Part 4: CSV Summary Export (Including Multi-K Table)
    # =========================================================================
    println("\n[4/4] Exporting comprehensive CSV summary table...")
    df_results = DataFrame(
        tau = tau_grid,
        mis_pr_mean_k20 = mis_pr_scan.mean_dims,
        mis_pr_std_k20 = mis_pr_scan.std_dims,
        mis_soft_mean = mis_soft_scan.mean_dims,
        hjsw_pr_mean_k20 = hjsw_pr_scan.mean_dims,
        hjsw_pr_std_k20 = hjsw_pr_scan.std_dims,
        hjsw_soft_mean = hjsw_soft_scan.mean_dims
    )
    results_csv = joinpath(results_directory, "participation_ratio_summary.csv")
    CSV.write(results_csv, df_results)
    println("  Summary table written to: $(results_csv)")

    println("\n=================================================================")
    println("  ALL PARTICIPATION RATIO FIGURES AND DATA GENERATED SUCCESSFULLY!")
    println("  Saved to: $(output_directory)")
    println("=================================================================")
    return (mis_pr = mis_pr_scan, hjsw_pr = hjsw_pr_scan)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_participation_ratio_experiment()
end
