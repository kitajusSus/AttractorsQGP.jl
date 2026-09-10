"""
    Script 08: Participation Ratio (PR) Dimensionality Analysis
    Calculates parameter-free local and ensemble Participation Ratio dimensions:
        d_PR = (Tr C)^2 / Tr(C^2) = (sum lambda_i)^2 / sum (lambda_i^2)
    and compares them directly with Soft-Weighted LPCA across proper time tau.
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
    println("   STARTING EXPERIMENT 08: PARTICIPATION RATIO (PR) DIMENSION    ")
    println("=================================================================")

    tau_grid = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.5, 5.0, 7.5, 10.0]
    k_fixed = 20

    # 1. MIS Model
    println("\n[1/3] Computing Participation Ratio for Conformal MIS...")
    mis_raw = load_hydro_dataset(mis_dataset_path)

    mis_pr_scan = scan_local_pr_dimension(
        mis_raw,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        k = k_fixed
    )

    mis_soft_scan = scan_soft_weighted_dimension(
        mis_raw,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        k = k_fixed,
        tol = 0.02,
        delta = 0.005
    )

    fig_mis = plot_local_pr_dimension(mis_pr_scan; soft_scan = mis_soft_scan)
    save(joinpath(output_directory, "pr_vs_soft_comparison_mis.pdf"), fig_mis)

    fig_slice_mis = plot_pr_phase_space_slice_2d(
        mis_raw,
        0.65;
        feature_indices = [2, 3],
        x_label = L"T\,[\mathrm{fm}^{-1}]",
        y_label = L"\mathcal{A}",
        color_limits = (1.0, 2.0),
        k = k_fixed
    )
    save(joinpath(output_directory, "pr_phase_space_slice_mis_tau_0.65.pdf"), fig_slice_mis)

    # 2. HJSW Model
    println("\n[2/3] Computing Participation Ratio for HJSW Model...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)

    hjsw_pr_scan = scan_local_pr_dimension(
        hjsw_raw,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        k = k_fixed
    )

    hjsw_soft_scan = scan_soft_weighted_dimension(
        hjsw_raw,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        k = k_fixed,
        tol = 0.02,
        delta = 0.005
    )

    fig_hjsw = plot_local_pr_dimension(hjsw_pr_scan; soft_scan = hjsw_soft_scan)
    save(joinpath(output_directory, "pr_vs_soft_comparison_hjsw.pdf"), fig_hjsw)

    fig_slice_hjsw_3d = plot_pr_phase_space_slice_3d(
        hjsw_raw,
        0.65;
        feature_indices = [2, 3, 4],
        color_limits = (1.0, 3.0),
        k = k_fixed
    )
    save(joinpath(output_directory, "pr_phase_space_slice_hjsw_3d_tau_0.65.pdf"), fig_slice_hjsw_3d)

    # 3. CSV Results Export
    println("\n[3/3] Exporting CSV summary table...")
    df_results = DataFrame(
        tau = tau_grid,
        mis_pr_mean = mis_pr_scan.mean_dims,
        mis_pr_std = mis_pr_scan.std_dims,
        mis_soft_mean = mis_soft_scan.mean_dims,
        hjsw_pr_mean = hjsw_pr_scan.mean_dims,
        hjsw_pr_std = hjsw_pr_scan.std_dims,
        hjsw_soft_mean = hjsw_soft_scan.mean_dims
    )
    results_csv = joinpath(results_directory, "participation_ratio_summary.csv")
    CSV.write(results_csv, df_results)
    println("  Summary table written to: $(results_csv)")

    println("\n=================================================================")
    println("  PARTICIPATION RATIO EXPERIMENT COMPLETED SUCCESSFULLY!")
    println("  Figures saved to: $(output_directory)")
    println("=================================================================")
    return (mis_pr = mis_pr_scan, hjsw_pr = hjsw_pr_scan)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_participation_ratio_experiment()
end
