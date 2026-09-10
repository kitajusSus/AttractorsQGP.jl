"""
    Script 07: Soft-Weighted Local PCA (Sigmoidal Spectral Cutoff & Boundary Reliability)
    Runs systematic evaluations of the new Soft-Weighted LPCA algorithm:
    1. Direct comparison: Standard Hard-threshold LPCA vs. Soft-Weighted LPCA ⟨d⟩_W(τ)
    2. Stability across nearest-neighbor count K (absence of threshold popping / boundary bias)
    3. Tolerance and sigmoidal transition width (Δ) sensitivity
    4. Continuous local dimension mapping in 2D and 3D phase spaces
"""

using AttractorsQGP
using CairoMakie
using LaTeXStrings
using Printf
using Statistics
using CSV
using DataFrames

function run_soft_weighted_lpca_experiment(;
    mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
    hjsw_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
    output_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "soft_weighted"),
    results_directory::AbstractString = joinpath(@__DIR__, "..", "results")
)
    mkpath(output_directory)
    mkpath(results_directory)
    CairoMakie.activate!()

    println("=================================================================")
    println("      STARTING EXPERIMENT 07: SOFT-WEIGHTED LOCAL PCA           ")
    println("=================================================================")

    tau_grid = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.5, 5.0, 7.5, 10.0]
    k_fixed = 20
    tol_fixed = 0.02
    delta_fixed = 0.005

    # =========================================================================
    # Part 1: Conformal MIS Model Evaluation
    # =========================================================================
    println("\n[1/4] Running Soft-Weighted LPCA on Conformal MIS Model...")
    mis_raw = load_hydro_dataset(mis_dataset_path)

    # 1A. Direct Comparison: Hard vs Soft-Weighted LPCA
    println("  - Computing Hard vs. Soft-Weighted comparison...")
    mis_scan = scan_soft_weighted_dimension(
        mis_raw,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        k = k_fixed,
        tol = tol_fixed,
        delta = delta_fixed
    )

    fig_mis_compare = plot_soft_weighted_dimension(mis_scan; compare_hard = true)
    save(joinpath(output_directory, "soft_vs_hard_comparison_mis.pdf"), fig_mis_compare)

    # 1B. K-Dependency of Soft-Weighted LPCA
    println("  - Computing Soft-Weighted K-dependency...")
    k_test_values = [6, 12, 24, 48]
    fig_mis_k = plot_soft_k_dependency(
        mis_raw,
        k_test_values,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        tol = tol_fixed,
        delta = delta_fixed
    )
    save(joinpath(output_directory, "soft_k_dependency_mis.pdf"), fig_mis_k)
    # 1C. Continuous Phase Space Slice at tau = 0.25, 0.65, 2.50
    println("  - Generating 2D continuous phase space slices...")
    for tau_sample in [0.25, 0.65, 2.50]
        fig_slice = plot_soft_phase_space_slice_2d(
            mis_raw,
            tau_sample;
            feature_indices = [2, 3],
            x_label = L"T\,[\mathrm{fm}^{-1}]",
            y_label = L"\mathcal{A}",
            color_limits = (1.0, 2.0),
            k = k_fixed,
            tol = tol_fixed,
            delta = delta_fixed
        )
        tau_tag = @sprintf("%.2f", tau_sample)
        save(joinpath(output_directory, "soft_phase_space_slice_mis_tau_$(tau_tag).pdf"), fig_slice)
    end

    # 1D. Continuous Phase Space Grid (2D)
    grid_taus = [0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.5, 2.5]
    fig_mis_grid = plot_soft_phase_space_grid_2d(
        mis_raw,
        grid_taus;
        feature_indices = [2, 3],
        x_label = L"T\,[\mathrm{fm}^{-1}]",
        y_label = L"\mathcal{A}",
        color_limits = (1.0, 2.0),
        k = k_fixed,
        tol = tol_fixed,
        delta = delta_fixed
    )
    save(joinpath(output_directory, "soft_phase_space_grid_mis.pdf"), fig_mis_grid)

    # =========================================================================
    # Part 2: HJSW Model Evaluation (Resolving 3 -> 2 -> 1 Dimensional Collapse)
    # =========================================================================
    println("\n[2/4] Running Soft-Weighted LPCA on HJSW Model (3D Space)...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)

    # 2A. Direct Comparison: Hard vs Soft-Weighted LPCA
    println("  - Computing Hard vs. Soft-Weighted comparison for HJSW...")
    hjsw_scan = scan_soft_weighted_dimension(
        hjsw_raw,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        k = k_fixed,
        tol = tol_fixed,
        delta = delta_fixed
    )

    fig_hjsw_compare = plot_soft_weighted_dimension(hjsw_scan; compare_hard = true)
    save(joinpath(output_directory, "soft_vs_hard_comparison_hjsw.pdf"), fig_hjsw_compare)

    # 2B. K-Dependency of Soft-Weighted LPCA for HJSW
    println("  - Computing Soft-Weighted K-dependency for HJSW...")
    fig_hjsw_k = plot_soft_k_dependency(
        hjsw_raw,
        k_test_values,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        tol = tol_fixed,
        delta = delta_fixed
    )
    save(joinpath(output_directory, "soft_k_dependency_hjsw.pdf"), fig_hjsw_k)

    # 2C. 3D Continuous Phase Space Slices
    println("  - Generating 3D continuous phase space slices for HJSW...")
    for tau_sample in [0.25, 0.65, 2.50]
        fig_slice_3d = plot_soft_phase_space_slice_3d(
            hjsw_raw,
            tau_sample;
            feature_indices = [2, 3, 4],
            color_limits = (1.0, 3.0),
            k = k_fixed,
            tol = tol_fixed,
            delta = delta_fixed
        )
        tau_tag = @sprintf("%.2f", tau_sample)
        save(joinpath(output_directory, "soft_phase_space_slice_hjsw_3d_tau_$(tau_tag).pdf"), fig_slice_3d)
    end

    # 2D. 2D Continuous Projection Grid on (A, B) plane
    # 2D. 3D Phase Space Grid
    println("  - Generating 3D phase space grid for HJSW...")
    fig_hjsw_grid_3d = plot_soft_phase_space_grid_3d(
        hjsw_raw,
        grid_taus;
        feature_indices = [2, 3, 4],
        x_label = L"T\,[\mathrm{MeV}]",
        y_label = L"\mathcal{A}",
        z_label = L"\mathcal{B}",
        color_limits = (1.0, 3.0),
        k = k_fixed,
        tol = tol_fixed,
        delta = delta_fixed
    )
    save(joinpath(output_directory, "soft_phase_space_grid_hjsw_3d.pdf"), fig_hjsw_grid_3d)

    # =========================================================================
    # Part 3: Soft Transition Width (Delta) & Cutoff (tol) Sensitivity
    # =========================================================================
    println("\n[3/4] Evaluating Sigmoidal Softness Parameter (Delta) Sensitivity...")
    deltas = [0.002, 0.005, 0.010]
    palette = [:dodgerblue, :forestgreen, :darkorange]

    fig_delta = Figure(size = (1100, 520))
    ax_mis_d = Axis(fig_delta[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\langle d \rangle_W", xautolimitmargin = (0.0, 0.04))
    ax_hjsw_d = Axis(fig_delta[1, 2], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\langle d \rangle_W", xautolimitmargin = (0.0, 0.04))

    for (d_idx, delta_val) in enumerate(deltas)
        c = palette[d_idx]
        res_m = scan_soft_weighted_dimension(mis_raw, tau_grid; feature_indices = [2, 3], normalize_method = :max, k = k_fixed, tol = tol_fixed, delta = delta_val)
        lines!(ax_mis_d, tau_grid, res_m.mean_dims; color = c, linewidth = 2.5, label = L"\Delta = %$(delta_val)")

        res_h = scan_soft_weighted_dimension(hjsw_raw, tau_grid; feature_indices = [2, 3, 4], normalize_method = :max, k = k_fixed, tol = tol_fixed, delta = delta_val)
        lines!(ax_hjsw_d, tau_grid, res_h.mean_dims; color = c, linewidth = 2.5, label = L"\Delta = %$(delta_val)")
    end
    axislegend(ax_mis_d, position = :rt)
    axislegend(ax_hjsw_d, position = :rt)
    save(joinpath(output_directory, "soft_delta_sensitivity_comparison.pdf"), fig_delta)

    # =========================================================================
    # Part 4: Numerical Results CSV Table Export
    # =========================================================================
    println("\n[4/4] Exporting numerical CSV summary table...")
    df_results = DataFrame(
        tau = tau_grid,
        mis_soft_mean = mis_scan.mean_dims,
        mis_soft_std = mis_scan.std_dims,
        mis_hard_mean = mis_scan.unweighted_hard_means,
        hjsw_soft_mean = hjsw_scan.mean_dims,
        hjsw_soft_std = hjsw_scan.std_dims,
        hjsw_hard_mean = hjsw_scan.unweighted_hard_means
    )
    results_csv = joinpath(results_directory, "soft_weighted_dimension_summary.csv")
    CSV.write(results_csv, df_results)
    println("  Summary table written to: $(results_csv)")

    println("\n=================================================================")
    println("  SOFT-WEIGHTED EXPERIMENTS COMPLETED SUCCESSFULLY!")
    println("  Figures saved to: $(output_directory)")
    println("=================================================================")
    return (mis_scan = mis_scan, hjsw_scan = hjsw_scan)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_soft_weighted_lpca_experiment()
end
