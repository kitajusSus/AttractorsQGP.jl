"""
    Script 11: Dimension Trajectories d(tau) vs Neighborhood Size K in [3, 50]
    Generates two distinct types of figures across tau in [0.2, 6.0] fm/c:
    1. Multi-K curves where the color of each curve indicates the value of K in {3, 5, 8, 12, 16, 24, 35, 50},
       with shaded ±1σ ensemble dispersion bands.
    2. K-averaged curves d(tau) where the trajectory is averaged over the full range K in [3, 50],
       with shaded ±1σ_K band and [min_K, max_K] envelope representing K-choice uncertainty.

    Generates strictly separate PDF figures for:
    - Discrete Hard-Threshold Local Dimension: dims() -> d(tau)
    - Continuous Participation Ratio Dimension: pr() -> d_PR(tau)
    For both Conformal MIS and HJSW models across Z-Score and Physical representations.
"""

using AttractorsQGP
using CairoMakie
using LaTeXStrings
using Printf
using Statistics

if !isdefined(Main, :PublicationLPCA)
    include(joinpath(@__DIR__, "..", "src", "PublicationLPCA.jl"))
end
using .PublicationLPCA

function run_d_vs_tau_k_experiment(;
        mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
        hjsw_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
        output_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "d_vs_tau_k_dependency"),
        results_directory::AbstractString = joinpath(@__DIR__, "..", "results")
    )
    mkpath(output_directory)
    mkpath(results_directory)
    CairoMakie.activate!()

    println("=== Starting 11_d_vs_tau_k_comparison Experiment ===")

    # 1. Full K range
    k_range = 3:1:50

    # 2. Subset of K for multi-curve plots
    k_subset = [3, 5, 8, 12, 16, 24, 35, 50]

    # 3. Dense tau grid covering [0.2, 6.0] fm/c
    tau_grid = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    tau_max = 6.0

    # High-contrast color palette for K curves
    k_palette = [
        :navy,         # K = 3
        :dodgerblue,   # K = 5
        :darkcyan,     # K = 8
        :forestgreen,  # K = 12
        :goldenrod,    # K = 16
        :darkorange,   # K = 24
        :crimson,      # K = 35
        :purple        # K = 50
    ]

    # -------------------------------------------------------------
    # A. Conformal MIS Model (Initial 2D Phase Space)
    # -------------------------------------------------------------
    println("\n[1/2] Analyzing Conformal MIS model...")
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)

    # A1. MIS Z-Score Normalization (Standard Best Practice)
    println("  - Computing Z-Score sweep for MIS...")
    mis_zscore_data = sweep_k_dimensions_over_tau(
        mis_raw,
        k_range,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :zscore,
        tolerance = 0.01
    )

    # A1.1 Multi-K curves (color indicates K)
    fig_mis_zscore_multi_dims = plot_d_vs_tau_multi_k(
        mis_zscore_data;
        method = :dims,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Hard LPCA }\langle d \rangle(\tau)\text{ for varied } K\text{ [Z-Score]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_multi_k_dims.pdf"), fig_mis_zscore_multi_dims)

    fig_mis_zscore_multi_pr = plot_d_vs_tau_multi_k(
        mis_zscore_data;
        method = :pr,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ for varied } K\text{ [Z-Score]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_multi_k_pr.pdf"), fig_mis_zscore_multi_pr)

    # A1.1b Multi-K curves (LINES ONLY - no std bands)
    fig_mis_zscore_lines_dims = plot_d_vs_tau_multi_k(
        mis_zscore_data;
        method = :dims,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        show_std = false,
        linewidth = 2.8,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Hard LPCA }\langle d \rangle(\tau)\text{ for varied } K\text{ [Z-Score, Lines Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_multi_k_lines_dims.pdf"), fig_mis_zscore_lines_dims)

    fig_mis_zscore_lines_pr = plot_d_vs_tau_multi_k(
        mis_zscore_data;
        method = :pr,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        show_std = false,
        linewidth = 2.8,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ for varied } K\text{ [Z-Score, Lines Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_multi_k_lines_pr.pdf"), fig_mis_zscore_lines_pr)

    # A1.2 K-Averaged over all K in [3, 50]
    fig_mis_zscore_avg_dims = plot_d_vs_tau_k_averaged(
        mis_zscore_data;
        method = :dims,
        tau_max = tau_max,
        line_color = :dodgerblue,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: }K\text{-Averaged Hard LPCA }\langle d \rangle(\tau)\text{ [Z-Score]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_k_averaged_dims.pdf"), fig_mis_zscore_avg_dims)

    fig_mis_zscore_avg_pr = plot_d_vs_tau_k_averaged(
        mis_zscore_data;
        method = :pr,
        tau_max = tau_max,
        line_color = :crimson,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: }K\text{-Averaged Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ [Z-Score]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_k_averaged_pr.pdf"), fig_mis_zscore_avg_pr)

    # A1.2b K-Averaged (LINE ONLY - no std band, no envelope)
    fig_mis_zscore_avg_line_dims = plot_d_vs_tau_k_averaged(
        mis_zscore_data;
        method = :dims,
        tau_max = tau_max,
        line_color = :dodgerblue,
        show_std = false,
        show_envelope = false,
        linewidth = 3.0,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: }K\text{-Averaged Hard LPCA }\langle d \rangle(\tau)\text{ [Z-Score, Line Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_k_averaged_line_only_dims.pdf"), fig_mis_zscore_avg_line_dims)

    fig_mis_zscore_avg_line_pr = plot_d_vs_tau_k_averaged(
        mis_zscore_data;
        method = :pr,
        tau_max = tau_max,
        line_color = :crimson,
        show_std = false,
        show_envelope = false,
        linewidth = 3.0,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: }K\text{-Averaged Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ [Z-Score, Line Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_k_averaged_line_only_pr.pdf"), fig_mis_zscore_avg_line_pr)

    # A2. MIS Physical Coordinates (T, A)
    println("  - Computing Physical coordinates sweep for MIS...")
    mis_phys_data = sweep_k_dimensions_over_tau(
        mis_variants.physical,
        k_range,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        tolerance = 0.01
    )

    fig_mis_phys_multi_dims = plot_d_vs_tau_multi_k(
        mis_phys_data;
        method = :dims,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Hard LPCA }\langle d \rangle(\tau)\text{ for varied } K\text{ [Physical } (T, \mathcal{A})\text{]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_physical_multi_k_dims.pdf"), fig_mis_phys_multi_dims)

    fig_mis_phys_multi_pr = plot_d_vs_tau_multi_k(
        mis_phys_data;
        method = :pr,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ for varied } K\text{ [Physical } (T, \mathcal{A})\text{]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_physical_multi_k_pr.pdf"), fig_mis_phys_multi_pr)

    # A2.1b Lines Only
    fig_mis_phys_lines_dims = plot_d_vs_tau_multi_k(
        mis_phys_data;
        method = :dims,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        show_std = false,
        linewidth = 2.8,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Hard LPCA }\langle d \rangle(\tau)\text{ for varied } K\text{ [Physical, Lines Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_physical_multi_k_lines_dims.pdf"), fig_mis_phys_lines_dims)

    fig_mis_phys_lines_pr = plot_d_vs_tau_multi_k(
        mis_phys_data;
        method = :pr,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        show_std = false,
        linewidth = 2.8,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ for varied } K\text{ [Physical, Lines Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_physical_multi_k_lines_pr.pdf"), fig_mis_phys_lines_pr)

    fig_mis_phys_avg_dims = plot_d_vs_tau_k_averaged(
        mis_phys_data;
        method = :dims,
        tau_max = tau_max,
        line_color = :dodgerblue,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: }K\text{-Averaged Hard LPCA }\langle d \rangle(\tau)\text{ [Physical } (T, \mathcal{A})\text{]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_physical_k_averaged_dims.pdf"), fig_mis_phys_avg_dims)

    fig_mis_phys_avg_pr = plot_d_vs_tau_k_averaged(
        mis_phys_data;
        method = :pr,
        tau_max = tau_max,
        line_color = :crimson,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: }K\text{-Averaged Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ [Physical } (T, \mathcal{A})\text{]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_physical_k_averaged_pr.pdf"), fig_mis_phys_avg_pr)

    # A2.2b Line Only
    fig_mis_phys_avg_line_dims = plot_d_vs_tau_k_averaged(
        mis_phys_data;
        method = :dims,
        tau_max = tau_max,
        line_color = :dodgerblue,
        show_std = false,
        show_envelope = false,
        linewidth = 3.0,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: }K\text{-Averaged Hard LPCA }\langle d \rangle(\tau)\text{ [Physical, Line Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_physical_k_averaged_line_only_dims.pdf"), fig_mis_phys_avg_line_dims)

    fig_mis_phys_avg_line_pr = plot_d_vs_tau_k_averaged(
        mis_phys_data;
        method = :pr,
        tau_max = tau_max,
        line_color = :crimson,
        show_std = false,
        show_envelope = false,
        linewidth = 3.0,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: }K\text{-Averaged Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ [Physical, Line Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_mis_physical_k_averaged_line_only_pr.pdf"), fig_mis_phys_avg_line_pr)

    # -------------------------------------------------------------
    # B. HJSW Model (Initial 3D Phase Space)
    # -------------------------------------------------------------
    println("\n[2/2] Analyzing HJSW model...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)

    # B1. HJSW Z-Score Normalization (Standard Best Practice)
    println("  - Computing Z-Score sweep for HJSW...")
    hjsw_zscore_data = sweep_k_dimensions_over_tau(
        hjsw_raw,
        k_range,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :zscore,
        tolerance = 0.01
    )

    # B1.1 Multi-K curves (color indicates K)
    fig_hjsw_zscore_multi_dims = plot_d_vs_tau_multi_k(
        hjsw_zscore_data;
        method = :dims,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Hard LPCA }\langle d \rangle(\tau)\text{ for varied } K\text{ [Z-Score]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_multi_k_dims.pdf"), fig_hjsw_zscore_multi_dims)

    fig_hjsw_zscore_multi_pr = plot_d_vs_tau_multi_k(
        hjsw_zscore_data;
        method = :pr,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ for varied } K\text{ [Z-Score]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_multi_k_pr.pdf"), fig_hjsw_zscore_multi_pr)

    # B1.1b Multi-K curves (LINES ONLY - no std bands)
    fig_hjsw_zscore_lines_dims = plot_d_vs_tau_multi_k(
        hjsw_zscore_data;
        method = :dims,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        show_std = false,
        linewidth = 2.8,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Hard LPCA }\langle d \rangle(\tau)\text{ for varied } K\text{ [Z-Score, Lines Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_multi_k_lines_dims.pdf"), fig_hjsw_zscore_lines_dims)

    fig_hjsw_zscore_lines_pr = plot_d_vs_tau_multi_k(
        hjsw_zscore_data;
        method = :pr,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        show_std = false,
        linewidth = 2.8,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ for varied } K\text{ [Z-Score, Lines Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_multi_k_lines_pr.pdf"), fig_hjsw_zscore_lines_pr)

    # B1.2 K-Averaged over all K in [3, 50]
    fig_hjsw_zscore_avg_dims = plot_d_vs_tau_k_averaged(
        hjsw_zscore_data;
        method = :dims,
        tau_max = tau_max,
        line_color = :dodgerblue,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: }K\text{-Averaged Hard LPCA }\langle d \rangle(\tau)\text{ [Z-Score]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_k_averaged_dims.pdf"), fig_hjsw_zscore_avg_dims)

    fig_hjsw_zscore_avg_pr = plot_d_vs_tau_k_averaged(
        hjsw_zscore_data;
        method = :pr,
        tau_max = tau_max,
        line_color = :crimson,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: }K\text{-Averaged Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ [Z-Score]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_k_averaged_pr.pdf"), fig_hjsw_zscore_avg_pr)

    # B1.2b K-Averaged (LINE ONLY - no std band, no envelope)
    fig_hjsw_zscore_avg_line_dims = plot_d_vs_tau_k_averaged(
        hjsw_zscore_data;
        method = :dims,
        tau_max = tau_max,
        line_color = :dodgerblue,
        show_std = false,
        show_envelope = false,
        linewidth = 3.0,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: }K\text{-Averaged Hard LPCA }\langle d \rangle(\tau)\text{ [Z-Score, Line Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_k_averaged_line_only_dims.pdf"), fig_hjsw_zscore_avg_line_dims)

    fig_hjsw_zscore_avg_line_pr = plot_d_vs_tau_k_averaged(
        hjsw_zscore_data;
        method = :pr,
        tau_max = tau_max,
        line_color = :crimson,
        show_std = false,
        show_envelope = false,
        linewidth = 3.0,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: }K\text{-Averaged Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ [Z-Score, Line Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_k_averaged_line_only_pr.pdf"), fig_hjsw_zscore_avg_line_pr)

    # B2. HJSW Physical Coordinates (T, A, B)
    println("  - Computing Physical coordinates sweep for HJSW...")
    hjsw_phys_data = sweep_k_dimensions_over_tau(
        hjsw_variants.physical,
        k_range,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        tolerance = 0.01
    )

    fig_hjsw_phys_multi_dims = plot_d_vs_tau_multi_k(
        hjsw_phys_data;
        method = :dims,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Hard LPCA }\langle d \rangle(\tau)\text{ for varied } K\text{ [Physical } (T, \mathcal{A}, \mathcal{B})\text{]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_physical_multi_k_dims.pdf"), fig_hjsw_phys_multi_dims)

    fig_hjsw_phys_multi_pr = plot_d_vs_tau_multi_k(
        hjsw_phys_data;
        method = :pr,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ for varied } K\text{ [Physical } (T, \mathcal{A}, \mathcal{B})\text{]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_physical_multi_k_pr.pdf"), fig_hjsw_phys_multi_pr)

    # B2.1b Lines Only
    fig_hjsw_phys_lines_dims = plot_d_vs_tau_multi_k(
        hjsw_phys_data;
        method = :dims,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        show_std = false,
        linewidth = 2.8,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Hard LPCA }\langle d \rangle(\tau)\text{ for varied } K\text{ [Physical, Lines Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_physical_multi_k_lines_dims.pdf"), fig_hjsw_phys_lines_dims)

    fig_hjsw_phys_lines_pr = plot_d_vs_tau_multi_k(
        hjsw_phys_data;
        method = :pr,
        k_subset = k_subset,
        tau_max = tau_max,
        palette = k_palette,
        show_std = false,
        linewidth = 2.8,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ for varied } K\text{ [Physical, Lines Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_physical_multi_k_lines_pr.pdf"), fig_hjsw_phys_lines_pr)

    fig_hjsw_phys_avg_dims = plot_d_vs_tau_k_averaged(
        hjsw_phys_data;
        method = :dims,
        tau_max = tau_max,
        line_color = :dodgerblue,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: }K\text{-Averaged Hard LPCA }\langle d \rangle(\tau)\text{ [Physical } (T, \mathcal{A}, \mathcal{B})\text{]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_physical_k_averaged_dims.pdf"), fig_hjsw_phys_avg_dims)

    fig_hjsw_phys_avg_pr = plot_d_vs_tau_k_averaged(
        hjsw_phys_data;
        method = :pr,
        tau_max = tau_max,
        line_color = :crimson,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: }K\text{-Averaged Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ [Physical } (T, \mathcal{A}, \mathcal{B})\text{]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_physical_k_averaged_pr.pdf"), fig_hjsw_phys_avg_pr)

    # B2.2b Line Only
    fig_hjsw_phys_avg_line_dims = plot_d_vs_tau_k_averaged(
        hjsw_phys_data;
        method = :dims,
        tau_max = tau_max,
        line_color = :dodgerblue,
        show_std = false,
        show_envelope = false,
        linewidth = 3.0,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: }K\text{-Averaged Hard LPCA }\langle d \rangle(\tau)\text{ [Physical, Line Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_physical_k_averaged_line_only_dims.pdf"), fig_hjsw_phys_avg_line_dims)

    fig_hjsw_phys_avg_line_pr = plot_d_vs_tau_k_averaged(
        hjsw_phys_data;
        method = :pr,
        tau_max = tau_max,
        line_color = :crimson,
        show_std = false,
        show_envelope = false,
        linewidth = 3.0,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: }K\text{-Averaged Participation Ratio }\langle d_{\mathrm{PR}} \rangle(\tau)\text{ [Physical, Line Only]}"
    )
    save(joinpath(output_directory, "d_vs_tau_hjsw_physical_k_averaged_line_only_pr.pdf"), fig_hjsw_phys_avg_line_pr)

    # -------------------------------------------------------------
    # C. Save CSV Summary Table
    # -------------------------------------------------------------
    summary_csv_path = joinpath(results_directory, "d_vs_tau_k_averaged_summary.csv")
    println("\nSaving summary table to: ", summary_csv_path)
    open(summary_csv_path, "w") do io
        write(io, "model,representation,tau,mean_k_dims,std_k_dims,min_k_dims,max_k_dims,mean_k_pr,std_k_pr,min_k_pr,max_k_pr\n")
        for tau in tau_grid
            r_mis = mis_zscore_data.results[Float64(tau)]
            @printf(io, "MIS,zscore,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                tau,
                mean(r_mis.mean_dims), std(r_mis.mean_dims), minimum(r_mis.mean_dims), maximum(r_mis.mean_dims),
                mean(r_mis.mean_pr), std(r_mis.mean_pr), minimum(r_mis.mean_pr), maximum(r_mis.mean_pr)
            )
        end
        for tau in tau_grid
            r_hjsw = hjsw_zscore_data.results[Float64(tau)]
            @printf(io, "HJSW,zscore,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                tau,
                mean(r_hjsw.mean_dims), std(r_hjsw.mean_dims), minimum(r_hjsw.mean_dims), maximum(r_hjsw.mean_dims),
                mean(r_hjsw.mean_pr), std(r_hjsw.mean_pr), minimum(r_hjsw.mean_pr), maximum(r_hjsw.mean_pr)
            )
        end
    end

    println("=== 11_d_vs_tau_k_comparison Completed Successfully! ===")
    return (
        mis_zscore = mis_zscore_data,
        mis_physical = mis_phys_data,
        hjsw_zscore = hjsw_zscore_data,
        hjsw_physical = hjsw_phys_data,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_d_vs_tau_k_experiment()
end
