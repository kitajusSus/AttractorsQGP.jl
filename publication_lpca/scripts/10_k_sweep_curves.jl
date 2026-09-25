"""
    Script 10: Sweep of Local Dimension d as a function of K in [3, 50]
    Evaluates how the estimated local dimension changes as K varies from 3 to 50
    for Conformal MIS and HJSW models.

    Generates strictly separate PDF figures for:
    - Discrete Hard-Threshold Local Dimension: dims() -> d(K)
    - Continuous Participation Ratio Dimension: pr() -> d_PR(K)
    With distinct colored curves for each proper time tau slice and shaded ±1σ bands.
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

function run_k_sweep_curve_experiment(;
        mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
        hjsw_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
        output_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "k_sweep_curves"),
        results_directory::AbstractString = joinpath(@__DIR__, "..", "results")
    )
    mkpath(output_directory)
    mkpath(results_directory)
    CairoMakie.activate!()

    println("=== Starting 10_k_sweep_curves Experiment ===")

    # 1. K range from 3 to 50
    k_range = 3:1:50

    # 2. Key representative proper time slices
    tau_slices = [0.20, 0.25, 0.35, 0.50, 0.75, 1.00, 1.50, 2.50]

    # Elegant publication color palette for tau curves (cool -> warm)
    tau_palette = [
        :dodgerblue,   # tau = 0.20
        :darkcyan,     # tau = 0.25
        :forestgreen,  # tau = 0.35
        :goldenrod,    # tau = 0.50
        :darkorange,   # tau = 0.75
        :crimson,      # tau = 1.00
        :purple,       # tau = 1.50
        :black         # tau = 2.50
    ]

    # -------------------------------------------------------------
    # A. Conformal MIS Analysis (Initial 2D Phase Space)
    # -------------------------------------------------------------
    println("\n[1/2] Analyzing Conformal MIS model (K in 3..50)...")
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)

    # A1. Z-Score Normalization (Standard Best Practice)
    println("  - Computing K-sweep for MIS (Z-Score)...")
    mis_zscore_data = sweep_k_dimensions_over_tau(
        mis_raw,
        k_range,
        tau_slices;
        feature_indices = [2, 3],
        normalize_method = :zscore,
        tolerance = 0.01
    )

    fig_mis_zscore_dims = plot_k_sweep_curve(
        mis_zscore_data;
        method = :dims,
        palette = tau_palette,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Hard LPCA }\langle d \rangle(K)\text{ [Z-Score]}"
    )
    save(joinpath(output_directory, "k_sweep_mis_zscore_dims.pdf"), fig_mis_zscore_dims)

    fig_mis_zscore_pr = plot_k_sweep_curve(
        mis_zscore_data;
        method = :pr,
        palette = tau_palette,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(K)\text{ [Z-Score]}"
    )
    save(joinpath(output_directory, "k_sweep_mis_zscore_pr.pdf"), fig_mis_zscore_pr)

    # A2. Physical Coordinates (T, A)
    println("  - Computing K-sweep for MIS Physical (T, A)...")
    mis_phys_data = sweep_k_dimensions_over_tau(
        mis_variants.physical,
        k_range,
        tau_slices;
        feature_indices = [2, 3],
        normalize_method = :max,
        tolerance = 0.01
    )

    fig_mis_phys_dims = plot_k_sweep_curve(
        mis_phys_data;
        method = :dims,
        palette = tau_palette,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Hard LPCA }\langle d \rangle(K)\text{ [Physical } (T, \mathcal{A})\text{]}"
    )
    save(joinpath(output_directory, "k_sweep_mis_physical_dims.pdf"), fig_mis_phys_dims)

    fig_mis_phys_pr = plot_k_sweep_curve(
        mis_phys_data;
        method = :pr,
        palette = tau_palette,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(K)\text{ [Physical } (T, \mathcal{A})\text{]}"
    )
    save(joinpath(output_directory, "k_sweep_mis_physical_pr.pdf"), fig_mis_phys_pr)

    # A3. Dimensionless Scaling Coordinates (w, A)
    println("  - Computing K-sweep for MIS Dimensionless (w, A)...")
    mis_dimless_data = sweep_k_dimensions_over_tau(
        mis_variants.dimensionless,
        k_range,
        tau_slices;
        feature_indices = [2, 3],
        normalize_method = :max,
        tolerance = 0.01
    )

    fig_mis_dimless_dims = plot_k_sweep_curve(
        mis_dimless_data;
        method = :dims,
        palette = tau_palette,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Hard LPCA }\langle d \rangle(K)\text{ [Dimensionless } (w, \mathcal{A})\text{]}"
    )
    save(joinpath(output_directory, "k_sweep_mis_dimensionless_dims.pdf"), fig_mis_dimless_dims)

    fig_mis_dimless_pr = plot_k_sweep_curve(
        mis_dimless_data;
        method = :pr,
        palette = tau_palette,
        y_limits = (0.8, 2.3),
        title = L"\text{Conformal MIS: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(K)\text{ [Dimensionless } (w, \mathcal{A})\text{]}"
    )
    save(joinpath(output_directory, "k_sweep_mis_dimensionless_pr.pdf"), fig_mis_dimless_pr)

    # -------------------------------------------------------------
    # B. HJSW Analysis (Initial 3D Phase Space)
    # -------------------------------------------------------------
    println("\n[2/2] Analyzing HJSW model (K in 3..50)...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)

    # B1. Z-Score Normalization (Standard Best Practice)
    println("  - Computing K-sweep for HJSW (Z-Score)...")
    hjsw_zscore_data = sweep_k_dimensions_over_tau(
        hjsw_raw,
        k_range,
        tau_slices;
        feature_indices = [2, 3, 4],
        normalize_method = :zscore,
        tolerance = 0.01
    )

    fig_hjsw_zscore_dims = plot_k_sweep_curve(
        hjsw_zscore_data;
        method = :dims,
        palette = tau_palette,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Hard LPCA }\langle d \rangle(K)\text{ [Z-Score]}"
    )
    save(joinpath(output_directory, "k_sweep_hjsw_zscore_dims.pdf"), fig_hjsw_zscore_dims)

    fig_hjsw_zscore_pr = plot_k_sweep_curve(
        hjsw_zscore_data;
        method = :pr,
        palette = tau_palette,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(K)\text{ [Z-Score]}"
    )
    save(joinpath(output_directory, "k_sweep_hjsw_zscore_pr.pdf"), fig_hjsw_zscore_pr)

    # B2. Physical Coordinates (T, A, B)
    println("  - Computing K-sweep for HJSW Physical (T, A, B)...")
    hjsw_phys_data = sweep_k_dimensions_over_tau(
        hjsw_variants.physical,
        k_range,
        tau_slices;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        tolerance = 0.01
    )

    fig_hjsw_phys_dims = plot_k_sweep_curve(
        hjsw_phys_data;
        method = :dims,
        palette = tau_palette,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Hard LPCA }\langle d \rangle(K)\text{ [Physical } (T, \mathcal{A}, \mathcal{B})\text{]}"
    )
    save(joinpath(output_directory, "k_sweep_hjsw_physical_dims.pdf"), fig_hjsw_phys_dims)

    fig_hjsw_phys_pr = plot_k_sweep_curve(
        hjsw_phys_data;
        method = :pr,
        palette = tau_palette,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(K)\text{ [Physical } (T, \mathcal{A}, \mathcal{B})\text{]}"
    )
    save(joinpath(output_directory, "k_sweep_hjsw_physical_pr.pdf"), fig_hjsw_phys_pr)

    # B3. Dimensionless Scaling Coordinates (w, A, B)
    println("  - Computing K-sweep for HJSW Dimensionless (w, A, B)...")
    hjsw_dimless_data = sweep_k_dimensions_over_tau(
        hjsw_variants.dimensionless,
        k_range,
        tau_slices;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        tolerance = 0.01
    )

    fig_hjsw_dimless_dims = plot_k_sweep_curve(
        hjsw_dimless_data;
        method = :dims,
        palette = tau_palette,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Hard LPCA }\langle d \rangle(K)\text{ [Dimensionless } (w, \mathcal{A}, \mathcal{B})\text{]}"
    )
    save(joinpath(output_directory, "k_sweep_hjsw_dimensionless_dims.pdf"), fig_hjsw_dimless_dims)

    fig_hjsw_dimless_pr = plot_k_sweep_curve(
        hjsw_dimless_data;
        method = :pr,
        palette = tau_palette,
        y_limits = (0.8, 3.4),
        title = L"\text{HJSW Model: Participation Ratio }\langle d_{\mathrm{PR}} \rangle(K)\text{ [Dimensionless } (w, \mathcal{A}, \mathcal{B})\text{]}"
    )
    save(joinpath(output_directory, "k_sweep_hjsw_dimensionless_pr.pdf"), fig_hjsw_dimless_pr)

    # -------------------------------------------------------------
    # C. Save CSV Summary Table
    # -------------------------------------------------------------
    summary_csv_path = joinpath(results_directory, "k_sweep_3_to_50_summary.csv")
    println("\nSaving summary table to: ", summary_csv_path)
    open(summary_csv_path, "w") do io
        write(io, "model,representation,tau,k,mean_dims,std_dims,mean_pr,std_pr\n")
        # MIS Z-score
        for tau in tau_slices
            res = mis_zscore_data.results[Float64(tau)]
            for (idx, k) in enumerate(res.k_values)
                @printf(io, "MIS,zscore,%.2f,%d,%.4f,%.4f,%.4f,%.4f\n",
                    tau, k, res.mean_dims[idx], res.std_dims[idx], res.mean_pr[idx], res.std_pr[idx])
            end
        end
        # HJSW Z-score
        for tau in tau_slices
            res = hjsw_zscore_data.results[Float64(tau)]
            for (idx, k) in enumerate(res.k_values)
                @printf(io, "HJSW,zscore,%.2f,%d,%.4f,%.4f,%.4f,%.4f\n",
                    tau, k, res.mean_dims[idx], res.std_dims[idx], res.mean_pr[idx], res.std_pr[idx])
            end
        end
    end

    println("=== 10_k_sweep_curves Completed Successfully! ===")
    return (
        mis_zscore = mis_zscore_data,
        mis_physical = mis_phys_data,
        mis_dimensionless = mis_dimless_data,
        hjsw_zscore = hjsw_zscore_data,
        hjsw_physical = hjsw_phys_data,
        hjsw_dimensionless = hjsw_dimless_data,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_k_sweep_curve_experiment()
end
