"""
    Script 10: Sweep of Local Dimension d as a function of K in [3, 50]
    Evaluates how the estimated local dimension changes as K varies from 3 to 50
    for Conformal MIS and HJSW models across ALL 5 standardization methods:
    - Z-Score (:zscore)
    - Min-Max (:minmax)
    - Abs-Max (:max)
    - Physical (:none)
    - Dimensionless (:dimensionless)

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
        hjsw_dataset_path::AbstractString = isfile(joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_10000_points.h5")) ?
            joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_10000_points.h5") :
            joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
        output_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "k_sweep_curves"),
        results_directory::AbstractString = joinpath(@__DIR__, "..", "results"),
        map_k_values::AbstractVector{<:Integer} = [10, 20, 40, 80],
        n_map_slices::Int = 15
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

    # All 5 normalization configurations to evaluate
    norm_configs = [
        (name = "zscore", method = :zscore, dataset_type = :physical, label = "Z-Score"),
        (name = "minmax", method = :minmax, dataset_type = :physical, label = "Min-Max"),
        (name = "max", method = :max, dataset_type = :physical, label = "Abs-Max"),
        (name = "physical", method = :none, dataset_type = :physical, label = "Physical (Raw Units)"),
        (name = "dimensionless", method = :max, dataset_type = :dimensionless, label = "Dimensionless (w, A)"),
    ]

    # -------------------------------------------------------------
    # A. Conformal MIS Analysis (Initial 2D Phase Space)
    # -------------------------------------------------------------
    println("\n[1/2] Analyzing Conformal MIS model (K in 3..50)...")
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)
    mis_results_all = Dict{String, Any}()

    for cfg in norm_configs
        println("  - Computing K-sweep for MIS [$(cfg.name)]...")
        ds = cfg.dataset_type == :dimensionless ? mis_variants.dimensionless : mis_raw
        data = sweep_k_dimensions_over_tau(
            ds,
            k_range,
            tau_slices;
            feature_indices = [2, 3],
            normalize_method = cfg.method,
            tolerance = 0.01,
            max_points = 10_000
        )
        mis_results_all[cfg.name] = data

        fig_dims = plot_k_sweep_curve(data; method = :dims, palette = tau_palette, y_limits = (0.8, 2.3))
        save(joinpath(output_directory, "k_sweep_mis_$(cfg.name)_dims.pdf"), fig_dims)

        fig_pr = plot_k_sweep_curve(data; method = :pr, palette = tau_palette, y_limits = (0.8, 2.3))
        save(joinpath(output_directory, "k_sweep_mis_$(cfg.name)_pr.pdf"), fig_pr)

        # Generate 2D Heatmap / Map of Local Dimension d(tau, K) using plot_map_lpca
        println("  - Generating LPCA map for MIS [$(cfg.name)]...")
        fig_map = plot_map_lpca(
            ds;
            zakres_K = map_k_values,
            n_slices = n_map_slices,
            feature_cols = [2, 3],
            normalize = cfg.method,
            title = L"\text{Conformal MIS: Map } d(\tau, K)\text{ [%$(cfg.label)]}",
            colorrange = (1, 2),
            ticks = [1, 2]
        )
        save(joinpath(output_directory, "map_lpca_mis_$(cfg.name).pdf"), fig_map)
        save(joinpath(output_directory, "map_lpca_mis_$(cfg.name).png"), fig_map)

        if cfg.name == "zscore"
            save(joinpath(output_directory, "k_sweep_mis_dims.pdf"), fig_dims)
            save(joinpath(output_directory, "k_sweep_mis_pr.pdf"), fig_pr)
            save(joinpath(output_directory, "map_lpca_mis.pdf"), fig_map)
            save(joinpath(output_directory, "map_lpca_mis.png"), fig_map)
        end
    end

    # -------------------------------------------------------------
    # B. HJSW Analysis (Initial 3D Phase Space)
    # -------------------------------------------------------------
    println("\n[2/2] Analyzing HJSW model (K in 3..50)...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)
    hjsw_results_all = Dict{String, Any}()

    for cfg in norm_configs
        println("  - Computing K-sweep for HJSW [$(cfg.name)]...")
        ds = cfg.dataset_type == :dimensionless ? hjsw_variants.dimensionless : hjsw_raw
        data = sweep_k_dimensions_over_tau(
            ds,
            k_range,
            tau_slices;
            feature_indices = [2, 3, 4],
            normalize_method = cfg.method,
            tolerance = 0.01,
            max_points = 10_000
        )
        hjsw_results_all[cfg.name] = data

        fig_dims = plot_k_sweep_curve(data; method = :dims, palette = tau_palette, y_limits = (0.8, 3.4))
        save(joinpath(output_directory, "k_sweep_hjsw_$(cfg.name)_dims.pdf"), fig_dims)

        fig_pr = plot_k_sweep_curve(data; method = :pr, palette = tau_palette, y_limits = (0.8, 3.4))
        save(joinpath(output_directory, "k_sweep_hjsw_$(cfg.name)_pr.pdf"), fig_pr)

        # Generate 2D Heatmap / Map of Local Dimension d(tau, K) using plot_map_lpca
        println("  - Generating LPCA map for HJSW [$(cfg.name)]...")
        fig_map = plot_map_lpca(
            ds;
            zakres_K = map_k_values,
            n_slices = n_map_slices,
            feature_cols = [2, 3, 4],
            normalize = cfg.method,
            title = L"\text{HJSW Model: Map } d(\tau, K)\text{ [%$(cfg.label)]}",
            colorrange = (1, 3),
            ticks = [1, 2, 3]
        )
        save(joinpath(output_directory, "map_lpca_hjsw_$(cfg.name).pdf"), fig_map)
        save(joinpath(output_directory, "map_lpca_hjsw_$(cfg.name).png"), fig_map)

        if cfg.name == "zscore"
            save(joinpath(output_directory, "k_sweep_hjsw_dims.pdf"), fig_dims)
            save(joinpath(output_directory, "k_sweep_hjsw_pr.pdf"), fig_pr)
            save(joinpath(output_directory, "map_lpca_hjsw.pdf"), fig_map)
            save(joinpath(output_directory, "map_lpca_hjsw.png"), fig_map)
        end
    end

    # -------------------------------------------------------------
    # C. Save CSV Summary Table
    # -------------------------------------------------------------
    summary_csv_path = joinpath(results_directory, "k_sweep_3_to_50_summary.csv")
    println("\nSaving summary table to: ", summary_csv_path)
    open(summary_csv_path, "w") do io
        write(io, "model,representation,tau,k,mean_dims,std_dims,mean_pr,std_pr\n")
        for (m_str, all_data) in [("MIS", mis_results_all), ("HJSW", hjsw_results_all)]
            for cfg in norm_configs
                sweep_data = all_data[cfg.name]
                for tau in tau_slices
                    res = sweep_data.results[Float64(tau)]
                    for (idx, k) in enumerate(res.k_values)
                        @printf(io, "%s,%s,%.2f,%d,%.4f,%.4f,%.4f,%.4f\n",
                            m_str, cfg.name, tau, k, res.mean_dims[idx], res.std_dims[idx], res.mean_pr[idx], res.std_pr[idx])
                    end
                end
            end
        end
    end

    println("=== 10_k_sweep_curves Completed Successfully! ===")
    return (
        mis = mis_results_all,
        hjsw = hjsw_results_all
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_k_sweep_curve_experiment()
end
