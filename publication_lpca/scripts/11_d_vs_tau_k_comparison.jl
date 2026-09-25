"""
    Script 11: Dimension Trajectories d(tau) vs Neighborhood Size K in [3, 50]
    Generates two distinct types of figures across tau in [0.2, 6.0] fm/c:
    1. Multi-K curves where the color of each curve indicates the value of K in {3, 5, 8, 12, 16, 24, 35, 50},
       with shaded ±1σ ensemble dispersion bands and lines-only variants.
    2. K-averaged curves d(tau) where the trajectory is averaged over the full range K in [3, 50],
       with shaded ±1σ_K band and [min_K, max_K] envelope representing K-choice uncertainty,
       as well as lines-only variants.

    Evaluated systematically across ALL 5 data standardization / normalization methods:
    - Z-Score standardization (:zscore)
    - Min-Max scaling to [0, 1] (:minmax)
    - Abs-Max scaling to [-1, 1] (:max)
    - Physical coordinates in raw units (:none)
    - Dimensionless theoretical coordinates w = tau * T (:dimensionless)

    Generates strictly separate PDF figures for:
    - Discrete Hard-Threshold Local Dimension: dims() -> d(tau)
    - Continuous Participation Ratio Dimension: pr() -> d_PR(tau)
    For both Conformal MIS (2D) and HJSW (3D) models.
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
        hjsw_dataset_path::AbstractString = isfile(joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_10000_points.h5")) ?
            joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_10000_points.h5") :
            joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
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

    # 3. Dense 50-point tau grid covering [0.2, 5.0] fm/c
    tau_grid = Float64[round(t, digits = 2) for t in vcat(collect(0.20:0.05:1.50), collect(1.65:0.15:5.00))]
    tau_max = 5.0

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

    # All 5 normalization configurations to evaluate
    norm_configs = [
        (name = "zscore", method = :zscore, dataset_type = :physical, label = "Z-Score"),
        (name = "minmax", method = :minmax, dataset_type = :physical, label = "Min-Max"),
        (name = "max", method = :max, dataset_type = :physical, label = "Abs-Max"),
        (name = "physical", method = :none, dataset_type = :physical, label = "Physical (Raw Units)"),
        (name = "dimensionless", method = :max, dataset_type = :dimensionless, label = "Dimensionless (w, A)"),
    ]

    # Helper function to plot and save all 8 figure variants for a given sweep result
    function generate_k_plots(data, model_str, cfg_name, y_lims)
        # 1. Multi-K curves with ±1σ band
        fig_multi_dims = plot_d_vs_tau_multi_k(
            data;
            method = :dims,
            k_subset = k_subset,
            tau_max = tau_max,
            palette = k_palette,
            y_limits = y_lims
        )
        save(joinpath(output_directory, "d_vs_tau_$(model_str)_$(cfg_name)_multi_k_dims.pdf"), fig_multi_dims)

        fig_multi_pr = plot_d_vs_tau_multi_k(
            data;
            method = :pr,
            k_subset = k_subset,
            tau_max = tau_max,
            palette = k_palette,
            y_limits = y_lims
        )
        save(joinpath(output_directory, "d_vs_tau_$(model_str)_$(cfg_name)_multi_k_pr.pdf"), fig_multi_pr)

        # 2. Multi-K curves (LINES ONLY - no std bands)
        fig_lines_dims = plot_d_vs_tau_multi_k(
            data;
            method = :dims,
            k_subset = k_subset,
            tau_max = tau_max,
            palette = k_palette,
            show_std = false,
            linewidth = 2.8,
            y_limits = y_lims
        )
        save(joinpath(output_directory, "d_vs_tau_$(model_str)_$(cfg_name)_multi_k_lines_dims.pdf"), fig_lines_dims)

        fig_lines_pr = plot_d_vs_tau_multi_k(
            data;
            method = :pr,
            k_subset = k_subset,
            tau_max = tau_max,
            palette = k_palette,
            show_std = false,
            linewidth = 2.8,
            y_limits = y_lims
        )
        save(joinpath(output_directory, "d_vs_tau_$(model_str)_$(cfg_name)_multi_k_lines_pr.pdf"), fig_lines_pr)

        # 3. K-Averaged over all K in [3, 50] with envelope and ±1σ_K band
        fig_avg_dims = plot_d_vs_tau_k_averaged(
            data;
            method = :dims,
            tau_max = tau_max,
            line_color = :dodgerblue,
            y_limits = y_lims
        )
        save(joinpath(output_directory, "d_vs_tau_$(model_str)_$(cfg_name)_k_averaged_dims.pdf"), fig_avg_dims)

        fig_avg_pr = plot_d_vs_tau_k_averaged(
            data;
            method = :pr,
            tau_max = tau_max,
            line_color = :crimson,
            y_limits = y_lims
        )
        save(joinpath(output_directory, "d_vs_tau_$(model_str)_$(cfg_name)_k_averaged_pr.pdf"), fig_avg_pr)

        # 4. K-Averaged (LINE ONLY - no std band, no envelope)
        fig_avg_line_dims = plot_d_vs_tau_k_averaged(
            data;
            method = :dims,
            tau_max = tau_max,
            line_color = :dodgerblue,
            show_std = false,
            show_envelope = false,
            linewidth = 3.0,
            y_limits = y_lims
        )
        save(joinpath(output_directory, "d_vs_tau_$(model_str)_$(cfg_name)_k_averaged_line_only_dims.pdf"), fig_avg_line_dims)

        fig_avg_line_pr = plot_d_vs_tau_k_averaged(
            data;
            method = :pr,
            tau_max = tau_max,
            line_color = :crimson,
            show_std = false,
            show_envelope = false,
            linewidth = 3.0,
            y_limits = y_lims
        )
        save(joinpath(output_directory, "d_vs_tau_$(model_str)_$(cfg_name)_k_averaged_line_only_pr.pdf"), fig_avg_line_pr)

        # Backward compatibility aliases for zscore (so previous references without _zscore_ still work)
        if cfg_name == "zscore"
            save(joinpath(output_directory, "d_vs_tau_$(model_str)_multi_k_dims.pdf"), fig_multi_dims)
            save(joinpath(output_directory, "d_vs_tau_$(model_str)_multi_k_pr.pdf"), fig_multi_pr)
            save(joinpath(output_directory, "d_vs_tau_$(model_str)_multi_k_lines_dims.pdf"), fig_lines_dims)
            save(joinpath(output_directory, "d_vs_tau_$(model_str)_multi_k_lines_pr.pdf"), fig_lines_pr)
            save(joinpath(output_directory, "d_vs_tau_$(model_str)_k_averaged_dims.pdf"), fig_avg_dims)
            save(joinpath(output_directory, "d_vs_tau_$(model_str)_k_averaged_pr.pdf"), fig_avg_pr)
            save(joinpath(output_directory, "d_vs_tau_$(model_str)_k_averaged_line_only_dims.pdf"), fig_avg_line_dims)
            save(joinpath(output_directory, "d_vs_tau_$(model_str)_k_averaged_line_only_pr.pdf"), fig_avg_line_pr)
        end
    end

    # -------------------------------------------------------------
    # A. Conformal MIS Model (Initial 2D Phase Space)
    # -------------------------------------------------------------
    println("\n[1/2] Analyzing Conformal MIS model across all 5 standardizations...")
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)
    mis_data_all = Dict{String, Any}()

    for cfg in norm_configs
        println("  - Computing MIS [$(cfg.name)]...")
        ds = cfg.dataset_type == :dimensionless ? mis_variants.dimensionless : mis_raw
        data = sweep_k_dimensions_over_tau(
            ds,
            k_range,
            tau_grid;
            feature_indices = [2, 3],
            normalize_method = cfg.method,
            tolerance = 0.01,
            max_points = 10_000
        )
        mis_data_all[cfg.name] = data
        generate_k_plots(data, "mis", cfg.name, (0.8, 2.3))
    end

    # -------------------------------------------------------------
    # B. HJSW Model (Initial 3D Phase Space)
    # -------------------------------------------------------------
    println("\n[2/2] Analyzing HJSW model across all 5 standardizations...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)
    hjsw_data_all = Dict{String, Any}()

    for cfg in norm_configs
        println("  - Computing HJSW [$(cfg.name)]...")
        ds = cfg.dataset_type == :dimensionless ? hjsw_variants.dimensionless : hjsw_raw
        data = sweep_k_dimensions_over_tau(
            ds,
            k_range,
            tau_grid;
            feature_indices = [2, 3, 4],
            normalize_method = cfg.method,
            tolerance = 0.01,
            max_points = 10_000
        )
        hjsw_data_all[cfg.name] = data
        generate_k_plots(data, "hjsw", cfg.name, (0.8, 3.4))
    end

    # -------------------------------------------------------------
    # C. Save CSV Summary Table
    # -------------------------------------------------------------
    summary_csv_path = joinpath(results_directory, "d_vs_tau_k_averaged_summary.csv")
    println("\nSaving summary table to: ", summary_csv_path)
    open(summary_csv_path, "w") do io
        write(io, "model,representation,tau,mean_k_dims,std_k_dims,min_k_dims,max_k_dims,mean_k_pr,std_k_pr,min_k_pr,max_k_pr\n")
        for (m_str, all_data) in [("MIS", mis_data_all), ("HJSW", hjsw_data_all)]
            for cfg in norm_configs
                sweep_data = all_data[cfg.name]
                for tau in tau_grid
                    r = sweep_data.results[Float64(tau)]
                    @printf(io, "%s,%s,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                        m_str, cfg.name, tau,
                        mean(r.mean_dims), std(r.mean_dims), minimum(r.mean_dims), maximum(r.mean_dims),
                        mean(r.mean_pr), std(r.mean_pr), minimum(r.mean_pr), maximum(r.mean_pr)
                    )
                end
            end
        end
    end

    println("=== 11_d_vs_tau_k_comparison Completed Successfully! ===")
    return (
        mis = mis_data_all,
        hjsw = hjsw_data_all
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_d_vs_tau_k_experiment()
end
