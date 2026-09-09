"""
    Script 02: Normalization and Scaling Method Comparison
    Evaluates how four normalization strategies (:none, :max, :minmax, :zscore)
    affect local PCA dimension estimation for MIS and HJSW models.
"""

using AttractorsQGP
using CairoMakie
using Printf
using Statistics

if !isdefined(Main, :PublicationLPCA)
    include(joinpath(@__DIR__, "..", "src", "PublicationLPCA.jl"))
end
using .PublicationLPCA

function run_normalization_experiment(;
    mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
    hjsw_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
    output_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "normalization"),
    results_directory::AbstractString = joinpath(@__DIR__, "..", "results")
)
    mkpath(output_directory)
    mkpath(results_directory)
    CairoMakie.activate!()

    println("=== Starting 02_normalization_tests Experiment ===")

    normalization_methods = [:none, :max, :minmax, :zscore]
    k_pairs = [(6, 12), (12, 24), (24, 48), (100, 200)]
    tau_grid = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.5, 10.0]

    # -------------------------------------------------------------
    # A. Conformal MIS Normalization Tests
    # -------------------------------------------------------------
    println("\n[1/2] Evaluating normalization methods on MIS model...")
    mis_raw = load_hydro_dataset(mis_dataset_path)

    println("  - Computing multi-panel normalization comparisons for MIS...")
    mis_norm_results = compare_normalization_methods(
        mis_raw,
        normalization_methods,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3],
        tolerance = 0.01
    )
    fig_mis_multipanel = plot_normalization_multipanel(
        mis_norm_results;
        methods_to_plot = normalization_methods
    )
    save(joinpath(output_directory, "normalization_methods_multipanel_mis.pdf"), fig_mis_multipanel)

    println("  - Computing direct overlay comparison at K=24 for MIS...")
    fig_mis_overlay = plot_normalization_direct_overlay(
        mis_raw,
        normalization_methods,
        24,
        tau_grid;
        feature_indices = [2, 3],
        tolerance = 0.01
    )
    save(joinpath(output_directory, "normalization_methods_direct_overlay_fixed_k24_mis.pdf"), fig_mis_overlay)

    # -------------------------------------------------------------
    # B. HJSW Normalization Tests
    # -------------------------------------------------------------
    println("\n[2/2] Evaluating normalization methods on HJSW model...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)

    println("  - Computing multi-panel normalization comparisons for HJSW...")
    hjsw_norm_results = compare_normalization_methods(
        hjsw_raw,
        normalization_methods,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3, 4],
        tolerance = 0.01
    )
    fig_hjsw_multipanel = plot_normalization_multipanel(
        hjsw_norm_results;
        methods_to_plot = normalization_methods
    )
    save(joinpath(output_directory, "normalization_methods_multipanel_hjsw.pdf"), fig_hjsw_multipanel)

    println("  - Computing direct overlay comparison at K=24 for HJSW...")
    fig_hjsw_overlay = plot_normalization_direct_overlay(
        hjsw_raw,
        normalization_methods,
        24,
        tau_grid;
        feature_indices = [2, 3, 4],
        tolerance = 0.01
    )
    save(joinpath(output_directory, "normalization_methods_direct_overlay_fixed_k24_hjsw.pdf"), fig_hjsw_overlay)

    # -------------------------------------------------------------
    # C. Save CSV Summary Table
    # -------------------------------------------------------------
    csv_path = joinpath(results_directory, "normalization_comparison.csv")
    println("\nSaving normalization summary table to: ", csv_path)
    open(csv_path, "w") do file_io
        write(file_io, "model,method,tau,mean_dimension_k24\n")
        for method in normalization_methods
            for tau in tau_grid
                _, mis_slice = get_tau_slice(mis_raw, tau; feature_cols = [2, 3])
                mis_norm = apply_normalization(mis_slice, method)
                mis_d = mean(dims(mis_norm; k = 24, tol = 0.01))
                @printf(file_io, "MIS,%s,%.3f,%.4f\n", string(method), tau, mis_d)

                _, hjsw_slice = get_tau_slice(hjsw_raw, tau; feature_cols = [2, 3, 4])
                hjsw_norm = apply_normalization(hjsw_slice, method)
                hjsw_d = mean(dims(hjsw_norm; k = 24, tol = 0.01))
                @printf(file_io, "HJSW,%s,%.3f,%.4f\n", string(method), tau, hjsw_d)
            end
        end
    end

    println("=== 02_normalization_tests Completed Successfully! ===")
    return (
        mis_norm_results = mis_norm_results,
        hjsw_norm_results = hjsw_norm_results
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_normalization_experiment()
end
