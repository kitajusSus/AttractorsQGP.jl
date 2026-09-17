"""
    Script 01: K-Dependency Evaluation for Local PCA
    Evaluates how the choice of nearest neighbors K affects the estimated local dimension
    for Conformal MIS and HJSW models.
"""

using AttractorsQGP
using CairoMakie
using Printf
using Statistics

if !isdefined(Main, :PublicationLPCA)
    include(joinpath(@__DIR__, "..", "src", "PublicationLPCA.jl"))
end
using .PublicationLPCA

function run_k_dependency_experiment(;
    mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
    hjsw_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
    output_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "k_dependency"),
    results_directory::AbstractString = joinpath(@__DIR__, "..", "results")
)
    mkpath(output_directory)
    mkpath(results_directory)
    CairoMakie.activate!()

    println("=== Starting 01_k_dependency Experiment ===")

    # 1. Neighbor count pairs (K_base, K_expanded = 2 * K_base)
    k_pairs = [
        (3, 6),
        (6, 12),
        (12, 24),
        (24, 48),
        (100, 200)
    ]

    # 2. Select uniform evaluation proper times tau in [0.2, 6.0] (cut at 6 fm)
    tau_grid = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 6.0]

    # -------------------------------------------------------------
    # A. Conformal MIS Analysis
    # -------------------------------------------------------------
    println("\n[1/2] Analyzing Conformal MIS model...")
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)

    # A1. MIS Physical coordinates (T, A)
    println("  - Computing K-dependency for MIS (T, A)...")
    mis_physical_results = evaluate_k_dependency(
        mis_variants.physical,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        tolerance = 0.01
    )
    fig_mis_phys = plot_k_dependency_bands(mis_physical_results; tau_max = 6.0)
    save(joinpath(output_directory, "k_dependency_mis_physical_coordinates_TA.pdf"), fig_mis_phys)

    # A2. MIS Dimensionless scaling coordinates (w, A)
    println("  - Computing K-dependency for MIS (w, A)...")
    mis_dimless_results = evaluate_k_dependency(
        mis_variants.dimensionless,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        tolerance = 0.01
    )
    fig_mis_dimless = plot_k_dependency_bands(mis_dimless_results; tau_max = 6.0)
    save(joinpath(output_directory, "k_dependency_mis_dimensionless_coordinates_wA.pdf"), fig_mis_dimless)

    # -------------------------------------------------------------
    # B. HJSW Model Analysis
    # -------------------------------------------------------------
    println("\n[2/2] Analyzing HJSW model...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)

    # B1. HJSW Physical coordinates (T, A, B)
    println("  - Computing K-dependency for HJSW (T, A, B)...")
    hjsw_physical_results = evaluate_k_dependency(
        hjsw_variants.physical,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        tolerance = 0.01
    )
    fig_hjsw_phys = plot_k_dependency_bands(hjsw_physical_results; tau_max = 6.0)
    save(joinpath(output_directory, "k_dependency_hjsw_physical_coordinates_TAB.pdf"), fig_hjsw_phys)

    # B2. HJSW Dimensionless coordinates (w, A, B)
    println("  - Computing K-dependency for HJSW (w, A, B)...")
    hjsw_dimless_results = evaluate_k_dependency(
        hjsw_variants.dimensionless,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        tolerance = 0.01
    )
    fig_hjsw_dimless = plot_k_dependency_bands(hjsw_dimless_results; tau_max = 6.0)
    save(joinpath(output_directory, "k_dependency_hjsw_dimensionless_coordinates_wAB.pdf"), fig_hjsw_dimless)

    # -------------------------------------------------------------
    # C. Save CSV Summary Table
    # -------------------------------------------------------------
    summary_csv_path = joinpath(results_directory, "k_dependency_summary.csv")
    println("\nSaving summary table to: ", summary_csv_path)
    open(summary_csv_path, "w") do file_io
        write(file_io, "model,representation,k_base,k_expanded,tau,mean_k1,mean_k2,relative_difference_percent\n")
        for res in mis_physical_results
            for (idx, tau) in enumerate(res.tau_values)
                @printf(file_io, "MIS,physical,%d,%d,%.3f,%.4f,%.4f,%.2f\n",
                    res.k_base, res.k_expanded, tau, res.mean_dimension_k1[idx], res.mean_dimension_k2[idx], res.relative_difference[idx])
            end
        end
        for res in hjsw_physical_results
            for (idx, tau) in enumerate(res.tau_values)
                @printf(file_io, "HJSW,physical,%d,%d,%.3f,%.4f,%.4f,%.2f\n",
                    res.k_base, res.k_expanded, tau, res.mean_dimension_k1[idx], res.mean_dimension_k2[idx], res.relative_difference[idx])
            end
        end
    end

    println("=== 01_k_dependency Completed Successfully! ===")
    return (
        mis_physical = mis_physical_results,
        mis_dimensionless = mis_dimless_results,
        hjsw_physical = hjsw_physical_results,
        hjsw_dimensionless = hjsw_dimless_results
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_k_dependency_experiment()
end
