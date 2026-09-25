"""
    Script 01: K-Dependency Evaluation for Local PCA
    Evaluates how the choice of nearest neighbors K affects the estimated local dimension d(tau)
    for Conformal MIS and HJSW models.

    Generates strictly separate PDF plots for:
    - Discrete Hard-Threshold Local Dimension: dims() -> d(tau)
    - Continuous Participation Ratio Dimension: pr() -> d_PR(tau)
    Both for:
    1. Paired stability bands between (K_base, K_expanded = 2 * K_base)
    2. Direct multi-K sweep curves across chosen K in {6, 12, 24, 48, 100}
    Across physical coordinates (T, A, B) and dimensionless coordinates (w, A, B).
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
        (100, 200),
    ]

    # 2. Chosen individual K values for direct sweep curves
    k_sweep_values = [6, 12, 24, 48, 100]

    # 3. Select uniform evaluation proper times tau in [0.2, 6.0] (cut strictly at tau_max = 6.0 fm)
    tau_grid = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 6.0]
    tau_max = 6.0

    # -------------------------------------------------------------
    # A. Conformal MIS Analysis (Initial 2D Phase Space)
    # -------------------------------------------------------------
    println("\n[1/2] Analyzing Conformal MIS model...")
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)

    # A1. MIS Physical coordinates (T, A)
    println("  - Computing K-dependency for MIS Physical (T, A)...")
    # A1.1 dims() paired bands
    mis_phys_dims_results = evaluate_k_dependency(
        mis_variants.physical,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        tolerance = 0.01
    )
    fig_mis_phys_dims = plot_k_dependency_bands(
        mis_phys_dims_results;
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_dependency_mis_physical_coordinates_TA_dims.pdf"), fig_mis_phys_dims)
    save(joinpath(output_directory, "k_dependency_mis_physical_coordinates_TA.pdf"), fig_mis_phys_dims)

    # A1.2 pr() paired bands
    mis_phys_pr_results = evaluate_k_dependency_pr(
        mis_variants.physical,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        use_density_weights = false
    )
    fig_mis_phys_pr = plot_k_dependency_bands(
        mis_phys_pr_results;
        ylabel = L"\text{Participation Ratio Dimension } \langle d_{\mathrm{PR}} \rangle",
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_dependency_mis_physical_coordinates_TA_pr.pdf"), fig_mis_phys_pr)

    # A1.3 Direct multi-K sweep curves
    fig_mis_phys_sweep_dims = plot_dims_k_dependency(
        mis_variants.physical,
        k_sweep_values,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        tolerance = 0.01,
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_sweep_mis_physical_coordinates_TA_dims.pdf"), fig_mis_phys_sweep_dims)

    fig_mis_phys_sweep_pr = plot_pr_k_dependency(
        mis_variants.physical,
        k_sweep_values,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        use_density_weights = false,
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_sweep_mis_physical_coordinates_TA_pr.pdf"), fig_mis_phys_sweep_pr)

    # A2. MIS Dimensionless scaling coordinates (w, A)
    println("  - Computing K-dependency for MIS Dimensionless (w, A)...")
    # A2.1 dims() paired bands
    mis_dimless_dims_results = evaluate_k_dependency(
        mis_variants.dimensionless,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        tolerance = 0.01
    )
    fig_mis_dimless_dims = plot_k_dependency_bands(
        mis_dimless_dims_results;
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_dependency_mis_dimensionless_coordinates_wA_dims.pdf"), fig_mis_dimless_dims)
    save(joinpath(output_directory, "k_dependency_mis_dimensionless_coordinates_wA.pdf"), fig_mis_dimless_dims)

    # A2.2 pr() paired bands
    mis_dimless_pr_results = evaluate_k_dependency_pr(
        mis_variants.dimensionless,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        use_density_weights = false
    )
    fig_mis_dimless_pr = plot_k_dependency_bands(
        mis_dimless_pr_results;
        ylabel = L"\text{Participation Ratio Dimension } \langle d_{\mathrm{PR}} \rangle",
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_dependency_mis_dimensionless_coordinates_wA_pr.pdf"), fig_mis_dimless_pr)

    # A2.3 Direct multi-K sweep curves
    fig_mis_dimless_sweep_dims = plot_dims_k_dependency(
        mis_variants.dimensionless,
        k_sweep_values,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        tolerance = 0.01,
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_sweep_mis_dimensionless_coordinates_wA_dims.pdf"), fig_mis_dimless_sweep_dims)

    fig_mis_dimless_sweep_pr = plot_pr_k_dependency(
        mis_variants.dimensionless,
        k_sweep_values,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        use_density_weights = false,
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_sweep_mis_dimensionless_coordinates_wA_pr.pdf"), fig_mis_dimless_sweep_pr)

    # -------------------------------------------------------------
    # B. HJSW Analysis (Initial 3D Phase Space)
    # -------------------------------------------------------------
    println("\n[2/2] Analyzing HJSW model...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)

    # B1. HJSW Physical coordinates (T, A, B)
    println("  - Computing K-dependency for HJSW Physical (T, A, B)...")
    # B1.1 dims() paired bands
    hjsw_phys_dims_results = evaluate_k_dependency(
        hjsw_variants.physical,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        tolerance = 0.01
    )
    fig_hjsw_phys_dims = plot_k_dependency_bands(
        hjsw_phys_dims_results;
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_dependency_hjsw_physical_coordinates_TAB_dims.pdf"), fig_hjsw_phys_dims)
    save(joinpath(output_directory, "k_dependency_hjsw_physical_coordinates_TAB.pdf"), fig_hjsw_phys_dims)

    # B1.2 pr() paired bands
    hjsw_phys_pr_results = evaluate_k_dependency_pr(
        hjsw_variants.physical,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        use_density_weights = false
    )
    fig_hjsw_phys_pr = plot_k_dependency_bands(
        hjsw_phys_pr_results;
        ylabel = L"\text{Participation Ratio Dimension } \langle d_{\mathrm{PR}} \rangle",
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_dependency_hjsw_physical_coordinates_TAB_pr.pdf"), fig_hjsw_phys_pr)

    # B1.3 Direct multi-K sweep curves
    fig_hjsw_phys_sweep_dims = plot_dims_k_dependency(
        hjsw_variants.physical,
        k_sweep_values,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        tolerance = 0.01,
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_sweep_hjsw_physical_coordinates_TAB_dims.pdf"), fig_hjsw_phys_sweep_dims)

    fig_hjsw_phys_sweep_pr = plot_pr_k_dependency(
        hjsw_variants.physical,
        k_sweep_values,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        use_density_weights = false,
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_sweep_hjsw_physical_coordinates_TAB_pr.pdf"), fig_hjsw_phys_sweep_pr)

    # B2. HJSW Dimensionless coordinates (w, A, B)
    println("  - Computing K-dependency for HJSW Dimensionless (w, A, B)...")
    # B2.1 dims() paired bands
    hjsw_dimless_dims_results = evaluate_k_dependency(
        hjsw_variants.dimensionless,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        tolerance = 0.01
    )
    fig_hjsw_dimless_dims = plot_k_dependency_bands(
        hjsw_dimless_dims_results;
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_dependency_hjsw_dimensionless_coordinates_wAB_dims.pdf"), fig_hjsw_dimless_dims)
    save(joinpath(output_directory, "k_dependency_hjsw_dimensionless_coordinates_wAB.pdf"), fig_hjsw_dimless_dims)

    # B2.2 pr() paired bands
    hjsw_dimless_pr_results = evaluate_k_dependency_pr(
        hjsw_variants.dimensionless,
        k_pairs,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        use_density_weights = false
    )
    fig_hjsw_dimless_pr = plot_k_dependency_bands(
        hjsw_dimless_pr_results;
        ylabel = L"\text{Participation Ratio Dimension } \langle d_{\mathrm{PR}} \rangle",
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_dependency_hjsw_dimensionless_coordinates_wAB_pr.pdf"), fig_hjsw_dimless_pr)

    # B2.3 Direct multi-K sweep curves
    fig_hjsw_dimless_sweep_dims = plot_dims_k_dependency(
        hjsw_variants.dimensionless,
        k_sweep_values,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        tolerance = 0.01,
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_sweep_hjsw_dimensionless_coordinates_wAB_dims.pdf"), fig_hjsw_dimless_sweep_dims)

    fig_hjsw_dimless_sweep_pr = plot_pr_k_dependency(
        hjsw_variants.dimensionless,
        k_sweep_values,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        use_density_weights = false,
        tau_max = tau_max
    )
    save(joinpath(output_directory, "k_sweep_hjsw_dimensionless_coordinates_wAB_pr.pdf"), fig_hjsw_dimless_sweep_pr)

    # -------------------------------------------------------------
    # C. Save CSV Summary Table
    # -------------------------------------------------------------
    summary_csv_path = joinpath(results_directory, "k_dependency_summary.csv")
    println("\nSaving summary table to: ", summary_csv_path)
    open(summary_csv_path, "w") do file_io
        write(file_io, "model,method,representation,k_base,k_expanded,tau,mean_k1,mean_k2,relative_difference_percent\n")
        # MIS dims
        for res in mis_phys_dims_results
            for (idx, tau) in enumerate(res.tau_values)
                @printf(file_io, "MIS,dims,physical,%d,%d,%.3f,%.4f,%.4f,%.2f\n",
                    res.k_base, res.k_expanded, tau, res.mean_dimension_k1[idx], res.mean_dimension_k2[idx], res.relative_difference[idx])
            end
        end
        for res in mis_dimless_dims_results
            for (idx, tau) in enumerate(res.tau_values)
                @printf(file_io, "MIS,dims,dimensionless,%d,%d,%.3f,%.4f,%.4f,%.2f\n",
                    res.k_base, res.k_expanded, tau, res.mean_dimension_k1[idx], res.mean_dimension_k2[idx], res.relative_difference[idx])
            end
        end
        # MIS pr
        for res in mis_phys_pr_results
            for (idx, tau) in enumerate(res.tau_values)
                @printf(file_io, "MIS,pr,physical,%d,%d,%.3f,%.4f,%.4f,%.2f\n",
                    res.k_base, res.k_expanded, tau, res.mean_dimension_k1[idx], res.mean_dimension_k2[idx], res.relative_difference[idx])
            end
        end
        for res in mis_dimless_pr_results
            for (idx, tau) in enumerate(res.tau_values)
                @printf(file_io, "MIS,pr,dimensionless,%d,%d,%.3f,%.4f,%.4f,%.2f\n",
                    res.k_base, res.k_expanded, tau, res.mean_dimension_k1[idx], res.mean_dimension_k2[idx], res.relative_difference[idx])
            end
        end
        # HJSW dims
        for res in hjsw_phys_dims_results
            for (idx, tau) in enumerate(res.tau_values)
                @printf(file_io, "HJSW,dims,physical,%d,%d,%.3f,%.4f,%.4f,%.2f\n",
                    res.k_base, res.k_expanded, tau, res.mean_dimension_k1[idx], res.mean_dimension_k2[idx], res.relative_difference[idx])
            end
        end
        for res in hjsw_dimless_dims_results
            for (idx, tau) in enumerate(res.tau_values)
                @printf(file_io, "HJSW,dims,dimensionless,%d,%d,%.3f,%.4f,%.4f,%.2f\n",
                    res.k_base, res.k_expanded, tau, res.mean_dimension_k1[idx], res.mean_dimension_k2[idx], res.relative_difference[idx])
            end
        end
        # HJSW pr
        for res in hjsw_phys_pr_results
            for (idx, tau) in enumerate(res.tau_values)
                @printf(file_io, "HJSW,pr,physical,%d,%d,%.3f,%.4f,%.4f,%.2f\n",
                    res.k_base, res.k_expanded, tau, res.mean_dimension_k1[idx], res.mean_dimension_k2[idx], res.relative_difference[idx])
            end
        end
        for res in hjsw_dimless_pr_results
            for (idx, tau) in enumerate(res.tau_values)
                @printf(file_io, "HJSW,pr,dimensionless,%d,%d,%.3f,%.4f,%.4f,%.2f\n",
                    res.k_base, res.k_expanded, tau, res.mean_dimension_k1[idx], res.mean_dimension_k2[idx], res.relative_difference[idx])
            end
        end
    end

    println("=== 01_k_dependency Completed Successfully! ===")
    return (
        mis_physical_dims = mis_phys_dims_results,
        mis_physical_pr = mis_phys_pr_results,
        mis_dimensionless_dims = mis_dimless_dims_results,
        mis_dimensionless_pr = mis_dimless_pr_results,
        hjsw_physical_dims = hjsw_phys_dims_results,
        hjsw_physical_pr = hjsw_phys_pr_results,
        hjsw_dimensionless_dims = hjsw_dimless_dims_results,
        hjsw_dimensionless_pr = hjsw_dimless_pr_results,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_k_dependency_experiment()
end
