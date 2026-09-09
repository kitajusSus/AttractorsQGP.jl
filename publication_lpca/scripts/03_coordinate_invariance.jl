"""
    Script 03: Coordinate & Scale Invariance Verification
    Investigates whether local PCA dimension estimation is invariant under:
    - Scalar temperature rescaling: T -> 10*T
    - Dimensionless coordinate substitution: T -> w = tau*T
    Comparing normalized representations (:max) against raw unnormalized units (:none).
"""

using AttractorsQGP
using CairoMakie
using Printf
using Statistics

if !isdefined(Main, :PublicationLPCA)
    include(joinpath(@__DIR__, "..", "src", "PublicationLPCA.jl"))
end
using .PublicationLPCA

function run_coordinate_invariance_experiment(;
    mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
    hjsw_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
    output_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "coordinate_invariance"),
    results_directory::AbstractString = joinpath(@__DIR__, "..", "results")
)
    mkpath(output_directory)
    mkpath(results_directory)
    CairoMakie.activate!()

    println("=== Starting 03_coordinate_invariance Experiment ===")

    k_fixed = 24
    tau_grid = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.5, 10.0]

    # -------------------------------------------------------------
    # A. Conformal MIS Invariance Tests
    # -------------------------------------------------------------
    println("\n[1/2] Testing coordinate invariance for Conformal MIS...")
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)

    # A1. Normalized (:max)
    mis_norm_invariance = test_coordinate_invariance(
        mis_variants,
        k_fixed,
        tau_grid;
        normalize_method = :max,
        tolerance = 0.01
    )
    # A2. Raw unscaled (:none)
    mis_raw_invariance = test_coordinate_invariance(
        mis_variants,
        k_fixed,
        tau_grid;
        normalize_method = :none,
        tolerance = 0.01
    )

    fig_mis_invariance = plot_coordinate_invariance(
        mis_norm_invariance,
        mis_raw_invariance;
        model_name_display = "Conformal MIS"
    )
    save(joinpath(output_directory, "coordinate_and_scale_invariance_mis.pdf"), fig_mis_invariance)

    fig_mis_single_norm = plot_coordinate_invariance_single(mis_norm_invariance)
    save(joinpath(output_directory, "coordinate_invariance_normalized_max_mis.pdf"), fig_mis_single_norm)

    fig_mis_single_raw = plot_coordinate_invariance_single(mis_raw_invariance)
    save(joinpath(output_directory, "coordinate_invariance_raw_none_mis.pdf"), fig_mis_single_raw)

    println(@sprintf("  -> MIS Normalized max diff (scaled): %.2e, max diff (dimless): %.2e",
        mis_norm_invariance.max_difference_scaled, mis_norm_invariance.max_difference_dimensionless))
    println(@sprintf("  -> MIS Raw unscaled max diff (scaled): %.2f, max diff (dimless): %.2f",
        mis_raw_invariance.max_difference_scaled, mis_raw_invariance.max_difference_dimensionless))

    # -------------------------------------------------------------
    # B. HJSW Model Invariance Tests
    # -------------------------------------------------------------
    println("\n[2/2] Testing coordinate invariance for HJSW...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)

    # B1. Normalized (:max)
    hjsw_norm_invariance = test_coordinate_invariance(
        hjsw_variants,
        k_fixed,
        tau_grid;
        normalize_method = :max,
        tolerance = 0.01
    )
    # B2. Raw unscaled (:none)
    hjsw_raw_invariance = test_coordinate_invariance(
        hjsw_variants,
        k_fixed,
        tau_grid;
        normalize_method = :none,
        tolerance = 0.01
    )

    fig_hjsw_invariance = plot_coordinate_invariance(
        hjsw_norm_invariance,
        hjsw_raw_invariance;
        model_name_display = "HJSW Model"
    )
    save(joinpath(output_directory, "coordinate_and_scale_invariance_hjsw.pdf"), fig_hjsw_invariance)

    fig_hjsw_single_norm = plot_coordinate_invariance_single(hjsw_norm_invariance)
    save(joinpath(output_directory, "coordinate_invariance_normalized_max_hjsw.pdf"), fig_hjsw_single_norm)

    fig_hjsw_single_raw = plot_coordinate_invariance_single(hjsw_raw_invariance)
    save(joinpath(output_directory, "coordinate_invariance_raw_none_hjsw.pdf"), fig_hjsw_single_raw)

    println(@sprintf("  -> HJSW Normalized max diff (scaled): %.2e, max diff (dimless): %.2e",
        hjsw_norm_invariance.max_difference_scaled, hjsw_norm_invariance.max_difference_dimensionless))
    println(@sprintf("  -> HJSW Raw unscaled max diff (scaled): %.2f, max diff (dimless): %.2f",
        hjsw_raw_invariance.max_difference_scaled, hjsw_raw_invariance.max_difference_dimensionless))

    # -------------------------------------------------------------
    # C. Save CSV Summary Table
    # -------------------------------------------------------------
    csv_path = joinpath(results_directory, "coordinate_invariance.csv")
    println("\nSaving invariance results to: ", csv_path)
    open(csv_path, "w") do file_io
        write(file_io, "model,norm_method,tau,d_physical,d_dimless,d_scaled10x,delta_scaled,delta_dimless\n")
        for (m_name, norm_res) in [("MIS", mis_norm_invariance), ("HJSW", hjsw_norm_invariance)]
            for (idx, tau) in enumerate(norm_res.tau_values)
                d_phys = norm_res.curves[:physical][idx]
                d_dim = norm_res.curves[:dimensionless][idx]
                d_sc = norm_res.curves[:scaled_10x][idx]
                @printf(file_io, "%s,%s,%.3f,%.4f,%.4f,%.4f,%.2e,%.2e\n",
                    m_name, string(norm_res.normalize_method), tau, d_phys, d_dim, d_sc, abs(d_sc - d_phys), abs(d_dim - d_phys))
            end
        end
        for (m_name, raw_res) in [("MIS", mis_raw_invariance), ("HJSW", hjsw_raw_invariance)]
            for (idx, tau) in enumerate(raw_res.tau_values)
                d_phys = raw_res.curves[:physical][idx]
                d_dim = raw_res.curves[:dimensionless][idx]
                d_sc = raw_res.curves[:scaled_10x][idx]
                @printf(file_io, "%s,%s,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    m_name, string(raw_res.normalize_method), tau, d_phys, d_dim, d_sc, abs(d_sc - d_phys), abs(d_dim - d_phys))
            end
        end
    end

    println("=== 03_coordinate_invariance Completed Successfully! ===")
    return (
        mis_norm = mis_norm_invariance,
        mis_raw = mis_raw_invariance,
        hjsw_norm = hjsw_norm_invariance,
        hjsw_raw = hjsw_raw_invariance
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_coordinate_invariance_experiment()
end
