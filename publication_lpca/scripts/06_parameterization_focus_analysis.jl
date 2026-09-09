"""
    Script 06: Parameterization Focus Analysis (2x2 Grid, K in [3, 100], tau up to 7 fm/c)
    Investigates coordinate parameterization invariance across a dense sweep of nearest neighbors K.
    Produces 2x2 publication figures in vector PDF format:
    - Panel (a): Focused on (T, A, B) or (T, A) in color (:phase colormap), others in gray.
    - Panel (b): Focused on (w, A, B) or (w, A) in color (:phase colormap), others in gray.
    - Panel (c): Focused on (10T, A, B) or (10T, A) in color (:phase colormap), others in gray.
    - Panel (d): Focused on (10w, 2A, B) or (10w, 2A) in color (:phase colormap), others in gray.
    Runs for normalization methods: :none (raw), :max, :minmax, :zscore for both MIS and HJSW models.
"""

using AttractorsQGP
using CairoMakie
using Printf
using Statistics

if !isdefined(Main, :PublicationLPCA)
    include(joinpath(@__DIR__, "..", "src", "PublicationLPCA.jl"))
end
using .PublicationLPCA

function run_parameterization_focus_experiment(;
    mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
    hjsw_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
    output_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "parameterization_focus"),
    results_directory::AbstractString = joinpath(@__DIR__, "..", "results")
)
    mkpath(output_directory)
    mkpath(results_directory)
    CairoMakie.activate!()

    println("=== Starting 06_parameterization_focus_analysis Experiment (2x2 Grid) ===")

    # Dense K sweep with step of 5 up to 100
    k_values = collect(5:5:100)
    # Dense tau grid covering early kinetics up to 7.0 fm/c
    tau_grid = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]
    norm_methods = Symbol[:none, :max, :minmax, :zscore]

    # -------------------------------------------------------------
    # A. Conformal MIS Model (4 parameterizations: (T,A), (w,A), (10T,A), (10w,2A))
    # -------------------------------------------------------------
    println("\n[1/2] Running 2x2 parameterization focus analysis for Conformal MIS...")
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)

    for method in norm_methods
        method_str = string(method)
        println("  -> Processing normalization :$(method_str)...")
        sweep_res = compute_parameterization_k_sweep(
            mis_variants,
            k_values,
            tau_grid;
            normalize_method = method,
            tolerance = 0.01
        )

        fig_2x2 = plot_parameterization_focus_2x2(
            sweep_res;
            colormap = :phase,
            figure_size = (1180, 880),
            max_tau = 7.0,
            draw_tunnel = true
        )

        pdf_path = joinpath(output_directory, "parameterization_focus_mis_$(method_str).pdf")
        save(pdf_path, fig_2x2)
        println("     Saved 2x2 PDF: $(pdf_path)")
    end

    # -------------------------------------------------------------
    # B. HJSW Model (4 parameterizations: (T,A,B), (w,A,B), (10T,A,B), (10w,2A,B))
    # -------------------------------------------------------------
    println("\n[2/2] Running 2x2 parameterization focus analysis for HJSW...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)

    for method in norm_methods
        method_str = string(method)
        println("  -> Processing normalization :$(method_str)...")
        sweep_res = compute_parameterization_k_sweep(
            hjsw_variants,
            k_values,
            tau_grid;
            normalize_method = method,
            tolerance = 0.01
        )

        fig_2x2 = plot_parameterization_focus_2x2(
            sweep_res;
            colormap = :phase,
            figure_size = (1180, 880),
            max_tau = 7.0,
            draw_tunnel = true
        )

        pdf_path = joinpath(output_directory, "parameterization_focus_hjsw_$(method_str).pdf")
        save(pdf_path, fig_2x2)
        println("     Saved 2x2 PDF: $(pdf_path)")
    end

    println("\n=== Completed 06_parameterization_focus_analysis Experiment ===")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_parameterization_focus_experiment()
end
