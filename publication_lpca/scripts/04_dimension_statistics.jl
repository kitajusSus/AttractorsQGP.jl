"""
    Script 04: In-Depth Dimension Statistics & dims() Evaluation
    Investigates how the continuous average <d> relates to discrete per-point dimensions d_i in {1, ..., D},
    analyzing medians, quantiles, discrete state fractions P(d=m), and eigenvalue tolerance sensitivity.
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

function run_dimension_statistics_experiment(;
    mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
    hjsw_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
    output_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "dimension_distribution"),
    phase_space_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "phase_space"),
    results_directory::AbstractString = joinpath(@__DIR__, "..", "results")
)
    mkpath(output_directory)
    mkpath(phase_space_directory)
    mkpath(results_directory)
    CairoMakie.activate!()

    println("=== Starting 04_dimension_statistics Experiment ===")

    k_fixed = 24
    tau_grid = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.5, 10.0]

    # -------------------------------------------------------------
    # A. Conformal MIS: Distribution & Discrete State Fractions
    # -------------------------------------------------------------
    println("\n[1/4] Analyzing MIS dimension statistics and discrete populations...")
    mis_raw = load_hydro_dataset(mis_dataset_path)

    mis_dist_result = analyze_dimension_distribution(
        mis_raw,
        k_fixed,
        tau_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        tolerance = 0.01
    )

    fig_mis_dist = plot_dimension_distribution(
        mis_dist_result;
        model_name = "Conformal MIS"
    )
    save(joinpath(output_directory, "dimension_statistics_and_fractions_mis.pdf"), fig_mis_dist)

    fig_mis_mean_single = plot_dimension_mean_single(mis_dist_result)
    save(joinpath(output_directory, "dimension_mean_and_quantiles_mis.pdf"), fig_mis_mean_single)

    fig_mis_pop_single = plot_dimension_populations_single(mis_dist_result)
    save(joinpath(output_directory, "dimension_discrete_populations_mis.pdf"), fig_mis_pop_single)

    # -------------------------------------------------------------
    # B. HJSW: Distribution & Discrete State Fractions (3D -> 2D -> 1D)
    # -------------------------------------------------------------
    println("\n[2/4] Analyzing HJSW dimension statistics and discrete populations...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)

    hjsw_dist_result = analyze_dimension_distribution(
        hjsw_raw,
        k_fixed,
        tau_grid;
        feature_indices = [2, 3, 4],
        normalize_method = :max,
        tolerance = 0.01
    )

    fig_hjsw_dist = plot_dimension_distribution(
        hjsw_dist_result;
        model_name = "HJSW Model"
    )
    save(joinpath(output_directory, "dimension_statistics_and_fractions_hjsw.pdf"), fig_hjsw_dist)

    fig_hjsw_mean_single = plot_dimension_mean_single(hjsw_dist_result)
    save(joinpath(output_directory, "dimension_mean_and_quantiles_hjsw.pdf"), fig_hjsw_mean_single)

    fig_hjsw_pop_single = plot_dimension_populations_single(hjsw_dist_result)
    save(joinpath(output_directory, "dimension_discrete_populations_hjsw.pdf"), fig_hjsw_pop_single)

    # -------------------------------------------------------------
    # C. Eigenvalue Tolerance Sensitivity (tol in [1e-3, 5e-3, 1e-2, 2e-2])
    # -------------------------------------------------------------
    println("\n[3/4] Analyzing tolerance parameter (tol) sensitivity...")
    tolerances = [0.001, 0.005, 0.01, 0.02]
    palette = [:dodgerblue, :forestgreen, :darkorange, :crimson]

    set_publication_theme()
    fig_tol = Figure(size = (1100, 520))

    ax_mis_tol = Axis(
        fig_tol[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )

    ax_hjsw_tol = Axis(
        fig_tol[1, 2],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )

    for (t_idx, tol_val) in enumerate(tolerances)
        color = palette[t_idx]

        # MIS
        mis_tol_means = Float64[]
        for tau in tau_grid
            _, s = get_tau_slice(mis_raw, tau; feature_cols = [2, 3])
            sn = apply_normalization(s, :max)
            push!(mis_tol_means, mean(dims(sn; k = k_fixed, tol = tol_val)))
        end
        lines!(ax_mis_tol, tau_grid, mis_tol_means; color = color, linewidth = 2.5, label = L"\mathrm{tol} = %$(tol_val)")

        # HJSW
        hjsw_tol_means = Float64[]
        for tau in tau_grid
            _, s = get_tau_slice(hjsw_raw, tau; feature_cols = [2, 3, 4])
            sn = apply_normalization(s, :max)
            push!(hjsw_tol_means, mean(dims(sn; k = k_fixed, tol = tol_val)))
        end
        lines!(ax_hjsw_tol, tau_grid, hjsw_tol_means; color = color, linewidth = 2.5, label = L"\mathrm{tol} = %$(tol_val)")
    end

    axislegend(ax_mis_tol, position = :rt)
    save(joinpath(output_directory, "eigenvalue_tolerance_cutoff_sensitivity.pdf"), fig_tol)

    fig_mis_tol_single = Figure(size = (900, 560))
    ax_m_s = Axis(
        fig_mis_tol_single[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )
    for (t_idx, tol_val) in enumerate(tolerances)
        means = Float64[mean(dims(apply_normalization(get_tau_slice(mis_raw, tau; feature_cols = [2, 3])[2], :max); k = k_fixed, tol = tol_val)) for tau in tau_grid]
        lines!(ax_m_s, tau_grid, means; color = palette[t_idx], linewidth = 2.5, label = L"\mathrm{tol} = %$(tol_val)")
    end
    axislegend(ax_m_s, position = :rt)
    save(joinpath(output_directory, "eigenvalue_tolerance_cutoff_sensitivity_mis.pdf"), fig_mis_tol_single)

    fig_hjsw_tol_single = Figure(size = (900, 560))
    ax_h_s = Axis(
        fig_hjsw_tol_single[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )
    for (t_idx, tol_val) in enumerate(tolerances)
        means = Float64[mean(dims(apply_normalization(get_tau_slice(hjsw_raw, tau; feature_cols = [2, 3, 4])[2], :max); k = k_fixed, tol = tol_val)) for tau in tau_grid]
        lines!(ax_h_s, tau_grid, means; color = palette[t_idx], linewidth = 2.5, label = L"\mathrm{tol} = %$(tol_val)")
    end
    axislegend(ax_h_s, position = :rt)
    save(joinpath(output_directory, "eigenvalue_tolerance_cutoff_sensitivity_hjsw.pdf"), fig_hjsw_tol_single)

    # -------------------------------------------------------------
    # D. Phase Space Visualization colored by Local Dimension
    # -------------------------------------------------------------
    println("\n[4/4] Generating Phase Space scatter plots colored by LPCA dimension...")
    key_times = [0.25, 0.65, 2.5]
    for tau_snap in key_times
        slice_data = compute_pointwise_dimensions(
            mis_raw,
            tau_snap;
            feature_indices = [2, 3],
            k_neighbor = k_fixed,
            tolerance = 0.01,
            normalize_method = :max
        )
        fig_ps = plot_colored_phase_space_slice_2d(
            slice_data;
            x_col_idx = 1,
            y_col_idx = 2,
            x_label = L"T\,[\mathrm{fm}^{-1}]",
            y_label = L"\mathcal{A}"
        )
        save(joinpath(phase_space_directory, @sprintf("phase_space_lpca_dimension_map_mis_tau_%.2f.pdf", tau_snap)), fig_ps)
    end

    # -------------------------------------------------------------
    # E. Save CSV Summary Table
    # -------------------------------------------------------------
    csv_path = joinpath(results_directory, "dimension_statistics.csv")
    println("\nSaving dimension statistics to: ", csv_path)
    open(csv_path, "w") do file_io
        write(file_io, "model,tau,mean,std,median,q25,q75,P_d1,P_d2,P_d3\n")
        for (idx, tau) in enumerate(mis_dist_result.tau_values)
            @printf(file_io, "MIS,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,0.0000\n",
                tau,
                mis_dist_result.mean_dimension[idx],
                mis_dist_result.std_dimension[idx],
                mis_dist_result.median_dimension[idx],
                mis_dist_result.q25_dimension[idx],
                mis_dist_result.q75_dimension[idx],
                mis_dist_result.dimension_fractions[idx, 1],
                mis_dist_result.dimension_fractions[idx, 2]
            )
        end
        for (idx, tau) in enumerate(hjsw_dist_result.tau_values)
            @printf(file_io, "HJSW,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                tau,
                hjsw_dist_result.mean_dimension[idx],
                hjsw_dist_result.std_dimension[idx],
                hjsw_dist_result.median_dimension[idx],
                hjsw_dist_result.q25_dimension[idx],
                hjsw_dist_result.q75_dimension[idx],
                hjsw_dist_result.dimension_fractions[idx, 1],
                hjsw_dist_result.dimension_fractions[idx, 2],
                hjsw_dist_result.dimension_fractions[idx, 3]
            )
        end
    end

    println("=== 04_dimension_statistics Completed Successfully! ===")
    return (
        mis_distribution = mis_dist_result,
        hjsw_distribution = hjsw_dist_result
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_dimension_statistics_experiment()
end
