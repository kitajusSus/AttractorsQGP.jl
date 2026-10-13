"""
    generate_all_paper_figures.jl

    Regenerates all publication figures required for k_selection_paper.pdf from scratch:
    1. plots/phase_space/mis/grid_mis_physical_pr.pdf
    2. plots/phase_space/mis/grid_mis_physical_pr_hist.pdf
    3. plots/sample_size_comparison/mis_phase_space_k8_sample_size_comparison.pdf
    4. plots/dimension_map/map_mean_dimension_tau_vs_k_dual.pdf
    5. plots/k_dependency/k_sweep_mis_physical_coordinates_TA_pr.pdf
    6. plots/phase_space/hjsw/grid_hjsw_physical_pr.pdf
    7. plots/phase_space/hjsw/grid_hjsw_physical_pr_hist.pdf
"""

using AttractorsQGP
using CairoMakie
using LaTeXStrings
using Printf
using Random
using Statistics

function generate_all_figures(;
    mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
    hjsw_dataset_path::AbstractString = isfile(joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_10000_points.h5")) ?
        joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_10000_points.h5") :
        joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
    base_plot_dir::AbstractString = joinpath(@__DIR__, "..", "plots")
)
    CairoMakie.activate!()
    set_publication_theme()

    mis_dir = joinpath(base_plot_dir, "phase_space", "mis")
    hjsw_dir = joinpath(base_plot_dir, "phase_space", "hjsw")
    sample_size_dir = joinpath(base_plot_dir, "sample_size_comparison")
    dimension_map_dir = joinpath(base_plot_dir, "dimension_map")
    k_dep_dir = joinpath(base_plot_dir, "k_dependency")

    mkpath(mis_dir)
    mkpath(hjsw_dir)
    mkpath(sample_size_dir)
    mkpath(dimension_map_dir)
    mkpath(k_dep_dir)

    grid_taus = Float64[0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80, 1.00, 2.50]
    k_eval = 100
    colormap_choice = lpca_style

    println("=== [1/5] Loading MIS dataset and generating MIS Phase Space & Stacked Histograms ===")
    println("  MIS Path: ", mis_dataset_path)
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)
    mis_ds = mis_variants.physical

    # 1. MIS Phase Space Grid PR
    println("  -> Generating grid_mis_physical_pr.pdf (K=$k_eval, zscore)...")
    fig_mis_pr = plot_pr_phase_space_grid_2d(
        mis_ds, grid_taus;
        feature_indices = [2, 3],
        x_label = L"T\,[\mathrm{fm}^{-1}]",
        y_label = L"\mathcal{A}",
        colormap = colormap_choice,
        color_limits = (1.0, 2.0),
        normalize_method = :zscore,
        return_normalized = false,
        k = k_eval,
        n_calc = 10_000,
        n_plot = 5_000
    )
    mis_pr_path = joinpath(mis_dir, "grid_mis_physical_pr.pdf")
    save(mis_pr_path, fig_mis_pr)
    println("     Saved: ", mis_pr_path)

    # 2. MIS Stacked Histograms Grid
    println("  -> Generating grid_mis_physical_pr_hist.pdf (K=$k_eval, color_by=:initial_A)...")
    fig_mis_hist = plot_pr_dimension_histogram_grid(
        mis_ds, grid_taus;
        feature_indices = [2, 3],
        k_neighbor = k_eval,
        color_by = :initial_A,
        n_color_bins = 8,
        normalize_method = :zscore,
        colormap = colormap_choice,
        color_limits = (1.0, 2.0),
        n_calc = 10_000
    )
    mis_hist_path = joinpath(mis_dir, "grid_mis_physical_pr_hist.pdf")
    save(mis_hist_path, fig_mis_hist)
    println("     Saved: ", mis_hist_path)

    # 3. Sample Size Dilemma (Script 13 logic, K=8)
    println("\n=== [2/5] Generating Sample Size Comparison (K=8 across N=5k, 10k, 20k) ===")
    taus_sample = [0.25, 0.50, 1.00]
    sample_sizes = [5_000, 10_000, 20_000]
    c1 = AttractorsQGP._get_dim_color(1.0, 2.0, colormap_choice)
    c2 = AttractorsQGP._get_dim_color(2.0, 2.0, colormap_choice)

    fig_sample = Figure(size = (1500, 1350), figure_padding = (65, 55, 45, 45))
    colgap!(fig_sample.layout, 45)
    rowgap!(fig_sample.layout, 55)

    for (row_idx, tau) in enumerate(taus_sample)
        tau_str = @sprintf("%.2f", tau)
        for (col_idx, N) in enumerate(sample_sizes)
            slice_data = compute_pointwise_dimensions(
                mis_raw, tau;
                feature_indices = [2, 3],
                k_neighbor = 8,
                tolerance = 0.01,
                normalize_method = :none,
                return_normalized = false,
                max_points = N,
                seed = 42 + row_idx * 10 + col_idx
            )
            x_vals = slice_data.coordinates[:, 1]
            y_vals = slice_data.coordinates[:, 2]
            d_vals = slice_data.dimensions

            sub_title = L"\tau = %$(tau_str)\,\mathrm{fm}/c,\; N = %$(N/1000)\mathrm{k}"
            ax = Axis(
                fig_sample[row_idx, col_idx],
                title = sub_title,
                titlesize = 21,
                xlabel = row_idx == 3 ? L"T\,[\mathrm{fm}^{-1}]" : "",
                ylabel = col_idx == 1 ? L"\mathcal{A}" : "",
                xlabelsize = 20,
                ylabelsize = 22,
                xticklabelsize = 15,
                yticklabelsize = 15,
                xticks = LinearTicks(4),
                yticks = LinearTicks(4),
                xautolimitmargin = (0.04, 0.04),
                yautolimitmargin = (0.05, 0.05)
            )

            mask_d2 = isapprox.(d_vals, 2.0; atol = 0.1)
            mask_d1 = isapprox.(d_vals, 1.0; atol = 0.1)

            scatter!(ax, x_vals[mask_d2], y_vals[mask_d2]; color = (c2, 0.70), markersize = 5.5, strokewidth = 0.2, strokecolor = (:black, 0.2), label = L"d = 2")
            scatter!(ax, x_vals[mask_d1], y_vals[mask_d1]; color = (c1, 0.85), markersize = 5.5, strokewidth = 0.2, strokecolor = (:black, 0.2), label = L"d = 1")

            if row_idx == 1 && col_idx == 3
                axislegend(ax, position = :rt, labelsize = 16)
            end
        end
    end
    sample_pdf_path = joinpath(sample_size_dir, "mis_phase_space_k8_sample_size_comparison.pdf")
    save(sample_pdf_path, fig_sample)
    save(joinpath(sample_size_dir, "mis_phase_space_k8_sample_size_comparison.png"), fig_sample; px_per_unit = 2)
    println("     Saved: ", sample_pdf_path)

    # 4. Continuous Dimension Map & Power-Law Fit (Script 14 logic)
    println("\n=== [3/5] Generating Continuous Dimension Map across (tau, K) with Collapse Fit ===")
    all_taus = sort(unique(mis_raw[:, 1]))
    filter_taus = filter(t -> 0.2 <= t <= 3.0, all_taus)
    tau_sweep_grid = filter_taus[1:4:end]
    k_range = 4:2:100
    k_vec = collect(k_range)
    n_k = length(k_vec)

    sweep_res = sweep_k_dimensions_over_tau(
        mis_raw, k_range, tau_sweep_grid;
        feature_indices = [2, 3],
        normalize_method = :max,
        max_points = 5_000,
        seed = 42
    )

    tau_vals = sort(collect(keys(sweep_res.results)))
    n_tau = length(tau_vals)
    mat_dims = zeros(n_tau, n_k)
    mat_pr = zeros(n_tau, n_k)

    for (ti, tau) in enumerate(tau_vals)
        mat_dims[ti, :] = sweep_res.results[tau].mean_dims
        mat_pr[ti, :] = sweep_res.results[tau].mean_pr
    end

    tau_crit_dims = Float64[]
    for (ki, k) in enumerate(k_vec)
        col = mat_dims[:, ki]
        idx = findfirst(v -> v <= 1.02, col)
        push!(tau_crit_dims, isnothing(idx) ? tau_vals[argmin(col)] : tau_vals[idx])
    end

    log_k = log.(k_vec)
    log_tau = log.(tau_crit_dims)
    X = hcat(ones(n_k), log_k)
    coeffs_pow = X \ log_tau
    b_pow = coeffs_pow[2]
    a_pow = exp(coeffs_pow[1])
    fit_tau_pow = a_pow .* (k_vec .^ b_pow)
    ss_tot = sum((tau_crit_dims .- mean(tau_crit_dims)) .^ 2)
    ss_res = sum((tau_crit_dims .- fit_tau_pow) .^ 2)
    r2_pow = 1.0 - (ss_res / ss_tot)
    fit_label = @sprintf("\\tau^*(K) = %.2f \\cdot K^{%.2f}\\; (R^2 = %.3f)", a_pow, b_pow, r2_pow)

    fig_dual = Figure(size = (1600, 680), figure_padding = (65, 55, 45, 45))
    colgap!(fig_dual.layout, 45)

    ax_a = Axis(fig_dual[1, 1], title = L"\text{(a) Cutoff Dimension } \langle d \rangle", titlesize = 21, xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"K", xlabelsize = 22, ylabelsize = 22, xticklabelsize = 16, yticklabelsize = 16)
    hm_a = heatmap!(ax_a, tau_vals, k_vec, mat_dims; colormap = :viridis, colorrange = (1.0, 2.0))
    contour!(ax_a, tau_vals, k_vec, mat_dims; levels = [1.1, 1.3, 1.5, 1.7, 1.9], color = (:white, 0.45), linewidth = 1.2)
    scatter!(ax_a, tau_crit_dims, k_vec; color = :cyan, markersize = 6, strokewidth = 0.5, strokecolor = :black, label = L"\text{Data } (\langle d \rangle \le 1.02)")
    lines!(ax_a, fit_tau_pow, k_vec; color = :crimson, linewidth = 3.2, label = LaTeXString(fit_label))
    axislegend(ax_a, position = :rt, labelsize = 14, bgcolor = (:white, 0.85))
    Colorbar(fig_dual[1, 2], hm_a; label = L"\langle d \rangle", labelsize = 18, ticklabelsize = 14)

    ax_b = Axis(fig_dual[1, 3], title = L"\text{(b) Participation Ratio } \langle d_{\mathrm{PR}} \rangle", titlesize = 21, xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"K", xlabelsize = 22, ylabelsize = 22, xticklabelsize = 16, yticklabelsize = 16)
    hm_b = heatmap!(ax_b, tau_vals, k_vec, mat_pr; colormap = :plasma, colorrange = (1.0, 2.0))
    contour!(ax_b, tau_vals, k_vec, mat_pr; levels = [1.05, 1.15, 1.3, 1.5, 1.7, 1.9], color = (:white, 0.5), linewidth = 1.2)
    lines!(ax_b, fit_tau_pow, k_vec; color = :cyan, linewidth = 3.0, linestyle = :dash, label = L"\text{Fitted 1D Boundary}")
    axislegend(ax_b, position = :rt, labelsize = 14, bgcolor = (:white, 0.85))
    Colorbar(fig_dual[1, 4], hm_b; label = L"\langle d_{\mathrm{PR}} \rangle", labelsize = 18, ticklabelsize = 14)

    map_dual_path = joinpath(dimension_map_dir, "map_mean_dimension_tau_vs_k_dual.pdf")
    save(map_dual_path, fig_dual)
    save(joinpath(dimension_map_dir, "map_mean_dimension_tau_vs_k_dual.png"), fig_dual; px_per_unit = 2)
    println("     Saved: ", map_dual_path)

    # 5. Multiscale K-dependency curves (Script 01 logic)
    println("\n=== [4/5] Generating Multiscale K-Dependency Curves with Uncertainty Bands ===")
    k_sweep_values = [4, 8, 16, 24, 32, 48, 64, 80, 100, 128]
    tau_k_dep_grid = Float64[0.20, 0.25, 0.35, 0.50, 0.75, 1.00, 1.50, 2.50]
    fig_k_dep = plot_pr_k_dependency(
        mis_ds,
        k_sweep_values,
        tau_k_dep_grid;
        feature_indices = [2, 3],
        normalize_method = :zscore,
        tau_max = 5.0
    )
    k_dep_path = joinpath(k_dep_dir, "k_sweep_mis_physical_coordinates_TA_pr.pdf")
    save(k_dep_path, fig_k_dep)
    println("     Saved: ", k_dep_path)

    # 6. HJSW Model (3D Phase Space & Stacked Histograms)
    println("\n=== [5/5] Loading HJSW dataset and generating HJSW 3D Phase Space & Histograms ===")
    println("  HJSW Path: ", hjsw_dataset_path)
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)
    hjsw_ds = hjsw_variants.physical

    println("  -> Generating grid_hjsw_physical_pr.pdf (3D Scatter, K=$k_eval)...")
    fig_hjsw_pr = plot_pr_phase_space_grid_3d(
        hjsw_ds, grid_taus;
        feature_indices = [2, 3, 4],
        x_label = L"T\,[\mathrm{MeV}]",
        y_label = L"\mathcal{A}",
        z_label = L"\mathcal{B}",
        colormap = colormap_choice,
        color_limits = (1.0, 3.0),
        normalize_method = :zscore,
        return_normalized = false,
        k = k_eval,
        n_calc = 10_000,
        n_plot = 5_000
    )
    hjsw_pr_path = joinpath(hjsw_dir, "grid_hjsw_physical_pr.pdf")
    save(hjsw_pr_path, fig_hjsw_pr)
    println("     Saved: ", hjsw_pr_path)

    println("  -> Generating grid_hjsw_physical_pr_hist.pdf (Stacked Histograms, K=$k_eval)...")
    fig_hjsw_hist = plot_pr_dimension_histogram_grid(
        hjsw_ds, grid_taus;
        feature_indices = [2, 3, 4],
        k_neighbor = k_eval,
        color_by = :initial_A,
        n_color_bins = 8,
        normalize_method = :zscore,
        colormap = colormap_choice,
        color_limits = (1.0, 3.0),
        n_calc = 10_000
    )
    hjsw_hist_path = joinpath(hjsw_dir, "grid_hjsw_physical_pr_hist.pdf")
    save(hjsw_hist_path, fig_hjsw_hist)
    println("     Saved: ", hjsw_hist_path)

    println("\n>>> SUCCESS! All 7 publication figures regenerated from scratch! <<<")
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_all_figures()
end
