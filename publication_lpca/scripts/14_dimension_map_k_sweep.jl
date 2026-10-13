"""
    Script 14: Mean Dimension Map across (tau, K) with Fitted Collapse Line at d ≈ 1
    Computes a 2D grid:
    - K in 4:1:100 (97 neighborhood sizes)
    - tau in 0.2:0.04:3.0 (71 proper time slices)
    - Extracts the collapse threshold tau*(K) where <d> <= 1.02
    - Fits an analytical curve tau*(K) = a * K^b
    - Produces publication-ready figures (both dual-panel and standalone single-panel).
"""

using AttractorsQGP
using CairoMakie
using LaTeXStrings
using Printf
using Random
using Statistics

function run_dimension_map(;
    mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
    output_dir::AbstractString = joinpath(@__DIR__, "..", "plots", "dimension_map")
)
    mkpath(output_dir)
    CairoMakie.activate!()
    set_publication_theme()

    println("Loading MIS dataset from: ", mis_dataset_path)
    ds = load_hydro_dataset(mis_dataset_path)

    all_taus = sort(unique(ds[:, 1]))
    filter_taus = filter(t -> 0.2 <= t <= 3.0, all_taus)
    # Select every 4th tau slice ~ 71 slices
    tau_grid = filter_taus[1:4:end]
    k_range = 4:1:100
    k_vec = collect(k_range)
    n_k = length(k_vec)

    println("Computing K-sweep across $(length(tau_grid)) proper time slices for K in $(first(k_range))..$(last(k_range))...")
    sweep_res = sweep_k_dimensions_over_tau(
        ds, k_range, tau_grid;
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

    # Extract collapse threshold tau*(K) where mean_dims <= 1.02
    tau_crit_dims = Float64[]
    for (ki, k) in enumerate(k_vec)
        col = mat_dims[:, ki]
        idx = findfirst(v -> v <= 1.02, col)
        push!(tau_crit_dims, isnothing(idx) ? tau_vals[argmin(col)] : tau_vals[idx])
    end

    # Fit Power Law: tau*(K) = a * K^b
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

    println("Power law fit: tau*(K) = $(round(a_pow, digits=3)) * K^($(round(b_pow, digits=3))), R2 = $(round(r2_pow, digits=4))")

    # -------------------------------------------------------------
    # Figure 1: Standalone Single-Panel (Discrete Cutoff Dimension with Fit)
    # -------------------------------------------------------------
    fig_single = Figure(size = (950, 680), figure_padding = (60, 50, 40, 40))
    ax1 = Axis(
        fig_single[1, 1],
        title = L"\text{Map of Mean Local Dimension } \langle d \rangle(\tau, K)",
        titlesize = 22,
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Neighborhood Size } K",
        xlabelsize = 22,
        ylabelsize = 22,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xautolimitmargin = (0.0, 0.0),
        yautolimitmargin = (0.0, 0.0)
    )

    # Heatmap of mat_dims (transposed to match tau on x-axis, K on y-axis)
    hm1 = heatmap!(ax1, tau_vals, k_vec, mat_dims; colormap = :viridis, colorrange = (1.0, 2.0))

    # Add contour isolines
    contour!(ax1, tau_vals, k_vec, mat_dims; levels = [1.1, 1.3, 1.5, 1.7, 1.9], color = (:white, 0.45), linewidth = 1.2)

    # Plot empirical collapse points tau*(K)
    scatter!(ax1, tau_crit_dims, k_vec; color = :cyan, markersize = 6, strokewidth = 0.5, strokecolor = :black, label = L"\text{Empirical collapse } (\langle d \rangle \le 1.02)")

    # Plot fitted line
    fit_label = @sprintf("\\tau^*(K) = %.2f \\cdot K^{%.2f}\\; (R^2 = %.3f)", a_pow, b_pow, r2_pow)
    lines!(ax1, fit_tau_pow, k_vec; color = :crimson, linewidth = 3.5, linestyle = :solid, label = LaTeXString(fit_label))

    axislegend(ax1, position = :rt, labelsize = 16, framecolor = (:white, 0.9), bgcolor = (:white, 0.85))
    Colorbar(fig_single[1, 2], hm1; label = L"\text{Mean Local Dimension } \langle d \rangle", labelsize = 20, ticklabelsize = 15)

    single_pdf = joinpath(output_dir, "map_mean_dimension_tau_vs_k_single.pdf")
    single_png = joinpath(output_dir, "map_mean_dimension_tau_vs_k_single.png")
    save(single_pdf, fig_single)
    save(single_png, fig_single; px_per_unit = 2)

    # -------------------------------------------------------------
    # Figure 2: Dual-Panel Comparison (Hard Dims vs Participation Ratio)
    # -------------------------------------------------------------
    fig_dual = Figure(size = (1600, 680), figure_padding = (65, 55, 45, 45))
    colgap!(fig_dual.layout, 45)

    # Panel A: Hard Cutoff dims()
    ax_a = Axis(
        fig_dual[1, 1],
        title = L"\text{(a) Cutoff Dimension } \langle d \rangle",
        titlesize = 21,
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"K",
        xlabelsize = 22,
        ylabelsize = 22,
        xticklabelsize = 16,
        yticklabelsize = 16
    )
    hm_a = heatmap!(ax_a, tau_vals, k_vec, mat_dims; colormap = :viridis, colorrange = (1.0, 2.0))
    contour!(ax_a, tau_vals, k_vec, mat_dims; levels = [1.1, 1.3, 1.5, 1.7, 1.9], color = (:white, 0.45), linewidth = 1.2)
    scatter!(ax_a, tau_crit_dims, k_vec; color = :cyan, markersize = 6, strokewidth = 0.5, strokecolor = :black, label = L"\text{Data } (\langle d \rangle \le 1.02)")
    lines!(ax_a, fit_tau_pow, k_vec; color = :crimson, linewidth = 3.2, label = LaTeXString(fit_label))
    axislegend(ax_a, position = :rt, labelsize = 14, bgcolor = (:white, 0.85))
    Colorbar(fig_dual[1, 2], hm_a; label = L"\langle d \rangle", labelsize = 18, ticklabelsize = 14)

    # Panel B: Continuous Participation Ratio
    ax_b = Axis(
        fig_dual[1, 3],
        title = L"\text{(b) Participation Ratio } \langle d_{\mathrm{PR}} \rangle",
        titlesize = 21,
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"K",
        xlabelsize = 22,
        ylabelsize = 22,
        xticklabelsize = 16,
        yticklabelsize = 16
    )
    hm_b = heatmap!(ax_b, tau_vals, k_vec, mat_pr; colormap = :plasma, colorrange = (1.0, 2.0))
    contour!(ax_b, tau_vals, k_vec, mat_pr; levels = [1.05, 1.15, 1.3, 1.5, 1.7, 1.9], color = (:white, 0.5), linewidth = 1.2)
    # Also plot the same collapse curve on PR map for comparison
    lines!(ax_b, fit_tau_pow, k_vec; color = :cyan, linewidth = 3.0, linestyle = :dash, label = L"\text{Fitted 1D Boundary}")
    axislegend(ax_b, position = :rt, labelsize = 14, bgcolor = (:white, 0.85))
    Colorbar(fig_dual[1, 4], hm_b; label = L"\langle d_{\mathrm{PR}} \rangle", labelsize = 18, ticklabelsize = 14)

    dual_pdf = joinpath(output_dir, "map_mean_dimension_tau_vs_k_dual.pdf")
    dual_png = joinpath(output_dir, "map_mean_dimension_tau_vs_k_dual.png")
    save(dual_pdf, fig_dual)
    save(dual_png, fig_dual; px_per_unit = 2)

    println("Saved all map figures to: ", output_dir)
    return fig_dual
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_dimension_map()
end
