@views function plot_local_pca(
        dataset_loaded::AbstractMatrix{<:Real};
        n_slices::Integer = 15,
        tablica_k::AbstractVector{<:Integer} = [10, 20, 40, 80, 160],
        feature_cols::AbstractVector{<:Integer} = 2:size(dataset_loaded, 2),
        normalize::Union{Symbol, Function} = :max,
        title::Union{String, Nothing} = nothing
    )
    set_publication_theme()
    fig = Figure(size = (950, 600))
    ax = Axis(fig[1, 1], title = title !== nothing ? title : "", xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Średni wymiar lokalny}", xautolimitmargin = (0.0, 0.05), yautolimitmargin = (0.05, 0.05))
    palette = [:crimson, :dodgerblue, :forestgreen, :darkorange, :purple, :goldenrod, :darkcyan, :mediumvioletred]
    for (i, k_bazowe) in enumerate(tablica_k)
        k_drugie = k_bazowe .* 2
        tau_vals, mean_k1, _ = compute_lpca(dataset_loaded, k_bazowe, n_slices; feature_cols = feature_cols, normalize = normalize)
        _, mean_k2, _ = compute_lpca(dataset_loaded, k_drugie, n_slices; feature_cols = feature_cols, normalize = normalize)
        c = palette[mod1(i, length(palette))]
        band!(ax, tau_vals, mean_k1, mean_k2; color = (c, 0.25))
        lines!(ax, tau_vals, mean_k1; linewidth = 2.5, color = c, label = L"K = %$(k_bazowe), %$(k_drugie)")
    end
    axislegend(ax, position = :rt)
    return fig
end

@views function plot_local_pca_regularizations(
        dataset_loaded::AbstractMatrix{<:Real};
        n_slices::Integer = 15,
        tablica_k::AbstractVector{<:Integer} = [10, 20, 40, 80],
        feature_cols::AbstractVector{<:Integer} = 2:size(dataset_loaded, 2),
        methods::AbstractVector{Symbol} = [:max, :minmax, :zscore, :none]
    )
    set_publication_theme()
    method_titles = Dict(
        :max => "Abs-Max Scaling ([-1, 1])",
        :minmax => "Min-Max Scaling ([0, 1])",
        :zscore => "Z-score Standardization (μ=0, σ=1)",
        :standard => "Z-score Standardization (μ=0, σ=1)",
        :none => "Brak regularyzacji (Dane surowe)",
        :raw => "Brak regularyzacji (Dane surowe)"
    )
    n_methods = length(methods)
    n_cols = n_methods <= 2 ? n_methods : 2
    n_rows = ceil(Int, n_methods / n_cols)
    fig = Figure(size = (500 * n_cols, 400 * n_rows))
    palette = [:crimson, :dodgerblue, :forestgreen, :darkorange, :purple, :goldenrod]

    for (m_idx, m) in enumerate(methods)
        row = div(m_idx - 1, n_cols) + 1
        col = mod1(m_idx, n_cols)
        ax = Axis(fig[row, col], title = get(method_titles, m, string(m)), xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Średni wymiar lokalny}", xautolimitmargin = (0.0, 0.05), yautolimitmargin = (0.05, 0.05))
        for (i, k_bazowe) in enumerate(tablica_k)
            k_drugie = k_bazowe * 2
            tau_vals, mean_k1, _ = compute_lpca(dataset_loaded, k_bazowe, n_slices; feature_cols = feature_cols, normalize = m)
            _, mean_k2, _ = compute_lpca(dataset_loaded, k_drugie, n_slices; feature_cols = feature_cols, normalize = m)
            c = palette[mod1(i, length(palette))]
            band!(ax, tau_vals, mean_k1, mean_k2; color = (c, 0.25))
            lines!(ax, tau_vals, mean_k1; linewidth = 2.0, color = c, label = L"K = %$(k_bazowe), %$(k_drugie)")
        end
        row == 1 && col == n_cols && axislegend(ax, position = :rt)
    end
    return fig
end

function plot_map_lpca(
        dataset::AbstractMatrix{<:Real};
        zakres_K::AbstractVector{<:Integer} = [10, 20, 40, 80, 160],
        n_slices::Int = 15,
        feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize::Union{Symbol, Function} = :standard,
        method::Symbol = :dims,
        title::Union{String, LaTeXString} = "",
        colormap = nothing,
        colorrange = nothing,
        ticks = nothing,
        show_optimal_k::Bool = false,
        figure_size::Tuple{Integer, Integer} = (1200, 600)
    )
    set_publication_theme()
    palette = isnothing(colormap) ? resolve_palette(lpca_style) : resolve_palette(colormap)

    taus = sort(unique(dataset[:, 1]))
    idxs = round.(Int, range(1, length(taus), length = n_slices))
    tau_vals = taus[idxs]

    D = length(feature_cols)
    _, first_slice = get_tau_slice(dataset, tau_vals[1]; feature_cols = feature_cols)
    N_sample = size(first_slice, 1)

    dim_matrix = zeros(Float64, n_slices, length(zakres_K))

    if method === :pr
        for (k_idx, k) in enumerate(zakres_K)
            for (t_idx, tau) in enumerate(tau_vals)
                _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_cols)
                norm_slice = apply_normalization(raw_slice, normalize isa Symbol ? normalize : :standard)
                res = compute_local_pr_dimension(norm_slice; k = k, use_density_weights = false)
                dim_matrix[t_idx, k_idx] = res.mean
            end
        end
        c_range = isnothing(colorrange) ? (1.0, Float64(D)) : colorrange
        c_label = L"\text{Participation Ratio } \langle d_{\mathrm{PR}} \rangle"
        c_ticks = isnothing(ticks) ? range(c_range[1], c_range[2], length = 3) : ticks
    else
        results = [compute_lpca(dataset, k, n_slices; feature_cols = feature_cols, normalize = normalize) for k in zakres_K]
        tau_vals = results[1][1]
        dim_matrix = reduce(hcat, [r[2] for r in results])
        c_range = isnothing(colorrange) ? (1, D) : colorrange
        c_label = L"\text{Local Dimension } \langle d \rangle"
        c_ticks = isnothing(ticks) ? collect(round(Int, c_range[1]):round(Int, c_range[2])) : ticks
    end

    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], title = title, xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"K")
    hm = heatmap!(ax, tau_vals, zakres_K, dim_matrix; colormap = palette, colorrange = c_range)

    if show_optimal_k
        k_opt = suggest_optimal_k(N_sample, D)
        hlines!(ax, [k_opt], color = :white, linestyle = :dash, linewidth = 2.5, label = L"K_{\mathrm{opt}} = %$(k_opt)")
        axislegend(ax, position = :rt)
    end

    Colorbar(fig[1, 2], hm; label = c_label, ticks = c_ticks)
    return fig
end

function plot_k_dependency_bands(
        k_results::AbstractVector{<:NamedTuple};
        ylabel::Union{LaTeXString, String} = L"\text{Mean Local Dimension } \langle d \rangle",
        figure_size::Tuple{Integer, Integer} = (900, 560),
        tau_max::Union{Nothing, Real} = 6.0
    )
    set_publication_theme()
    figure = Figure(size = figure_size)
    axis = Axis(figure[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = ylabel, xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    !isnothing(tau_max) && xlims!(axis, 0.0, Float64(tau_max))
    palette = [:crimson, :dodgerblue, :forestgreen, :darkorange, :purple, :goldenrod, :darkcyan]
    for (index, item) in enumerate(k_results)
        color = palette[mod1(index, length(palette))]
        band!(axis, item.tau_values, item.mean_dimension_k1, item.mean_dimension_k2; color = (color, 0.25))
        lines!(axis, item.tau_values, item.mean_dimension_k1; linewidth = 2.5, color = color, label = L"K = %$(item.k_base), %$(item.k_expanded)")
    end
    axislegend(axis, position = :rt)
    return figure
end

function plot_k_dependency_bands(
        dataset::AbstractArray{<:Real};
        k_pairs = [(5, 10), (10, 20), (20, 40), (40, 80)],
        tau_values = range(0.2, 5.0, length = 15),
        method::Symbol = :dims,
        kwargs...
    )
    if method === :pr
        k_res = evaluate_k_dependency_pr(dataset, k_pairs, tau_values; use_density_weights = false)
        return plot_k_dependency_bands(k_res; ylabel = L"\text{Participation Ratio } \langle d_{\mathrm{PR}} \rangle", kwargs...)
    else
        return plot_k_dependency_bands(evaluate_k_dependency(dataset, k_pairs, tau_values); kwargs...)
    end
end

function plot_k_sweep_curve(
        sweep_data::NamedTuple;
        method::Symbol = :dims,
        tau_subset::Union{Nothing, AbstractVector{<:Real}} = nothing,
        palette = [:dodgerblue, :darkcyan, :forestgreen, :goldenrod, :darkorange, :crimson, :purple, :darkmagenta],
        figure_size::Tuple{Integer, Integer} = (920, 580),
        y_limits::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
        title::Union{Nothing, LaTeXString, String} = nothing
    )
    set_publication_theme()
    k_values = sweep_data.k_values
    taus_to_plot = isnothing(tau_subset) ? sweep_data.tau_values : tau_subset
    y_label = method === :dims ? L"\text{Local Dimension } \langle d \rangle" : L"\text{Participation Ratio Dimension } \langle d_{\mathrm{PR}} \rangle"
    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], xlabel = L"\text{Neighborhood Size } K", ylabel = y_label, title = isnothing(title) ? "" : title, titlesize = 18, xautolimitmargin = (0.02, 0.04), yautolimitmargin = (0.05, 0.05))
    !isnothing(y_limits) && ylims!(ax, Float64(y_limits[1]), Float64(y_limits[2]))
    xlims!(ax, Float64(minimum(k_values)), Float64(maximum(k_values)))

    for (t_idx, tau) in enumerate(taus_to_plot)
        tau_key = Float64(tau)
        found_key = findfirst(k -> isapprox(k, tau_key; atol = 1.0e-4), collect(keys(sweep_data.results)))
        isnothing(found_key) && continue
        res = sweep_data.results[collect(keys(sweep_data.results))[found_key]]
        color = palette[mod1(t_idx, length(palette))]
        mean_vec = method === :dims ? res.mean_dims : res.mean_pr
        std_vec = method === :dims ? res.std_dims : res.std_pr
        band!(ax, k_values, mean_vec .- std_vec, mean_vec .+ std_vec; color = (color, 0.2))
        lines!(ax, k_values, mean_vec; color = color, linewidth = 2.5, label = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c")
    end
    axislegend(ax, position = :rt, nbanks = 2, labelsize = 13)
    return fig
end

function plot_d_vs_tau_multi_k(
        sweep_data::NamedTuple;
        method::Symbol = :dims,
        k_subset::AbstractVector{<:Integer} = [3, 5, 8, 12, 16, 24, 35, 50],
        tau_max::Union{Nothing, Real} = 6.0,
        palette = [:navy, :dodgerblue, :darkcyan, :forestgreen, :goldenrod, :darkorange, :crimson, :darkmagenta],
        figure_size::Tuple{Integer, Integer} = (920, 580),
        show_std::Bool = true,
        linewidth::Real = 2.5,
        y_limits::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
        title::Union{Nothing, LaTeXString, String} = nothing
    )
    set_publication_theme()
    tau_values = sort(collect(sweep_data.tau_values))
    !isnothing(tau_max) && (tau_values = filter(t -> t <= tau_max + 1.0e-4, tau_values))
    n_tau = length(tau_values)
    all_k = sweep_data.k_values
    available_k = filter(k -> k in all_k, k_subset)
    y_label = method === :dims ? L"\text{Local Dimension } \langle d \rangle" : L"\text{Participation Ratio Dimension } \langle d_{\mathrm{PR}} \rangle"

    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = y_label, title = isnothing(title) ? "" : title, titlesize = 18, xautolimitmargin = (0.01, 0.04), yautolimitmargin = (0.05, 0.05))
    !isnothing(tau_max) && xlims!(ax, 0.0, Float64(tau_max))
    !isnothing(y_limits) && ylims!(ax, Float64(y_limits[1]), Float64(y_limits[2]))

    for (idx_k, k_val) in enumerate(available_k)
        color = palette[mod1(idx_k, length(palette))]
        k_pos = findfirst(==(k_val), all_k)
        mean_traj = zeros(Float64, n_tau)
        std_traj = zeros(Float64, n_tau)
        for (t_idx, tau) in enumerate(tau_values)
            found_key = findfirst(k -> isapprox(k, Float64(tau); atol = 1.0e-4), collect(keys(sweep_data.results)))
            if !isnothing(found_key)
                res = sweep_data.results[collect(keys(sweep_data.results))[found_key]]
                mean_traj[t_idx] = method === :dims ? res.mean_dims[k_pos] : res.mean_pr[k_pos]
                std_traj[t_idx] = method === :dims ? res.std_dims[k_pos] : res.std_pr[k_pos]
            end
        end
        show_std && band!(ax, tau_values, mean_traj .- std_traj, mean_traj .+ std_traj; color = (color, 0.18))
        lines!(ax, tau_values, mean_traj; color = color, linewidth = linewidth, label = L"K = %$(k_val)")
    end
    axislegend(ax, position = :rt, nbanks = 2, labelsize = 13)
    return fig
end

function plot_d_vs_tau_k_averaged(
        sweep_data::NamedTuple;
        method::Symbol = :dims,
        tau_max::Union{Nothing, Real} = 6.0,
        line_color = :dodgerblue,
        figure_size::Tuple{Integer, Integer} = (920, 580),
        show_std::Bool = true,
        show_envelope::Bool = true,
        linewidth::Real = 2.8,
        y_limits::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
        title::Union{Nothing, LaTeXString, String} = nothing
    )
    set_publication_theme()
    tau_values = sort(collect(sweep_data.tau_values))
    !isnothing(tau_max) && (tau_values = filter(t -> t <= tau_max + 1.0e-4, tau_values))
    n_tau = length(tau_values)
    min_k, max_k = extrema(sweep_data.k_values)
    mean_k_traj = zeros(Float64, n_tau)
    std_k_traj = zeros(Float64, n_tau)
    min_k_traj = zeros(Float64, n_tau)
    max_k_traj = zeros(Float64, n_tau)

    for (t_idx, tau) in enumerate(tau_values)
        found_key = findfirst(k -> isapprox(k, Float64(tau); atol = 1.0e-4), collect(keys(sweep_data.results)))
        if !isnothing(found_key)
            vals = method === :dims ? sweep_data.results[collect(keys(sweep_data.results))[found_key]].mean_dims : sweep_data.results[collect(keys(sweep_data.results))[found_key]].mean_pr
            mean_k_traj[t_idx] = mean(vals)
            std_k_traj[t_idx] = std(vals)
            min_k_traj[t_idx] = minimum(vals)
            max_k_traj[t_idx] = maximum(vals)
        end
    end
    y_label = method === :dims ? L"\text{Averaged Dimension } \langle d \rangle_{K}" : L"\text{Averaged PR Dimension } \langle d_{\mathrm{PR}} \rangle_{K}"
    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = y_label, title = isnothing(title) ? "" : title, titlesize = 18, xautolimitmargin = (0.01, 0.04), yautolimitmargin = (0.05, 0.05))
    !isnothing(tau_max) && xlims!(ax, 0.0, Float64(tau_max))
    !isnothing(y_limits) && ylims!(ax, Float64(y_limits[1]), Float64(y_limits[2]))

    show_envelope && band!(ax, tau_values, min_k_traj, max_k_traj; color = (line_color, 0.12), label = L"\text{Envelope } [\min_K, \max_K]")
    show_std && band!(ax, tau_values, mean_k_traj .- std_k_traj, mean_k_traj .+ std_k_traj; color = (line_color, 0.28), label = L"\pm 1\sigma_K \text{ (uncertainty over } K \in [%$(min_k), %$(max_k)])")
    lines!(ax, tau_values, mean_k_traj; color = line_color, linewidth = linewidth, label = L"\text{Mean over } K \in [%$(min_k), %$(max_k)]")
    (show_envelope || show_std) && axislegend(ax, position = :rt, labelsize = 13)
    return fig
end

function plot_normalization_multipanel(
        method_results::Dict{Symbol, Vector{NamedTuple}};
        methods_to_plot::AbstractVector{Symbol} = [:none, :max, :minmax, :zscore],
        ylabel::Union{LaTeXString, String} = L"\text{Mean Local Dimension } \langle d \rangle",
        tau_max::Union{Nothing, Real} = 6.0
    )
    set_publication_theme()
    palette = [:crimson, :dodgerblue, :forestgreen, :darkorange, :purple, :goldenrod]
    panel_count = length(methods_to_plot)
    column_count = panel_count <= 2 ? panel_count : 2
    row_count = ceil(Int, panel_count / column_count)
    figure = Figure(size = (520 * column_count, 420 * row_count))

    for (panel_index, method) in enumerate(methods_to_plot)
        row = div(panel_index - 1, column_count) + 1
        col = mod1(panel_index, column_count)
        axis = Axis(figure[row, col], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = ylabel, xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
        !isnothing(tau_max) && xlims!(axis, 0.0, Float64(tau_max))
        results_for_method = method_results[method]
        for (k_index, item) in enumerate(results_for_method)
            color = palette[mod1(k_index, length(palette))]
            band!(axis, item.tau_values, item.mean_dimension_k1, item.mean_dimension_k2; color = (color, 0.25))
            lines!(axis, item.tau_values, item.mean_dimension_k1; linewidth = 2.0, color = color, label = L"K = %$(item.k_base), %$(item.k_expanded)")
        end
        row == 1 && col == column_count && axislegend(axis, position = :rt)
    end
    return figure
end

function plot_normalization_multipanel(dataset::AbstractArray{<:Real}; methods::AbstractVector{Symbol} = [:none, :max, :minmax, :zscore], k_pairs = [(5, 10), (10, 20), (20, 40), (40, 80)], tau_values = range(0.2, 5.0, length = 15), kwargs...)
    return plot_normalization_multipanel(compare_normalization_methods(dataset, methods, k_pairs, tau_values); methods_to_plot = methods, kwargs...)
end

function plot_normalization_direct_overlay(
        dataset::AbstractMatrix{<:Real},
        normalization_methods::AbstractVector{Symbol},
        k_neighbor::Integer,
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        tolerance::Real = 0.01,
        tau_max::Union{Nothing, Real} = 6.0
    )
    set_publication_theme()
    figure = Figure(size = (900, 560))
    axis = Axis(figure[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Mean Local Dimension } \langle d \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    !isnothing(tau_max) && xlims!(axis, 0.0, Float64(tau_max))
    labels = Dict(:none => L"\text{Raw Units}", :max => L"\text{Abs-Max Scaling } ([-1, 1])", :minmax => L"\text{Min-Max Scaling } ([0, 1])", :zscore => L"\text{Z-Score Standardization}")
    palette = [:crimson, :dodgerblue, :forestgreen, :darkorange]

    for (index, method) in enumerate(normalization_methods)
        means = [mean(dims(apply_normalization(get_tau_slice(dataset, tau; feature_cols = feature_indices)[2], method); k = k_neighbor, tol = tolerance)) for tau in tau_values]
        color = palette[mod1(index, length(palette))]
        lines!(axis, tau_values, means; linewidth = 3.0, color = color, label = get(labels, method, string(method)))
        scatter!(axis, tau_values, means; markersize = 8, color = color)
    end
    axislegend(axis, position = :rt)
    return figure
end

function plot_pr_normalization_direct_overlay(
        dataset::AbstractMatrix{<:Real},
        normalization_methods::AbstractVector{Symbol},
        k_neighbor::Integer,
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        tau_max::Union{Nothing, Real} = 6.0,
        figure_size::Tuple{Integer, Integer} = (900, 560)
    )
    set_publication_theme()
    figure = Figure(size = figure_size)
    axis = Axis(figure[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Mean Local PR Dimension } \langle d_{\mathrm{PR}} \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    !isnothing(tau_max) && xlims!(axis, 0.0, Float64(tau_max))
    labels = Dict(:none => L"\text{Raw Units}", :max => L"\text{Abs-Max Scaling } ([-1, 1])", :minmax => L"\text{Min-Max Scaling } ([0, 1])", :zscore => L"\text{Z-Score Standardization}")
    palette = [:crimson, :dodgerblue, :forestgreen, :darkorange]

    for (index, method) in enumerate(normalization_methods)
        means = [compute_local_pr_dimension(apply_normalization(get_tau_slice(dataset, tau; feature_cols = feature_indices)[2], method); k = k_neighbor).mean for tau in tau_values]
        color = palette[mod1(index, length(palette))]
        lines!(axis, tau_values, means; linewidth = 3.0, color = color, label = get(labels, method, string(method)))
        scatter!(axis, tau_values, means; markersize = 8, color = color)
    end
    axislegend(axis, position = :rt)
    return figure
end

function plot_coordinate_invariance(invariance_result_normalized::NamedTuple, invariance_result_raw::NamedTuple; figure_size::Tuple{Integer, Integer} = (1100, 520))
    set_publication_theme()
    figure = Figure(size = figure_size)
    tau_values = invariance_result_normalized.tau_values
    axis_left = Axis(figure[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Mean Local Dimension } \langle d \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    axis_right = Axis(figure[1, 2], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Mean Local Dimension } \langle d \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    is_hjsw = invariance_result_normalized.model == :hjsw
    variants = is_hjsw ? [
            (:physical, L"\text{Physical: }(T, \mathcal{A}, \mathcal{B})", :dodgerblue, :solid, 3.0),
            (:dimensionless, L"\text{Dimensionless: }(w, \mathcal{A}, \mathcal{B})", :crimson, :dash, 2.5),
            (:scaled_10x, L"\text{Rescaled: }(10T, \mathcal{A}, \mathcal{B})", :forestgreen, :dot, 2.5),
        ] : [
            (:physical, L"\text{Physical: }(T, \mathcal{A})", :dodgerblue, :solid, 3.0),
            (:dimensionless, L"\text{Dimensionless: }(w, \mathcal{A})", :crimson, :dash, 2.5),
            (:scaled_10x, L"\text{Rescaled: }(10T, \mathcal{A})", :forestgreen, :dot, 2.5),
        ]
    for (key, label_text, color, line_style, width) in variants
        lines!(axis_left, tau_values, invariance_result_normalized.curves[key]; color = color, linestyle = line_style, linewidth = width, label = label_text)
        lines!(axis_right, tau_values, invariance_result_raw.curves[key]; color = color, linestyle = line_style, linewidth = width, label = label_text)
    end
    axislegend(axis_left, position = :rt)
    axislegend(axis_right, position = :rt)
    return figure
end

function plot_coordinate_invariance(variant_datasets::NamedTuple, k_neighbor::Integer, tau_values::AbstractVector{<:Real}; kwargs...)
    res_norm = test_coordinate_invariance(variant_datasets, k_neighbor, tau_values; normalize_method = :max)
    res_raw = test_coordinate_invariance(variant_datasets, k_neighbor, tau_values; normalize_method = :none)
    return plot_coordinate_invariance(res_norm, res_raw; kwargs...)
end

function plot_coordinate_invariance_single(invariance_result::NamedTuple; figure_size::Tuple{Integer, Integer} = (900, 560))
    set_publication_theme()
    figure = Figure(size = figure_size)
    tau_values = invariance_result.tau_values
    axis = Axis(figure[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Mean Local Dimension } \langle d \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    is_hjsw = invariance_result.model == :hjsw
    variants = is_hjsw ? [
            (:physical, L"\text{Physical: }(T, \mathcal{A}, \mathcal{B})", :dodgerblue, :solid, 3.0),
            (:dimensionless, L"\text{Dimensionless: }(w, \mathcal{A}, \mathcal{B})", :crimson, :dash, 2.5),
            (:scaled_10x, L"\text{Rescaled: }(10T, \mathcal{A}, \mathcal{B})", :forestgreen, :dot, 2.5),
        ] : [
            (:physical, L"\text{Physical: }(T, \mathcal{A})", :dodgerblue, :solid, 3.0),
            (:dimensionless, L"\text{Dimensionless: }(w, \mathcal{A})", :crimson, :dash, 2.5),
            (:scaled_10x, L"\text{Rescaled: }(10T, \mathcal{A})", :forestgreen, :dot, 2.5),
        ]
    for (key, label_text, color, line_style, width) in variants
        lines!(axis, tau_values, invariance_result.curves[key]; color = color, linestyle = line_style, linewidth = width, label = label_text)
    end
    axislegend(axis, position = :rt)
    return figure
end

function plot_dimension_distribution(distribution_result::NamedTuple; figure_size::Tuple{Integer, Integer} = (1100, 500))
    set_publication_theme()
    figure = Figure(size = figure_size)
    tau_values = distribution_result.tau_values
    embedding_dim = distribution_result.embedding_dimension
    axis_left = Axis(figure[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Local Dimension } d", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    axis_right = Axis(figure[1, 2], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Fraction of Points } P(d)", limits = (nothing, nothing, 0.0, 1.05), xautolimitmargin = (0.0, 0.04))

    band!(axis_left, tau_values, distribution_result.q25_dimension, distribution_result.q75_dimension; color = (:dodgerblue, 0.25), label = L"\text{IQR (25\% -- 75\%)}")
    lines!(axis_left, tau_values, distribution_result.mean_dimension; color = :crimson, linewidth = 3.0, label = L"\text{Mean } \langle d \rangle")
    lines!(axis_left, tau_values, distribution_result.median_dimension; color = :navy, linewidth = 2.5, linestyle = :dash, label = L"\text{Median } d_{1/2}")
    axislegend(axis_left, position = :rt)

    dim_colors = [:forestgreen, :darkorange, :purple, :dodgerblue]
    for dim_val in 1:embedding_dim
        color = dim_colors[mod1(dim_val, length(dim_colors))]
        lines!(axis_right, tau_values, distribution_result.dimension_fractions[:, dim_val]; color = color, linewidth = 2.8, label = L"P(d = %$dim_val)")
        scatter!(axis_right, tau_values, distribution_result.dimension_fractions[:, dim_val]; color = color, markersize = 7)
    end
    axislegend(axis_right, position = :rt)
    return figure
end

plot_dimension_distribution(dataset::AbstractArray{<:Real}, k_neighbor::Integer, tau_values::AbstractVector{<:Real}; kwargs...) = plot_dimension_distribution(analyze_dimension_distribution(dataset, k_neighbor, tau_values; kwargs...))

function plot_dimension_mean_single(distribution_result::NamedTuple; figure_size::Tuple{Integer, Integer} = (900, 560))
    set_publication_theme()
    figure = Figure(size = figure_size)
    axis = Axis(figure[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Local Dimension } d", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    band!(axis, distribution_result.tau_values, distribution_result.q25_dimension, distribution_result.q75_dimension; color = (:dodgerblue, 0.25), label = L"\text{IQR (25\% -- 75\%)}")
    lines!(axis, distribution_result.tau_values, distribution_result.mean_dimension; color = :crimson, linewidth = 3.0, label = L"\text{Mean } \langle d \rangle")
    lines!(axis, distribution_result.tau_values, distribution_result.median_dimension; color = :navy, linewidth = 2.5, linestyle = :dash, label = L"\text{Median } d_{1/2}")
    axislegend(axis, position = :rt)
    return figure
end

function plot_dimension_populations_single(distribution_result::NamedTuple; figure_size::Tuple{Integer, Integer} = (900, 560))
    set_publication_theme()
    figure = Figure(size = figure_size)
    axis = Axis(figure[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Fraction of Points } P(d)", limits = (nothing, nothing, 0.0, 1.05), xautolimitmargin = (0.0, 0.04))
    dim_colors = [:forestgreen, :darkorange, :purple, :dodgerblue]
    for dim_val in 1:distribution_result.embedding_dimension
        color = dim_colors[mod1(dim_val, length(dim_colors))]
        lines!(axis, distribution_result.tau_values, distribution_result.dimension_fractions[:, dim_val]; color = color, linewidth = 2.8, label = L"P(d = %$dim_val)")
        scatter!(axis, distribution_result.tau_values, distribution_result.dimension_fractions[:, dim_val]; color = color, markersize = 7)
    end
    axislegend(axis, position = :rt)
    return figure
end

function plot_tolerance_sensitivity(sensitivity_result::NamedTuple; palette = [:dodgerblue, :forestgreen, :darkorange, :crimson], figure_size::Tuple{Integer, Integer} = (900, 560))
    set_publication_theme()
    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Mean Local Dimension } \langle d \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    tau_grid = sensitivity_result.tau_values
    for (t_idx, tol_val) in enumerate(sensitivity_result.tolerances)
        lines!(ax, tau_grid, sensitivity_result.results[Float64(tol_val)]; color = palette[mod1(t_idx, length(palette))], linewidth = 2.5, label = L"\mathrm{tol} = %$(tol_val)")
    end
    axislegend(ax, position = :rt)
    return fig
end

function plot_tolerance_sensitivity(mis_result::NamedTuple, hjsw_result::NamedTuple; palette = [:dodgerblue, :forestgreen, :darkorange, :crimson], figure_size::Tuple{Integer, Integer} = (1100, 520))
    set_publication_theme()
    fig = Figure(size = figure_size)
    ax_mis = Axis(fig[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Mean Local Dimension } \langle d \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    ax_hjsw = Axis(fig[1, 2], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Mean Local Dimension } \langle d \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    for (t_idx, tol_val) in enumerate(mis_result.tolerances)
        color = palette[mod1(t_idx, length(palette))]
        lines!(ax_mis, mis_result.tau_values, mis_result.results[Float64(tol_val)]; color = color, linewidth = 2.5, label = L"\mathrm{tol} = %$(tol_val)")
        lines!(ax_hjsw, hjsw_result.tau_values, hjsw_result.results[Float64(tol_val)]; color = color, linewidth = 2.5, label = L"\mathrm{tol} = %$(tol_val)")
    end
    axislegend(ax_mis, position = :rt)
    return fig
end

function plot_parameterization_focus_2x2(sweep_result::NamedTuple; colormap::Symbol = :phase, figure_size::Tuple{Integer, Integer} = (1180, 880), max_tau::Real = 7.0, draw_tunnel::Bool = true)
    set_publication_theme()
    fig = Figure(size = figure_size, figure_padding = (25, 25, 25, 25))
    k_values = sweep_result.k_values
    tau_indices = findall(t -> t <= max_tau + 1.0e-5, sweep_result.tau_values)
    tau_plot = sweep_result.tau_values[tau_indices]
    k_min, k_max = extrema(k_values)
    is_hjsw = sweep_result.model == :hjsw
    panel_defs = is_hjsw ? [
            (:physical, L"\mathbf{(a)}\quad (T, \mathcal{A}, \mathcal{B})", :solid, 1, 1),
            (:dimensionless, L"\mathbf{(b)}\quad (w, \mathcal{A}, \mathcal{B})", :dash, 1, 2),
            (:scaled_10x, L"\mathbf{(c)}\quad (10T, \mathcal{A}, \mathcal{B})", :dot, 2, 1),
            (:mixed_scaled, L"\mathbf{(d)}\quad (10w, 2\mathcal{A}, \mathcal{B})", :dashdot, 2, 2),
        ] : [
            (:physical, L"\mathbf{(a)}\quad (T, \mathcal{A})", :solid, 1, 1),
            (:dimensionless, L"\mathbf{(b)}\quad (w, \mathcal{A})", :dash, 1, 2),
            (:scaled_10x, L"\mathbf{(c)}\quad (10T, \mathcal{A})", :dot, 2, 1),
            (:mixed_scaled, L"\mathbf{(d)}\quad (10w, 2\mathcal{A})", :dashdot, 2, 2),
        ]
    axes_list = Axis[]
    for (focused_key, panel_title, _, row, col) in panel_defs
        ax = Axis(fig[row, col], xlabel = row == 2 ? L"\tau\,[\mathrm{fm}/c]" : "", ylabel = col == 1 ? L"\text{Mean Local Dimension } \langle d \rangle" : "", title = panel_title, titlesize = 18, xautolimitmargin = (0.02, 0.04), yautolimitmargin = (0.05, 0.05))
        push!(axes_list, ax)
        for (bg_key, _, bg_style, _, _) in panel_defs
            if bg_key != focused_key && haskey(sweep_result.curves, bg_key)
                draw_tunnel && band!(ax, tau_plot, sweep_result.envelopes[bg_key].min[tau_indices], sweep_result.envelopes[bg_key].max[tau_indices]; color = (:gray85, 0.3))
                for k in k_values
                    lines!(ax, tau_plot, sweep_result.curves[bg_key][k][tau_indices]; color = (:gray65, 0.35), linewidth = 0.9, linestyle = bg_style)
                end
            end
        end
        draw_tunnel && haskey(sweep_result.envelopes, focused_key) && band!(ax, tau_plot, sweep_result.envelopes[focused_key].min[tau_indices], sweep_result.envelopes[focused_key].max[tau_indices]; color = (:gray80, 0.18))
        if haskey(sweep_result.curves, focused_key)
            for k in k_values
                norm_val = k_max > k_min ? (k - k_min) / (k_max - k_min) : 0.5
                lines!(ax, tau_plot, sweep_result.curves[focused_key][k][tau_indices]; color = cgrad(colormap)[norm_val], linewidth = 1.8, linestyle = :solid)
            end
        end
        xlims!(ax, minimum(tau_plot), max_tau)
        ylims!(ax, 0.95, is_hjsw ? 3.05 : 2.05)
    end
    linkaxes!(axes_list...)
    Colorbar(fig[1:2, 3], limits = (k_min, k_max), colormap = colormap, label = L"\text{Nearest Neighbors } K", width = 18, ticklabelsize = 14, labelsize = 16)
    return fig
end

function plot_parameterization_focus_tripanel(sweep_result::NamedTuple; colormap::Symbol = :phase, figure_size::Tuple{Integer, Integer} = (1550, 460), max_tau::Real = 7.0, draw_tunnel::Bool = true)
    set_publication_theme()
    fig = Figure(size = figure_size, figure_padding = (30, 25, 25, 25))
    k_values = sweep_result.k_values
    tau_indices = findall(t -> t <= max_tau + 1.0e-5, sweep_result.tau_values)
    tau_plot = sweep_result.tau_values[tau_indices]
    k_min, k_max = extrema(k_values)
    is_hjsw = sweep_result.model == :hjsw
    panel_defs = is_hjsw ? [
            (:physical, L"\mathbf{(a)}\quad (T, \mathcal{A}, \mathcal{B})", :solid),
            (:dimensionless, L"\mathbf{(b)}\quad (w, \mathcal{A}, \mathcal{B})", :dash),
            (:scaled_10x, L"\mathbf{(c)}\quad (10T, \mathcal{A}, \mathcal{B})", :dot),
        ] : [
            (:physical, L"\mathbf{(a)}\quad (T, \mathcal{A})", :solid),
            (:dimensionless, L"\mathbf{(b)}\quad (w, \mathcal{A})", :dash),
            (:scaled_10x, L"\mathbf{(c)}\quad (10T, \mathcal{A})", :dot),
        ]
    axes_list = Axis[]
    for (p_idx, (focused_key, panel_title, _)) in enumerate(panel_defs)
        ax = Axis(fig[1, p_idx], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = p_idx == 1 ? L"\text{Mean Local Dimension } \langle d \rangle" : "", title = panel_title, titlesize = 18, xautolimitmargin = (0.02, 0.04), yautolimitmargin = (0.05, 0.05))
        push!(axes_list, ax)
        for (bg_key, _, bg_style) in panel_defs
            if bg_key != focused_key && haskey(sweep_result.curves, bg_key)
                draw_tunnel && band!(ax, tau_plot, sweep_result.envelopes[bg_key].min[tau_indices], sweep_result.envelopes[bg_key].max[tau_indices]; color = (:gray85, 0.35))
                for k in k_values
                    lines!(ax, tau_plot, sweep_result.curves[bg_key][k][tau_indices]; color = (:gray65, 0.4), linewidth = 1.0, linestyle = bg_style)
                end
            end
        end
        draw_tunnel && haskey(sweep_result.envelopes, focused_key) && band!(ax, tau_plot, sweep_result.envelopes[focused_key].min[tau_indices], sweep_result.envelopes[focused_key].max[tau_indices]; color = (:gray80, 0.2))
        if haskey(sweep_result.curves, focused_key)
            for k in k_values
                norm_val = k_max > k_min ? (k - k_min) / (k_max - k_min) : 0.5
                lines!(ax, tau_plot, sweep_result.curves[focused_key][k][tau_indices]; color = cgrad(colormap)[norm_val], linewidth = 2.2, linestyle = :solid)
            end
        end
        xlims!(ax, minimum(tau_plot), max_tau)
    end
    linkaxes!(axes_list...)
    Colorbar(fig[1, 4], limits = (k_min, k_max), colormap = colormap, label = L"\text{Nearest Neighbors } K", width = 18, ticklabelsize = 14, labelsize = 16)
    return fig
end

function plot_soft_weighted_dimension(scan_result::NamedTuple; compare_hard::Bool = true, figure_size::Tuple{Integer, Integer} = (900, 560))
    set_publication_theme()
    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Local Dimension } \langle d \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    band!(ax, scan_result.tau_values, scan_result.mean_dims .- scan_result.std_dims, scan_result.mean_dims .+ scan_result.std_dims; color = (:dodgerblue, 0.25), label = L"\pm 1\sigma_W \text{ (Weighted Band)}")
    lines!(ax, scan_result.tau_values, scan_result.mean_dims; color = :dodgerblue, linewidth = 3.0, label = L"\langle d \rangle_W \text{ (Soft-Weighted)}")
    if compare_hard && hasfield(typeof(scan_result), :unweighted_hard_means)
        lines!(ax, scan_result.tau_values, scan_result.unweighted_hard_means; color = :crimson, linewidth = 2.0, linestyle = :dash, label = L"\langle d \rangle \text{ (Standard Hard LPCA)}")
    end
    axislegend(ax, position = :rt)
    return fig
end

plot_soft_weighted_dimension(dataset::AbstractArray{<:Real}; tau_values = range(0.2, 5.0, length = 20), kwargs...) = plot_soft_weighted_dimension(scan_soft_weighted_dimension(dataset, tau_values; kwargs...))

function plot_soft_k_dependency(dataset::AbstractArray{<:Real}, k_values::AbstractVector{<:Integer}, tau_values::AbstractVector{<:Real}; feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)), palette = [:dodgerblue, :forestgreen, :darkorange, :crimson, :purple], figure_size::Tuple{Integer, Integer} = (900, 560), kwargs...)
    set_publication_theme()
    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Weighted Local Dimension } \langle d \rangle_W", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    for (k_idx, k_val) in enumerate(k_values)
        color = palette[mod1(k_idx, length(palette))]
        scan_res = scan_soft_weighted_dimension(dataset, tau_values; feature_indices = feature_indices, k = k_val, kwargs...)
        band!(ax, tau_values, scan_res.mean_dims .- scan_res.std_dims, scan_res.mean_dims .+ scan_res.std_dims; color = (color, 0.2))
        lines!(ax, tau_values, scan_res.mean_dims; color = color, linewidth = 2.5, label = L"K = %$(k_val)")
    end
    axislegend(ax, position = :rt)
    return fig
end

function plot_local_pr_dimension(pr_scan::NamedTuple; soft_scan::Union{NamedTuple, Nothing} = nothing, palette = (:darkviolet, :teal), figure_size::Tuple{Integer, Integer} = (850, 520))
    set_publication_theme()
    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Participation Ratio Dimension } \langle d_{\mathrm{PR}} \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    band!(ax, pr_scan.tau_values, pr_scan.mean_dims .- pr_scan.std_dims, pr_scan.mean_dims .+ pr_scan.std_dims; color = (palette[1], 0.22), label = L"\pm 1\sigma_{\mathrm{PR}}")
    lines!(ax, pr_scan.tau_values, pr_scan.mean_dims; color = palette[1], linewidth = 3.0, label = L"\langle d_{\mathrm{PR}} \rangle_W")
    if soft_scan !== nothing
        lines!(ax, soft_scan.tau_values, soft_scan.mean_dims; color = palette[2], linewidth = 2.5, linestyle = :dash, label = L"\langle d_{\mathrm{soft}} \rangle_W")
    end
    axislegend(ax, position = :rt)
    return fig
end

function plot_pr_k_dependency(dataset::AbstractArray{<:Real}, k_values::AbstractVector{<:Integer}, tau_values::AbstractVector{<:Real}; feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)), palette = [:dodgerblue, :forestgreen, :darkorange, :crimson, :purple], figure_size::Tuple{Integer, Integer} = (900, 560), tau_max::Union{Nothing, Real} = 6.0, use_density_weights::Bool = false, kwargs...)
    set_publication_theme()
    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Participation Ratio Dimension } \langle d_{\mathrm{PR}} \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    !isnothing(tau_max) && xlims!(ax, 0.0, Float64(tau_max))
    for (k_idx, k_val) in enumerate(k_values)
        color = palette[mod1(k_idx, length(palette))]
        scan_res = scan_local_pr_dimension(dataset, tau_values; feature_indices = feature_indices, k = k_val, use_density_weights = use_density_weights, kwargs...)
        band!(ax, tau_values, scan_res.mean_dims .- scan_res.std_dims, scan_res.mean_dims .+ scan_res.std_dims; color = (color, 0.2))
        lines!(ax, tau_values, scan_res.mean_dims; color = color, linewidth = 2.5, label = L"K = %$(k_val)")
    end
    axislegend(ax, position = :rt)
    return fig
end

function plot_dims_k_dependency(dataset::AbstractArray{<:Real}, k_values::AbstractVector{<:Integer}, tau_values::AbstractVector{<:Real}; feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)), palette = [:dodgerblue, :forestgreen, :darkorange, :crimson, :purple], figure_size::Tuple{Integer, Integer} = (900, 560), tau_max::Union{Nothing, Real} = 6.0, tolerance::Real = 0.01, normalize_method::Symbol = :max, show_std_band::Bool = true)
    set_publication_theme()
    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Mean Local Dimension } \langle d \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    !isnothing(tau_max) && xlims!(ax, 0.0, Float64(tau_max))
    for (k_idx, k_val) in enumerate(k_values)
        color = palette[mod1(k_idx, length(palette))]
        dist_res = analyze_dimension_distribution(dataset, k_val, tau_values; feature_indices = feature_indices, normalize_method = normalize_method, tolerance = tolerance)
        show_std_band && band!(ax, tau_values, dist_res.mean_dimension .- dist_res.std_dimension, dist_res.mean_dimension .+ dist_res.std_dimension; color = (color, 0.2))
        lines!(ax, tau_values, dist_res.mean_dimension; color = color, linewidth = 2.5, label = L"K = %$(k_val)")
    end
    axislegend(ax, position = :rt)
    return fig
end

function plot_unified_lpca_comparison(mis_dataset::AbstractArray{<:Real}, hjsw_dataset::AbstractArray{<:Real}, tau_grid::AbstractVector{<:Real}; k::Integer = 20, tol::Real = 0.02, delta::Real = 0.005, figure_size::Tuple{Integer, Integer} = (1200, 520))
    set_publication_theme()
    mis_soft = scan_soft_weighted_dimension(mis_dataset, tau_grid; feature_indices = [2, 3], k = k, tol = tol, delta = delta)
    mis_pr = scan_local_pr_dimension(mis_dataset, tau_grid; feature_indices = [2, 3], k = k)
    hjsw_soft = scan_soft_weighted_dimension(hjsw_dataset, tau_grid; feature_indices = [2, 3, 4], k = k, tol = tol, delta = delta)
    hjsw_pr = scan_local_pr_dimension(hjsw_dataset, tau_grid; feature_indices = [2, 3, 4], k = k)

    fig = Figure(size = figure_size)
    ax_mis = Axis(fig[1, 1], title = L"\text{Conformal MIS (Initial 2D: } T, \mathcal{A})", titlesize = 19, xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Estimated Dimension } d(\tau)", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    ax_hjsw = Axis(fig[1, 2], title = L"\text{HJSW Model (Initial 3D: } T, \mathcal{A}, \mathcal{B})", titlesize = 19, xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Estimated Dimension } d(\tau)", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))

    lines!(ax_mis, tau_grid, mis_soft.unweighted_hard_means; color = :gray50, linewidth = 2.0, linestyle = :dot, label = L"\text{Hard LPCA}")
    lines!(ax_mis, tau_grid, mis_soft.mean_dims; color = :dodgerblue, linewidth = 2.8, label = L"\text{Soft-Weighted LPCA}")
    band!(ax_mis, tau_grid, mis_pr.mean_dims .- mis_pr.std_dims, mis_pr.mean_dims .+ mis_pr.std_dims; color = (:darkviolet, 0.18))
    lines!(ax_mis, tau_grid, mis_pr.mean_dims; color = :darkviolet, linewidth = 2.8, linestyle = :dash, label = L"\text{Participation Ratio (PR)}")

    lines!(ax_hjsw, tau_grid, hjsw_soft.unweighted_hard_means; color = :gray50, linewidth = 2.0, linestyle = :dot, label = L"\text{Hard LPCA}")
    lines!(ax_hjsw, tau_grid, hjsw_soft.mean_dims; color = :dodgerblue, linewidth = 2.8, label = L"\text{Soft-Weighted LPCA}")
    band!(ax_hjsw, tau_grid, hjsw_pr.mean_dims .- hjsw_pr.std_dims, hjsw_pr.mean_dims .+ hjsw_pr.std_dims; color = (:darkviolet, 0.18))
    lines!(ax_hjsw, tau_grid, hjsw_pr.mean_dims; color = :darkviolet, linewidth = 2.8, linestyle = :dash, label = L"\text{Participation Ratio (PR)}")

    axislegend(ax_mis, position = :rt)
    axislegend(ax_hjsw, position = :rt)
    return fig
end

plot_local_pca(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_local_pca(to_2d_local_plots(dataset), args...; kwargs...)
plot_local_pca_regularizations(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_local_pca_regularizations(to_2d_local_plots(dataset), args...; kwargs...)
plot_map_lpca(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_map_lpca(to_2d_local_plots(dataset), args...; kwargs...)
plot_pr_normalization_direct_overlay(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_pr_normalization_direct_overlay(to_2d_local_plots(dataset), args...; kwargs...)
plot_normalization_direct_overlay(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_normalization_direct_overlay(to_2d_local_plots(dataset), args...; kwargs...)
