const _LPCA_CURVE_PALETTE = [:crimson, :dodgerblue, :forestgreen, :darkorange, :purple, :goldenrod, :darkcyan, :mediumvioletred]

const _NORM_METHOD_LABELS = Dict(
    :none => L"\text{Raw Units}",
    :max => L"\text{Abs-Max Scaling } ([-1, 1])",
    :minmax => L"\text{Min-Max Scaling } ([0, 1])",
    :zscore => L"\text{Z-Score Standardization}"
)

function _plot_lpca_k_bands!(
        ax::Axis,
        dataset::AbstractMatrix{<:Real},
        tablica_k::AbstractVector{<:Integer},
        n_slices::Integer,
        feature_cols::AbstractVector{<:Integer},
        normalize;
        linewidth::Real = 2.5
    )
    for (i, k_bazowe) in enumerate(tablica_k)
        k_drugie = k_bazowe * 2
        tau_vals, mean_k1, _ = compute_lpca(dataset, k_bazowe, n_slices; feature_cols = feature_cols, normalize = normalize)
        _, mean_k2, _ = compute_lpca(dataset, k_drugie, n_slices; feature_cols = feature_cols, normalize = normalize)
        c = _LPCA_CURVE_PALETTE[mod1(i, length(_LPCA_CURVE_PALETTE))]
        band!(ax, tau_vals, mean_k1, mean_k2; color = (c, 0.25))
        lines!(ax, tau_vals, mean_k1; linewidth = linewidth, color = c, label = L"K = %$(k_bazowe), %$(k_drugie)")
    end
    return ax
end

function _plot_normalization_overlay_impl(
        metric_fn::Function,
        dataset::AbstractMatrix{<:Real},
        normalization_methods::AbstractVector{Symbol},
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        tau_max::Union{Nothing, Real} = 6.0,
        figure_size::Tuple{Integer, Integer} = (900, 560)
    )
    set_publication_theme()
    figure = Figure(size = figure_size)
    axis = Axis(figure[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = ylabel, xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    !isnothing(tau_max) && xlims!(axis, 0.0, Float64(tau_max))

    for (index, method) in enumerate(normalization_methods)
        means = [metric_fn(apply_normalization(get_tau_slice(dataset, tau; feature_cols = feature_indices)[2], method)) for tau in tau_values]
        color = _LPCA_CURVE_PALETTE[mod1(index, 4)]
        lines!(axis, tau_values, means; linewidth = 3.0, color = color, label = get(_NORM_METHOD_LABELS, method, string(method)))
        scatter!(axis, tau_values, means; markersize = 8, color = color)
    end
    axislegend(axis, position = :rt)
    return figure
end

@views function plot_local_pca(
        dataset_loaded::AbstractMatrix{<:Real};
        n_slices::Integer = 15,
        tablica_k::AbstractVector{<:Integer} = [10, 20, 40, 80, 160],
        feature_cols::AbstractVector{<:Integer} = 2:size(dataset_loaded, 2),
        normalize::Union{Symbol, Function, AbstractVector{<:Real}} = :max,
        title::Union{String, Nothing} = nothing
    )
    set_publication_theme()
    fig = Figure(size = (950, 600))
    ax = Axis(
        fig[1, 1],
        title = isnothing(title) ? "" : title,
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Średni wymiar lokalny}",
        xautolimitmargin = (0.0, 0.05),
        yautolimitmargin = (0.05, 0.05)
    )
    _plot_lpca_k_bands!(ax, dataset_loaded, tablica_k, n_slices, feature_cols, normalize; linewidth = 2.5)
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
    n_cols = min(2, n_methods)
    n_rows = ceil(Int, n_methods / n_cols)
    fig = Figure(size = (500 * n_cols, 400 * n_rows))

    for (m_idx, m) in enumerate(methods)
        row = div(m_idx - 1, n_cols) + 1
        col = mod1(m_idx, n_cols)
        ax = Axis(
            fig[row, col],
            title = get(method_titles, m, string(m)),
            xlabel = L"\tau\,[\mathrm{fm}/c]",
            ylabel = L"\text{Średni wymiar lokalny}",
            xautolimitmargin = (0.0, 0.05),
            yautolimitmargin = (0.05, 0.05)
        )
        _plot_lpca_k_bands!(ax, dataset_loaded, tablica_k, n_slices, feature_cols, m; linewidth = 2.0)
        row == 1 && col == n_cols && axislegend(ax, position = :rt)
    end
    return fig
end

function plot_map_lpca(
        dataset::AbstractMatrix{<:Real};
        zakres_K::AbstractVector{<:Integer} = [10, 20, 40, 80, 160],
        n_slices::Int = 15,
        feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize::Union{Symbol, Function, AbstractVector{<:Real}} = :standard,
        method::Symbol = :dims,
        title::Union{String, LaTeXString} = "",
        colormap = nothing,
        colorrange = nothing,
        ticks = nothing,
        figure_size::Tuple{Integer, Integer} = (1200, 600)
    )
    set_publication_theme()
    palette = isnothing(colormap) ? resolve_palette(lpca_style) : resolve_palette(colormap)

    taus = sort(unique(dataset[:, 1]))
    idxs = round.(Int, range(1, length(taus), length = n_slices))
    tau_vals = taus[idxs]
    D = length(feature_cols)

    dim_matrix = zeros(Float64, n_slices, length(zakres_K))

    if method === :pr
        for (k_idx, k) in enumerate(zakres_K), (t_idx, tau) in enumerate(tau_vals)
            _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_cols)
            norm_slice = apply_normalization(raw_slice, normalize)
            dim_matrix[t_idx, k_idx] = compute_local_pr_dimension(norm_slice; k = k).mean
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
    Colorbar(fig[1, 2], hm; label = c_label, ticks = c_ticks)
    return fig
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
    return _plot_normalization_overlay_impl(
        X -> mean(dims(X; k = k_neighbor, tol = tolerance)),
        dataset, normalization_methods, tau_values;
        feature_indices = feature_indices,
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        tau_max = tau_max
    )
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
    return _plot_normalization_overlay_impl(
        X -> compute_local_pr_dimension(X; k = k_neighbor).mean,
        dataset, normalization_methods, tau_values;
        feature_indices = feature_indices,
        ylabel = L"\text{Mean Local PR Dimension } \langle d_{\mathrm{PR}} \rangle",
        tau_max = tau_max,
        figure_size = figure_size
    )
end

function plot_local_pr_dimension(pr_scan::NamedTuple; palette = (:darkviolet, :teal), figure_size::Tuple{Integer, Integer} = (850, 520))
    set_publication_theme()
    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Participation Ratio Dimension } \langle d_{\mathrm{PR}} \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    band!(ax, pr_scan.tau_values, pr_scan.mean_dims .- pr_scan.std_dims, pr_scan.mean_dims .+ pr_scan.std_dims; color = (palette[1], 0.22), label = L"\pm 1\sigma_{\mathrm{PR}}")
    lines!(ax, pr_scan.tau_values, pr_scan.mean_dims; color = palette[1], linewidth = 3.0, label = L"\langle d_{\mathrm{PR}} \rangle")
    axislegend(ax, position = :rt)
    return fig
end

function plot_pr_k_dependency(
        dataset::AbstractArray{<:Real},
        k_values::AbstractVector{<:Integer},
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize_method::Union{Symbol, Function, AbstractVector{<:Real}} = :standard,
        palette = [:dodgerblue, :forestgreen, :darkorange, :crimson, :purple],
        figure_size::Tuple{Integer, Integer} = (900, 560),
        tau_max::Union{Nothing, Real} = 6.0
    )
    set_publication_theme()
    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1], xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\text{Participation Ratio Dimension } \langle d_{\mathrm{PR}} \rangle", xautolimitmargin = (0.0, 0.04), yautolimitmargin = (0.05, 0.05))
    !isnothing(tau_max) && xlims!(ax, 0.0, Float64(tau_max))
    for (k_idx, k_val) in enumerate(k_values)
        color = palette[mod1(k_idx, length(palette))]
        scan_res = scan_local_pr_dimension(dataset, tau_values; feature_indices = feature_indices, normalize_method = normalize_method, k = k_val)
        band!(ax, tau_values, scan_res.mean_dims .- scan_res.std_dims, scan_res.mean_dims .+ scan_res.std_dims; color = (color, 0.2))
        lines!(ax, tau_values, scan_res.mean_dims; color = color, linewidth = 2.5, label = L"K = %$(k_val)")
    end
    axislegend(ax, position = :rt)
    return fig
end

plot_local_pca(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_local_pca(to_2d_local_plots(dataset), args...; kwargs...)
plot_local_pca_regularizations(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_local_pca_regularizations(to_2d_local_plots(dataset), args...; kwargs...)
plot_map_lpca(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_map_lpca(to_2d_local_plots(dataset), args...; kwargs...)
plot_pr_normalization_direct_overlay(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_pr_normalization_direct_overlay(to_2d_local_plots(dataset), args...; kwargs...)
plot_normalization_direct_overlay(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_normalization_direct_overlay(to_2d_local_plots(dataset), args...; kwargs...)
