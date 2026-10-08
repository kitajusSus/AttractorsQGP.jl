function _bin_pr_dimensions(
        d_vals::AbstractVector{<:Real};
        n_bins::Integer = 35,
        range_limits::Tuple{<:Real, <:Real} = (1.0, 2.0)
    )
    edges = range(range_limits[1], range_limits[2], length = n_bins + 1)
    centers = [(edges[i] + edges[i + 1]) / 2 for i in 1:n_bins]
    counts = zeros(Int, n_bins)
    bin_width = step(edges)

    for v in d_vals
        if isfinite(v)
            v_clamped = clamp(v, range_limits[1] + 1.0e-6, range_limits[2] - 1.0e-6)
            idx = clamp(floor(Int, (v_clamped - range_limits[1]) / bin_width) + 1, 1, n_bins)
            counts[idx] += 1
        end
    end
    return centers, counts, bin_width
end

function plot_pr_dimension_histogram(
        dataset::AbstractArray{<:Real},
        tau::Real;
        feature_indices::AbstractVector{<:Integer} = [2, 3],
        k_neighbor::Integer = 8,
        normalize_method::Symbol = :zscore,
        use_density_weights::Bool = false,
        n_bins::Integer = 40,
        colormap = japidi_stype,
        color_limits::Tuple{<:Real, <:Real} = (1.0, 2.0),
        figure_size::Tuple{Integer, Integer} = (850, 550),
        show_mean::Bool = false,
        n_calc::Integer = 10_000,
        seed::Integer = 5
    )
    set_publication_theme()

    _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
    calc_slice = size(raw_slice, 1) > n_calc ? raw_slice[randperm(MersenneTwister(seed), size(raw_slice, 1))[1:n_calc], :] : raw_slice
    norm_slice = apply_normalization(calc_slice, normalize_method)
    res = compute_local_pr_dimension(norm_slice; k = k_neighbor, use_density_weights = use_density_weights)

    centers, counts, bin_width = _bin_pr_dimensions(res.d_pr; n_bins = n_bins, range_limits = color_limits)

    fig = Figure(size = figure_size, figure_padding = (40, 40, 30, 30))
    ax = Axis(
        fig[1, 1],
        title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c",
        titlesize = 24,
        xlabel = L"\text{Local PR Dimension } d_{\mathrm{PR}}",
        ylabel = L"\text{Number of points} N",
        xlabelsize = 22,
        ylabelsize = 22,
        xticklabelsize = 16,
        yticklabelsize = 16,
        limits = (color_limits[1] - 0.04, color_limits[2] + 0.04, 0, maximum(counts) * 1.15)
    )

    bp = barplot!(
        ax,
        centers,
        counts;
        width = bin_width * 0.95,
        color = centers,
        colormap = colormap,
        colorrange = color_limits,
        strokewidth = 0.5,
        strokecolor = (:black, 0.4)
    )

    if show_mean
        vlines!(ax, [res.mean], color = :black, linestyle = :dash, linewidth = 2.5, label = L"\langle d_{\mathrm{PR}} \rangle = %$(round(res.mean, digits=2))")
        axislegend(ax, position = :rt, labelsize = 18)
    end

    Colorbar(fig[1, 2], bp, label = L"d_{\mathrm{PR}}", width = 16, ticks = color_limits[1]:0.5:color_limits[2])
    return fig
end

function plot_pr_dimension_histogram_grid(
        dataset::AbstractArray{<:Real},
        tau_grid::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = [2, 3],
        k_neighbor::Integer = 8,
        normalize_method::Symbol = :zscore,
        use_density_weights::Bool = true,
        n_bins::Integer = 40,
        colormap = japidi_stype,
        color_limits::Tuple{<:Real, <:Real} = (1.0, 2.0),
        figure_size::Tuple{Integer, Integer} = (1400, 1150),
        n_calc::Integer = 10_000,
        seed::Integer = 5
    )
    set_publication_theme()
    slice_count = length(tau_grid)
    ncols = min(3, slice_count)
    nrows = ceil(Int, slice_count / ncols)

    fig = Figure(size = figure_size, figure_padding = (65, 55, 45, 45))
    colgap!(fig.layout, 55)
    rowgap!(fig.layout, 65)

    for (idx, tau) in enumerate(tau_grid)
        row = div(idx - 1, ncols) + 1
        col = mod1(idx, ncols)

        _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
        calc_slice = size(raw_slice, 1) > n_calc ? raw_slice[randperm(MersenneTwister(seed + idx), size(raw_slice, 1))[1:n_calc], :] : raw_slice
        norm_slice = apply_normalization(calc_slice, normalize_method)
        res = compute_local_pr_dimension(norm_slice; k = k_neighbor, use_density_weights = use_density_weights)

        centers, counts, bin_width = _bin_pr_dimensions(res.d_pr; n_bins = n_bins, range_limits = color_limits)

        ax = Axis(
            fig[row, col],
            title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c",
            titlesize = 22,
            xlabel = row == nrows ? L"d_{\mathrm{PR}}" : "",
            ylabel = col == 1 ? L"N" : "",
            xlabelsize = 20,
            ylabelsize = 20,
            xticklabelsize = 16,
            yticklabelsize = 16,
            limits = (color_limits[1] - 0.04, color_limits[2] + 0.04, 0, maximum(counts) * 1.25)
        )

        barplot!(
            ax,
            centers,
            counts;
            width = bin_width * 0.95,
            color = centers,
            colormap = colormap,
            colorrange = color_limits,
            strokewidth = 0.4,
            strokecolor = (:black, 0.3)
        )

        vlines!(ax, [res.mean], color = :black, linestyle = :dash, linewidth = 2.0)
        text!(ax, color_limits[1] + 0.05, maximum(counts) * 1.05, text = L"\langle d_{\mathrm{PR}} \rangle = %$(round(res.mean, digits=2))", fontsize = 16)
    end
    return fig
end

plot_pr_dimension_histogram(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_pr_dimension_histogram(to_2d_local_plots(dataset), args...; kwargs...)
plot_pr_dimension_histogram_grid(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_pr_dimension_histogram_grid(to_2d_local_plots(dataset), args...; kwargs...)
