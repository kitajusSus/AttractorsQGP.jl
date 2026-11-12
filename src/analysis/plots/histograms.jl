function _bin_pr_dimensions(
        d_vals::AbstractVector{<:Real};
        n_bins::Integer = 40,
        range_limits::Tuple{<:Real, <:Real} = (1.0, 2.0)
    )
    low, high = range_limits
    bin_width = (high - low) / n_bins
    centers = range(low + bin_width / 2, high - bin_width / 2, length = n_bins)
    counts = zeros(Int, n_bins)

    @inbounds for v in d_vals
        if isfinite(v) && (low <= v <= high)
            idx = min(floor(Int, (v - low) / bin_width) + 1, n_bins)
            counts[idx] += 1
        end
    end

    max_c = maximum(counts; init = 0)
    scaled = max_c > 0 ? counts ./ max_c : zeros(Float64, n_bins)
    return collect(centers), scaled, bin_width
end

function _bin_pr_dimensions_stacked(
        d_vals::AbstractVector{<:Real},
        color_vals::AbstractVector{<:Real};
        n_bins::Integer = 40,
        range_limits::Tuple{<:Real, <:Real} = (1.0, 2.0),
        n_color_bins::Integer = 8,
        color_range::Tuple{<:Real, <:Real} = extrema(filter(isfinite, color_vals))
    )
    low_d, high_d = range_limits
    bin_width_d = (high_d - low_d) / n_bins
    centers_d = collect(range(low_d + bin_width_d / 2, high_d - bin_width_d / 2, length = n_bins))

    low_c, high_c = color_range
    if low_c == high_c
        low_c -= 0.1
        high_c += 0.1
    end
    bin_width_c = (high_c - low_c) / n_color_bins

    H = zeros(Float64, n_bins, n_color_bins)

    @inbounds for (d_val, c_val) in zip(d_vals, color_vals)
        if isfinite(d_val) && (low_d <= d_val <= high_d) && isfinite(c_val)
            idx_d = clamp(floor(Int, (d_val - low_d) / bin_width_d) + 1, 1, n_bins)
            idx_c = clamp(floor(Int, (c_val - low_c) / bin_width_c) + 1, 1, n_color_bins)
            H[idx_d, idx_c] += 1
        end
    end

    total_per_bin = vec(sum(H, dims = 2))
    max_c = maximum(total_per_bin; init = 0.0)
    H_norm = max_c > 0 ? H ./ max_c : H

    return centers_d, H_norm, bin_width_d, (low_c, high_c)
end

function _is_initial_feature(color_by)::Bool
    if color_by in (:initial_A, :A0, :initial_anisotropy, :A, :anisotropy, :initial_T, :T0, :initial_temperature, :initial_B, :B0)
        return true
    elseif color_by in (:current_A, :instantaneous_A, :instant_A, :current_T, :T, :temperature, :current_B, :B)
        return false
    else
        return !(color_by isa Integer || color_by isa AbstractVector)
    end
end

function _normalize_slice(feature_slice::AbstractMatrix{<:Real}, feature_slice0::AbstractMatrix{<:Real}, method::Union{Symbol, Function, AbstractVector{<:Real}})
    if method isa Symbol && method in (:var1, :frozen, :var2, :var3)
        scales0 = std(feature_slice0, dims = 1)
        scales0[scales0 .== 0.0] .= 1.0
        if method in (:var1, :frozen)
            return feature_slice ./ scales0
        end
        other_scales = length(scales0) > 1 ? scales0[2:end] : Float64[]
        scaled_others = !isempty(other_scales) ? (feature_slice[:, 2:end] ./ reshape(other_scales, 1, :)) : Matrix{Float64}(undef, size(feature_slice, 1), 0)
        mean_T0 = mean(feature_slice0[:, 1])

        if method === :var2
            mean_Tt = mean(feature_slice[:, 1])
            scale_T = scales0[1] * (mean_T0 > 0 ? (mean_Tt / mean_T0) : 1.0)
            scale_T = scale_T == 0.0 ? 1.0 : scale_T
            return hcat(feature_slice[:, 1] ./ scale_T, scaled_others)
        else
            T0_pos = filter(x -> x > 0, feature_slice0[:, 1])
            std_lnT0 = isempty(T0_pos) ? 1.0 : std(log.(T0_pos))
            std_lnT0 = std_lnT0 == 0.0 ? 1.0 : std_lnT0
            y_T = log.(max.(feature_slice[:, 1], 1.0e-12) ./ (mean_T0 > 0 ? mean_T0 : 1.0)) ./ std_lnT0
            return hcat(y_T, scaled_others)
        end
    end
    return apply_normalization(feature_slice, method)
end

function _resolve_color_feature(
        slice_data::AbstractMatrix{<:Real},
        initial_slice_data::AbstractMatrix{<:Real},
        color_by,
        custom_label = nothing
    )
    ncols = size(slice_data, 2)
    has_tau_col = ncols >= 3
    n_pts = size(slice_data, 1)
    is_aligned = (n_pts == size(initial_slice_data, 1))
    init_src = is_aligned ? initial_slice_data : slice_data

    if color_by in (:initial_A, :A0, :initial_anisotropy, :A, :anisotropy)
        col_idx = has_tau_col ? 3 : 2
        return init_src[:, col_idx], something(custom_label, L"\mathcal{A}_0")
    elseif color_by in (:current_A, :instantaneous_A, :instant_A)
        col_idx = has_tau_col ? 3 : 2
        return slice_data[:, col_idx], something(custom_label, L"\mathcal{A}")
    elseif color_by in (:initial_T, :T0, :initial_temperature)
        col_idx = has_tau_col ? 2 : 1
        return init_src[:, col_idx], something(custom_label, L"T_0\,[\mathrm{fm}^{-1}]")
    elseif color_by in (:current_T, :T, :temperature)
        col_idx = has_tau_col ? 2 : 1
        return slice_data[:, col_idx], something(custom_label, L"T\,[\mathrm{fm}^{-1}]")
    elseif color_by in (:initial_B, :B0)
        col_idx = has_tau_col ? 4 : 3
        vals = ncols >= col_idx ? init_src[:, col_idx] : zeros(n_pts)
        return vals, something(custom_label, L"\mathcal{B}_0")
    elseif color_by in (:current_B, :B)
        col_idx = has_tau_col ? 4 : 3
        vals = ncols >= col_idx ? slice_data[:, col_idx] : zeros(n_pts)
        return vals, something(custom_label, L"\mathcal{B}")
    elseif color_by isa Integer
        return slice_data[:, color_by], something(custom_label, LaTeXString("x_{$color_by}"))
    elseif color_by isa AbstractVector
        return color_by, something(custom_label, L"\text{Feature}")
    else
        col_idx = has_tau_col ? 3 : min(2, ncols)
        return init_src[:, col_idx], something(custom_label, L"\mathcal{A}_0")
    end
end

_resolve_color_feature(slice_data::AbstractMatrix{<:Real}, color_by, custom_label = nothing) =
    _resolve_color_feature(slice_data, slice_data, color_by, custom_label)

function _safe_extrema(vals::AbstractVector{<:Real})
    finite_vals = filter(isfinite, vals)
    isempty(finite_vals) && return (0.0, 1.0)
    lo, hi = extrema(finite_vals)
    return lo == hi ? (lo - 0.1, hi + 0.1) : (lo, hi)
end

function _compute_effective_color_meta(
        dataset::AbstractMatrix{<:Real},
        tau_grid::AbstractVector{<:Real},
        full_slice0::AbstractMatrix{<:Real},
        is_stacked::Bool,
        color_by::Union{Symbol, Integer, Nothing},
        color_val_limits::Union{Tuple{<:Real, <:Real}, Nothing},
        color_label::Union{LaTeXString, String, Nothing},
        actual_color_limits::Tuple{<:Real, <:Real}
    )
    !is_stacked && return (actual_color_limits, L"d_{\mathrm{PR}}")

    _, default_lbl = _resolve_color_feature(full_slice0, full_slice0, color_by, color_label)
    c_range = if !isnothing(color_val_limits)
        color_val_limits
    elseif _is_initial_feature(color_by)
        full_vals0, _ = _resolve_color_feature(full_slice0, full_slice0, color_by)
        _safe_extrema(full_vals0)
    else
        all_color_vals = Float64[]
        for tg in tau_grid
            _, s = get_tau_slice(dataset, tg; feature_cols = 1:size(dataset, 2))
            vals, _ = _resolve_color_feature(s, s, color_by)
            append!(all_color_vals, filter(isfinite, vals))
        end
        _safe_extrema(all_color_vals)
    end
    return (c_range, default_lbl)
end

function _prepare_tau_calc_slices(
        dataset::AbstractMatrix{<:Real},
        tau::Real,
        full_slice0::AbstractMatrix{<:Real},
        n_calc::Integer,
        rng_seed::Integer
    )
    _, full_slice = get_tau_slice(dataset, tau; feature_cols = 1:size(dataset, 2))
    same_len = (size(full_slice0, 1) == size(full_slice, 1))
    if size(full_slice, 1) > n_calc
        rng = MersenneTwister(rng_seed)
        p_idxs = randperm(rng, size(full_slice, 1))[1:n_calc]
        return full_slice[p_idxs, :], (same_len ? full_slice0[p_idxs, :] : full_slice0)
    else
        return full_slice, (same_len ? full_slice0 : full_slice)
    end
end

function _plot_pr_histogram_panel!(
        grid_pos,
        tau::Real,
        calc_slice::AbstractMatrix{<:Real},
        calc_slice0::AbstractMatrix{<:Real},
        actual_features::AbstractVector{<:Integer},
        actual_color_limits::Tuple{<:Real, <:Real},
        effective_color_range::Tuple{<:Real, <:Real},
        cs;
        k_neighbor::Integer = 120,
        normalize_method::Union{Symbol, Function, AbstractVector{<:Real}} = :var1,
        n_bins::Integer = 40,
        is_stacked::Bool = true,
        color_by::Union{Symbol, Integer, Nothing} = :initial_A,
        n_color_bins::Integer = 8,
        colormap = lpca_style,
        titlesize::Real = 22,
        xlabel = L"d_{\mathrm{PR}}",
        ylabel = L"N / N_{\max}",
        xlabelsize::Real = 20,
        ylabelsize::Real = 20,
        show_mean::Bool = true,
        mean_legend::Bool = false
    )
    norm_slice = _normalize_slice(calc_slice[:, actual_features], calc_slice0[:, actual_features], normalize_method)
    res = compute_local_pr_dimension(norm_slice; k = Int(k_neighbor))

    ax = Axis(
        grid_pos,
        title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c",
        titlesize = titlesize,
        xlabel = xlabel,
        ylabel = ylabel,
        xlabelsize = xlabelsize,
        ylabelsize = ylabelsize,
        xticklabelsize = 16,
        yticklabelsize = 16,
        limits = (actual_color_limits[1] - 0.04, actual_color_limits[2] + 0.04, 0.0, 1.05)
    )

    if is_stacked
        color_vals, _ = _resolve_color_feature(calc_slice, calc_slice0, color_by)
        centers, H_norm, bin_width, _ = _bin_pr_dimensions_stacked(
            res.d_pr, color_vals;
            n_bins = n_bins, range_limits = actual_color_limits,
            n_color_bins = n_color_bins, color_range = effective_color_range
        )
        bottom = zeros(Float64, n_bins)
        for j in 1:n_color_bins
            bar_color = ColorSchemes.get(cs, (j - 0.5) / n_color_bins)
            barplot!(
                ax, centers, H_norm[:, j];
                offset = bottom, width = bin_width * 0.95,
                color = bar_color, strokewidth = 0.2, strokecolor = (:black, 0.25)
            )
            bottom .+= H_norm[:, j]
        end
    else
        centers, scaled, bin_width = _bin_pr_dimensions(res.d_pr; n_bins = n_bins, range_limits = actual_color_limits)
        barplot!(
            ax, centers, scaled;
            width = bin_width * 0.95, color = centers,
            colormap = colormap, colorrange = actual_color_limits,
            strokewidth = 0.4, strokecolor = (:black, 0.3)
        )
    end

    if show_mean
        vlines!(
            ax, [res.mean],
            color = :black, linestyle = :dash,
            linewidth = mean_legend ? 2.5 : 2.0,
            label = L"\langle d_{\mathrm{PR}} \rangle = %$(round(res.mean, digits=2))"
        )
        mean_legend && axislegend(ax, position = :rt, labelsize = 18)
    end

    return ax
end

function plot_pr_dimension_histogram(
        dataset::AbstractMatrix{<:Real},
        tau::Real;
        feature_indices::Union{AbstractVector{<:Integer}, Nothing} = nothing,
        k_neighbor::Integer = 120,
        normalize_method::Union{Symbol, Function, AbstractVector{<:Real}} = :var1,
        n_bins::Integer = 40,
        color_by::Union{Symbol, Integer, Nothing} = :initial_A,
        n_color_bins::Integer = 8,
        color_val_limits::Union{Tuple{<:Real, <:Real}, Nothing} = nothing,
        color_label::Union{LaTeXString, String, Nothing} = nothing,
        colormap = lpca_style,
        color_limits::Union{Tuple{<:Real, <:Real}, Nothing} = nothing,
        figure_size::Tuple{Integer, Integer} = (850, 550),
        show_mean::Bool = false,
        n_calc::Integer = 10_000,
        seed::Integer = 5
    )
    set_publication_theme()
    actual_features = isnothing(feature_indices) ? collect(2:size(dataset, 2)) : feature_indices
    actual_color_limits = isnothing(color_limits) ? (1.0, Float64(length(actual_features))) : color_limits

    is_stacked = color_by !== nothing && color_by !== :d_pr
    cs = resolve_palette(colormap)

    tau0 = minimum(dataset[:, 1])
    _, full_slice0 = get_tau_slice(dataset, tau0; feature_cols = 1:size(dataset, 2))
    calc_slice, calc_slice0 = _prepare_tau_calc_slices(dataset, tau, full_slice0, n_calc, seed)

    effective_color_range, effective_color_label = _compute_effective_color_meta(
        dataset, [Float64(tau)], full_slice0, is_stacked,
        color_by, color_val_limits, color_label, actual_color_limits
    )

    fig = Figure(size = figure_size, figure_padding = (40, 40, 30, 30))
    _plot_pr_histogram_panel!(
        fig[1, 1], tau, calc_slice, calc_slice0,
        actual_features, actual_color_limits, effective_color_range, cs;
        k_neighbor = k_neighbor, normalize_method = normalize_method,
        n_bins = n_bins, is_stacked = is_stacked, color_by = color_by,
        n_color_bins = n_color_bins, colormap = colormap,
        titlesize = 24, xlabel = L"\text{Local PR Dimension } d_{\mathrm{PR}}",
        xlabelsize = 22, ylabelsize = 22,
        show_mean = show_mean, mean_legend = true
    )

    if is_stacked
        Colorbar(fig[1, 2], colormap = cs, limits = effective_color_range, label = effective_color_label, width = 16)
    else
        Colorbar(fig[1, 2], colormap = cs, limits = actual_color_limits, label = L"d_{\mathrm{PR}}", width = 16, ticks = actual_color_limits[1]:0.5:actual_color_limits[2])
    end

    return fig
end

function plot_pr_dimension_histogram_grid(
        dataset::AbstractMatrix{<:Real},
        tau_grid::AbstractVector{<:Real};
        feature_indices::Union{AbstractVector{<:Integer}, Nothing} = nothing,
        k_neighbor::Integer = 120,
        normalize_method::Union{Symbol, Function, AbstractVector{<:Real}} = :var1,
        n_bins::Integer = 40,
        color_by::Union{Symbol, Integer, Nothing} = :initial_A,
        n_color_bins::Integer = 8,
        color_val_limits::Union{Tuple{<:Real, <:Real}, Nothing} = nothing,
        color_label::Union{LaTeXString, String, Nothing} = nothing,
        colormap = lpca_style,
        color_limits::Union{Tuple{<:Real, <:Real}, Nothing} = nothing,
        figure_size::Tuple{Integer, Integer} = (1450, 1150),
        show_mean::Bool = true,
        n_calc::Integer = 10_000,
        seed::Integer = 5
    )
    set_publication_theme()
    actual_features = isnothing(feature_indices) ? collect(2:size(dataset, 2)) : feature_indices
    actual_color_limits = isnothing(color_limits) ? (1.0, Float64(length(actual_features))) : color_limits

    slice_count = length(tau_grid)
    ncols = min(3, slice_count)
    nrows = ceil(Int, slice_count / ncols)

    fig = Figure(size = figure_size, figure_padding = (65, 55, 45, 45))
    colgap!(fig.layout, 55)
    rowgap!(fig.layout, 65)

    is_stacked = color_by !== nothing && color_by !== :d_pr
    cs = resolve_palette(colormap)

    tau0 = minimum(dataset[:, 1])
    _, full_slice0 = get_tau_slice(dataset, tau0; feature_cols = 1:size(dataset, 2))

    effective_color_range, effective_color_label = _compute_effective_color_meta(
        dataset, tau_grid, full_slice0, is_stacked,
        color_by, color_val_limits, color_label, actual_color_limits
    )

    for (idx, tau) in enumerate(tau_grid)
        row = div(idx - 1, ncols) + 1
        col = mod1(idx, ncols)
        calc_slice, calc_slice0 = _prepare_tau_calc_slices(dataset, tau, full_slice0, n_calc, seed + idx)

        _plot_pr_histogram_panel!(
            fig[row, col], tau, calc_slice, calc_slice0,
            actual_features, actual_color_limits, effective_color_range, cs;
            k_neighbor = k_neighbor, normalize_method = normalize_method,
            n_bins = n_bins, is_stacked = is_stacked, color_by = color_by,
            n_color_bins = n_color_bins, colormap = colormap,
            xlabel = row == nrows ? L"d_{\mathrm{PR}}" : "",
            ylabel = col == 1 ? L"N / N_{\max}" : "",
            show_mean = show_mean, mean_legend = false
        )
    end

    Colorbar(
        fig[:, ncols + 1],
        colormap = cs,
        limits = effective_color_range,
        label = effective_color_label,
        width = 18,
        labelsize = 22,
        ticklabelsize = 16
    )

    return fig
end

plot_pr_dimension_histogram(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_pr_dimension_histogram(to_2d_local_plots(dataset), args...; kwargs...)
plot_pr_dimension_histogram_grid(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_pr_dimension_histogram_grid(to_2d_local_plots(dataset), args...; kwargs...)
