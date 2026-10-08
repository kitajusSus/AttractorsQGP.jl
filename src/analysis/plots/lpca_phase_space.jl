@views function plot_phase_space_lpca_dims(
        dataset::AbstractMatrix{<:Real};
        liczba_sąsiadów::Int = 20,
        wybrane_czasy::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        tolerancja_czasowa::Real = 1.0e-3,
        tolerancja_wartości_własnych::Real = 0.01,
        metoda_regularyzacji::Union{Symbol, Function} = :max,
        wymiary_cech::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        zmienna_x::Union{Symbol, Integer, Tuple} = :T,
        zmienna_y::Union{Symbol, Integer, Tuple} = :A,
        zmienna_z::Union{Nothing, Symbol, Integer, Tuple} = nothing,
        rozmiar_punktu::Real = 7.0,
        przezroczystość::Real = 0.85,
        tytuł_wykresu::Union{Nothing, String} = nothing
    )
    set_publication_theme()
    taus_in_data = sort(unique(dataset[:, 1]))
    target_taus = isnothing(wybrane_czasy) ? taus_in_data : (wybrane_czasy isa Real ? [Float64(wybrane_czasy)] : Float64.(wybrane_czasy))

    all_indices = Int[]
    all_dims = Float64[]

    for τ in target_taus
        rows = findall(isapprox.(dataset[:, 1], τ; atol = tolerancja_czasowa))
        if isempty(rows)
            nearest_idx = argmin(abs.(dataset[:, 1] .- τ))
            rows = findall(isapprox.(dataset[:, 1], dataset[nearest_idx, 1]; atol = tolerancja_czasowa))
        end
        X_tau = dataset[rows, wymiary_cech]
        size(X_tau, 1) < 2 && continue
        X_norm = apply_normalization(X_tau, metoda_regularyzacji)
        d_local = dims(X_norm; k = min(liczba_sąsiadów, size(X_tau, 1)), tol = tolerancja_wartości_własnych)
        append!(all_indices, rows)
        append!(all_dims, d_local)
    end
    isempty(all_indices) && error("Nie znaleziono żadnych punktów danych dla podanych kryteriów czasu.")

    sub_dataset = dataset[all_indices, :]
    lbl_x, vals_x = _extract_column_or_key(sub_dataset, zmienna_x)
    lbl_y, vals_y = _extract_column_or_key(sub_dataset, zmienna_y)
    fig = Figure(size = (950, 650))
    palette = [:crimson, :dodgerblue, :forestgreen, :darkorange, :purple, :goldenrod, :darkcyan, :mediumvioletred]
    unique_dims = sort(unique(all_dims))

    if zmienna_z !== nothing
        lbl_z, vals_z = _extract_column_or_key(sub_dataset, zmienna_z)
        default_title = tytuł_wykresu !== nothing ? tytuł_wykresu : L"\text{Lokalny wymiar } d_{\mathrm{lokalny}} \text{ w przestrzeni fazowej 3D}"
        ax3 = Axis3(fig[1, 1], title = default_title, xlabel = lbl_x, ylabel = lbl_y, zlabel = lbl_z, azimuth = 1.3 * π, elevation = 0.15 * π)
        for (i, d_val) in enumerate(unique_dims)
            mask = all_dims .== d_val
            scatter!(ax3, vals_x[mask], vals_y[mask], vals_z[mask]; color = (palette[mod1(i, length(palette))], przezroczystość), markersize = rozmiar_punktu, strokewidth = 0.2, strokecolor = :white, label = L"d_{\mathrm{lokalny}} = %$(Int(round(d_val)))")
        end
        axislegend(ax3, position = :rt)
    else
        default_title = tytuł_wykresu !== nothing ? tytuł_wykresu : L"\text{Lokalny wymiar } d_{\mathrm{lokalny}} \text{ w przestrzeni fazowej}"
        ax2 = Axis(fig[1, 1], title = default_title, xlabel = lbl_x, ylabel = lbl_y, xautolimitmargin = (0.05, 0.05), yautolimitmargin = (0.05, 0.05))
        for (i, d_val) in enumerate(unique_dims)
            mask = all_dims .== d_val
            scatter!(ax2, vals_x[mask], vals_y[mask]; color = (palette[mod1(i, length(palette))], przezroczystość), markersize = rozmiar_punktu, strokewidth = 0.2, strokecolor = :white, label = L"d_{\mathrm{lokalny}} = %$(Int(round(d_val)))")
        end
        axislegend(ax2, position = :rt)
    end
    return fig
end

const plot_phase_space_local_pca_dimensions = plot_phase_space_lpca_dims

function plot_colored_phase_space_slice_2d(
        slice_data::PointwiseDimensionSlice;
        x_col_idx::Integer = 1,
        y_col_idx::Integer = 2,
        x_label::LaTeXString = L"T\,[\mathrm{fm}^{-1}]",
        y_label::LaTeXString = L"\mathcal{A}",
        attractor_curve::Union{NamedTuple, Nothing} = nothing,
        colormap = lpca_style,
        figure_size::Tuple{Integer, Integer} = (800, 600),
        n_plot::Integer = 5_000,
        seed::Integer = 42
    )
    set_publication_theme()
    tau_str = string(round(slice_data.tau, digits = 2))
    figure = Figure(size = figure_size, figure_padding = (35, 35, 25, 25))
    axis = Axis(figure[1, 1], title = L"\tau = %$(tau_str)\,\mathrm{fm}/c", titlesize = 24, xlabel = x_label, ylabel = y_label, xautolimitmargin = (0.04, 0.04), yautolimitmargin = (0.05, 0.05))

    x_values = slice_data.coordinates[:, x_col_idx]
    y_values = slice_data.coordinates[:, y_col_idx]
    dim_values = slice_data.dimensions
    n_pts = length(dim_values)
    if n_pts > n_plot
        rng = MersenneTwister(seed)
        p_idxs = randperm(rng, n_pts)[1:n_plot]
        x_values = x_values[p_idxs]
        y_values = y_values[p_idxs]
        dim_values = dim_values[p_idxs]
    end

    c1 = _get_dim_color(1.0, 2.0, colormap)
    c2 = _get_dim_color(2.0, 2.0, colormap)
    palette = Dict(1.0 => (c1, L"d = 1"), 2.0 => (c2, L"d = 2"))

    if !isnothing(attractor_curve)
        lines!(axis, attractor_curve.x, attractor_curve.y; color = (:black, 0.6), linewidth = 3.0, linestyle = :dash, label = L"\text{Attractor}")
    end

    for target_dim in (2.0, 1.0)
        mask = isapprox.(dim_values, target_dim; atol = 0.1)
        if any(mask)
            color, label_text = palette[target_dim]
            scatter!(axis, x_values[mask], y_values[mask]; color = (color, 0.85), markersize = 8, strokewidth = 0.3, strokecolor = (:black, 0.3), label = label_text)
        end
    end
    axislegend(axis, position = :rt)
    return figure
end

function plot_colored_phase_space_grid_2d(
        dataset::AbstractArray{<:Real},
        tau_grid::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = [2, 3],
        x_label::LaTeXString = L"T\,[\mathrm{fm}^{-1}]",
        y_label::LaTeXString = L"\mathcal{A}",
        k_neighbor::Integer = 24,
        tolerance::Real = 0.01,
        normalize_method::Symbol = :max,
        return_normalized::Bool = false,
        colormap = lpca_style,
        figure_size::Tuple{Integer, Integer} = (1400, 1200),
        n_calc::Integer = 10_000,
        n_plot::Integer = 5_000,
        seed::Integer = 42,
        show_legend::Bool = false
    )
    set_publication_theme()
    slice_count = length(tau_grid)
    column_count = min(3, slice_count)
    row_count = ceil(Int, slice_count / column_count)
    figure = Figure(size = (figure_size[1] + 250, figure_size[2] + 200), figure_padding = (70, 60, 50, 50))
    colgap!(figure.layout, 70)
    rowgap!(figure.layout, 75)

    c2 = _get_dim_color(2.0, 2.0, colormap)
    c1 = _get_dim_color(1.0, 2.0, colormap)
    legend_placed = false

    for (index, tau) in enumerate(tau_grid)
        row = div(index - 1, column_count) + 1
        col = mod1(index, column_count)
        tau_str = string(round(tau, digits = 2))
        axis = Axis(figure[row, col], title = L"\tau = %$(tau_str)\,\mathrm{fm}/c", titlesize = 24, xlabel = x_label, ylabel = y_label, xlabelsize = 22, ylabelsize = 22, xticklabelsize = 16, yticklabelsize = 16, xticks = LinearTicks(4), yticks = LinearTicks(4), xautolimitmargin = (0.05, 0.05), yautolimitmargin = (0.05, 0.05))

        slice_data = compute_pointwise_dimensions(dataset, tau; feature_indices = feature_indices, k_neighbor = k_neighbor, tolerance = tolerance, normalize_method = normalize_method, return_normalized = return_normalized, max_points = n_calc, seed = seed + index)
        x_vals = slice_data.coordinates[:, 1]
        y_vals = slice_data.coordinates[:, 2]
        d_vals = slice_data.dimensions
        if length(d_vals) > n_plot
            rng_plot = MersenneTwister(seed + index + 10_000)
            plot_idxs = randperm(rng_plot, length(d_vals))[1:n_plot]
            x_vals, y_vals, d_vals = x_vals[plot_idxs], y_vals[plot_idxs], d_vals[plot_idxs]
        end

        mask_d2 = isapprox.(d_vals, 2.0; atol = 0.1)
        mask_d1 = isapprox.(d_vals, 1.0; atol = 0.1)
        any(mask_d2) && scatter!(axis, x_vals[mask_d2], y_vals[mask_d2]; color = (c2, 0.75), markersize = 6, strokewidth = 0.2, strokecolor = (:black, 0.2), label = show_legend ? L"d = 2" : nothing)
        any(mask_d1) && scatter!(axis, x_vals[mask_d1], y_vals[mask_d1]; color = (c1, 0.85), markersize = 6, strokewidth = 0.2, strokecolor = (:black, 0.2), label = show_legend ? L"d = 1" : nothing)

        if show_legend && !legend_placed && (any(mask_d1) && any(mask_d2) || index == slice_count)
            axislegend(axis, position = :rt)
            legend_placed = true
        end
    end
    return figure
end

function plot_colored_phase_space_grid_hjsw_projections(
        dataset::AbstractArray{<:Real},
        tau_grid::AbstractVector{<:Real};
        k_neighbor::Integer = 24,
        tolerance::Real = 0.01,
        normalize_method::Symbol = :max,
        colormap = lpca_style,
        figure_size::Tuple{Integer, Integer} = (1200, 1050),
        n_calc::Integer = 10_000,
        n_plot::Integer = 5_000,
        seed::Integer = 42,
        show_legend::Bool = false
    )
    set_publication_theme()
    slice_count = length(tau_grid)
    column_count = min(3, slice_count)
    row_count = ceil(Int, slice_count / column_count)
    figure = Figure(size = (figure_size[1] + 250, figure_size[2] + 200), figure_padding = (65, 55, 50, 50))
    colgap!(figure.layout, 65)
    rowgap!(figure.layout, 70)

    c3 = _get_dim_color(3.0, 3.0, colormap)
    c2 = _get_dim_color(2.0, 3.0, colormap)
    c1 = _get_dim_color(1.0, 3.0, colormap)
    legend_placed = false

    for (index, tau) in enumerate(tau_grid)
        row = div(index - 1, column_count) + 1
        col = mod1(index, column_count)
        tau_str = string(round(tau, digits = 2))
        axis = Axis(figure[row, col], title = L"\tau = %$(tau_str)\,\mathrm{fm}/c", titlesize = 24, xlabel = L"\mathcal{A}", ylabel = L"\mathcal{B}", xlabelsize = 22, ylabelsize = 22, xticklabelsize = 16, yticklabelsize = 16, xticks = LinearTicks(4), yticks = LinearTicks(4), xautolimitmargin = (0.04, 0.04), yautolimitmargin = (0.05, 0.05))

        slice_data = compute_pointwise_dimensions(dataset, tau; feature_indices = [2, 3, 4], k_neighbor = k_neighbor, tolerance = tolerance, normalize_method = normalize_method, max_points = n_calc, seed = seed + index)
        a_vals, b_vals, d_vals = slice_data.coordinates[:, 2], slice_data.coordinates[:, 3], slice_data.dimensions
        if length(d_vals) > n_plot
            rng_plot = MersenneTwister(seed + index + 10_000)
            plot_idxs = randperm(rng_plot, length(d_vals))[1:n_plot]
            a_vals, b_vals, d_vals = a_vals[plot_idxs], b_vals[plot_idxs], d_vals[plot_idxs]
        end

        mask_d3 = isapprox.(d_vals, 3.0; atol = 0.1)
        mask_d2 = isapprox.(d_vals, 2.0; atol = 0.1)
        mask_d1 = isapprox.(d_vals, 1.0; atol = 0.1)
        any(mask_d3) && scatter!(axis, a_vals[mask_d3], b_vals[mask_d3]; color = (c3, 0.6), markersize = 5, label = show_legend ? L"d = 3" : nothing)
        any(mask_d2) && scatter!(axis, a_vals[mask_d2], b_vals[mask_d2]; color = (c2, 0.8), markersize = 5, label = show_legend ? L"d = 2" : nothing)
        any(mask_d1) && scatter!(axis, a_vals[mask_d1], b_vals[mask_d1]; color = (c1, 0.9), markersize = 6, label = show_legend ? L"d = 1" : nothing)

        if show_legend && !legend_placed && (any(mask_d3) && any(mask_d2) || index == slice_count)
            axislegend(axis, position = :rt)
            legend_placed = true
        end
    end
    return figure
end

function plot_colored_phase_space_slice_hjsw_3d(
        slice_data::PointwiseDimensionSlice;
        x_label::LaTeXString = L"T\,[\mathrm{MeV}]",
        y_label::LaTeXString = L"\mathcal{A}",
        z_label::LaTeXString = L"\mathcal{B}",
        azimuth::Real = -1.2,
        elevation::Real = 0.2,
        colormap = lpca_style,
        figure_size::Tuple{Integer, Integer} = (900, 750),
        n_plot::Integer = 5_000,
        seed::Integer = 42
    )
    set_publication_theme()
    tau_str = string(round(slice_data.tau, digits = 2))
    figure = Figure(size = figure_size, figure_padding = (100, 60, 70, 50))
    axis = Axis3(figure[1, 1], title = L"\tau = %$(tau_str)\,\mathrm{fm}/c", titlesize = 24, xlabel = x_label, ylabel = y_label, zlabel = z_label, azimuth = azimuth, elevation = elevation, xlabeloffset = 40, ylabeloffset = 40, zlabeloffset = 50, xticks = LinearTicks(4), yticks = LinearTicks(4), zticks = LinearTicks(4))

    t_vals, a_vals, b_vals, d_vals = slice_data.coordinates[:, 1], slice_data.coordinates[:, 2], slice_data.coordinates[:, 3], slice_data.dimensions
    if length(d_vals) > n_plot
        rng = MersenneTwister(seed)
        p_idxs = randperm(rng, length(d_vals))[1:n_plot]
        t_vals, a_vals, b_vals, d_vals = t_vals[p_idxs], a_vals[p_idxs], b_vals[p_idxs], d_vals[p_idxs]
    end

    c3, c2, c1 = _get_dim_color(3.0, 3.0, colormap), _get_dim_color(2.0, 3.0, colormap), _get_dim_color(1.0, 3.0, colormap)
    mask_d3, mask_d2, mask_d1 = isapprox.(d_vals, 3.0; atol = 0.1), isapprox.(d_vals, 2.0; atol = 0.1), isapprox.(d_vals, 1.0; atol = 0.1)

    any(mask_d3) && scatter!(axis, t_vals[mask_d3], a_vals[mask_d3], b_vals[mask_d3]; color = (c3, 0.5), markersize = 6, label = L"d = 3")
    any(mask_d2) && scatter!(axis, t_vals[mask_d2], a_vals[mask_d2], b_vals[mask_d2]; color = (c2, 0.75), markersize = 6, label = L"d = 2")
    any(mask_d1) && scatter!(axis, t_vals[mask_d1], a_vals[mask_d1], b_vals[mask_d1]; color = (c1, 0.9), markersize = 8, label = L"d = 1")
    axislegend(axis, position = :rt)
    return figure
end

function plot_colored_phase_space_grid_3d(
        dataset::AbstractArray{<:Real},
        tau_grid::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = [2, 3, 4],
        x_label::LaTeXString = L"T\,[\mathrm{MeV}]",
        y_label::LaTeXString = L"\mathcal{A}",
        z_label::LaTeXString = L"\mathcal{B}",
        k_neighbor::Integer = 24,
        tolerance::Real = 0.01,
        normalize_method::Symbol = :max,
        return_normalized::Bool = false,
        colormap = lpca_style,
        azimuth::Real = 1.2,
        elevation::Real = 0.25,
        xreversed::Bool = true,
        markersize::Real = 5,
        figure_size::Union{Nothing, Tuple{Integer, Integer}} = nothing,
        n_calc::Integer = 10_000,
        n_plot::Integer = 5_000,
        seed::Integer = 42,
        show_legend::Bool = false
    )
    set_publication_theme()
    slice_count = length(tau_grid)
    column_count = min(3, slice_count)
    row_count = ceil(Int, slice_count / column_count)
    figsize = isnothing(figure_size) ? (720 * column_count + 200, 620 * row_count) : figure_size
    fig = Figure(size = figsize, figure_padding = (80, 70, 60, 60))
    colgap!(fig.layout, 80)
    rowgap!(fig.layout, 90)

    c3, c2, c1 = _get_dim_color(3.0, 3.0, colormap), _get_dim_color(2.0, 3.0, colormap), _get_dim_color(1.0, 3.0, colormap)
    legend_placed = false

    for (index, tau) in enumerate(tau_grid)
        row = div(index - 1, column_count) + 1
        col = mod1(index, column_count)
        tau_str = string(round(tau, digits = 2))
        ax = Axis3(fig[row, col], title = L"\tau = %$(tau_str)\,\mathrm{fm}/c", titlesize = 24, xlabel = x_label, ylabel = y_label, zlabel = z_label, xlabelsize = 22, ylabelsize = 22, zlabelsize = 22, xticklabelsize = 16, yticklabelsize = 16, zticklabelsize = 16, azimuth = azimuth, elevation = elevation, xreversed = xreversed, xlabeloffset = 55, ylabeloffset = 55, zlabeloffset = 70, xticks = LinearTicks(4), yticks = LinearTicks(4), zticks = LinearTicks(4))

        slice_data = compute_pointwise_dimensions(dataset, tau; feature_indices = feature_indices, k_neighbor = k_neighbor, tolerance = tolerance, normalize_method = normalize_method, return_normalized = return_normalized, max_points = n_calc, seed = seed + index)
        t_vals, a_vals, b_vals, d_vals = slice_data.coordinates[:, 1], slice_data.coordinates[:, 2], slice_data.coordinates[:, 3], slice_data.dimensions
        if length(d_vals) > n_plot
            rng_plot = MersenneTwister(seed + index + 10_000)
            plot_idxs = randperm(rng_plot, length(d_vals))[1:n_plot]
            t_vals, a_vals, b_vals, d_vals = t_vals[plot_idxs], a_vals[plot_idxs], b_vals[plot_idxs], d_vals[plot_idxs]
        end

        mask_d3, mask_d2, mask_d1 = isapprox.(d_vals, 3.0; atol = 0.1), isapprox.(d_vals, 2.0; atol = 0.1), isapprox.(d_vals, 1.0; atol = 0.1)
        any(mask_d3) && scatter!(ax, t_vals[mask_d3], a_vals[mask_d3], b_vals[mask_d3]; color = (c3, 0.5), markersize = markersize, label = show_legend ? L"d = 3" : nothing)
        any(mask_d2) && scatter!(ax, t_vals[mask_d2], a_vals[mask_d2], b_vals[mask_d2]; color = (c2, 0.75), markersize = markersize, label = show_legend ? L"d = 2" : nothing)
        any(mask_d1) && scatter!(ax, t_vals[mask_d1], a_vals[mask_d1], b_vals[mask_d1]; color = (c1, 0.9), markersize = markersize + 1, label = show_legend ? L"d = 1" : nothing)

        if show_legend && !legend_placed && (any(mask_d3) && any(mask_d2) || index == slice_count)
            axislegend(ax, position = :rt)
            legend_placed = true
        end
    end
    return fig
end

function plot_soft_phase_space_slice_2d(dataset::AbstractArray{<:Real}, tau::Real; feature_indices = [2, 3], x_label = L"T\,[\mathrm{fm}^{-1}]", y_label = L"\mathcal{A}", colormap = lpca_style, color_limits = (1.0, 2.0), figure_size = (850, 600), kwargs...)
    set_publication_theme()
    _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
    norm_slice = apply_normalization(raw_slice, :max)
    res = compute_soft_weighted_dimension(norm_slice; tau_val = tau, kwargs...)
    fig = Figure(size = figure_size, figure_padding = (35, 35, 25, 25))
    ax = Axis(fig[1, 1], title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c", titlesize = 19, xlabel = x_label, ylabel = y_label, xautolimitmargin = (0.04, 0.04), yautolimitmargin = (0.05, 0.05))
    sc = scatter!(ax, raw_slice[:, 1], raw_slice[:, 2]; color = res.d_soft, colormap = colormap, colorrange = color_limits, markersize = 7, strokewidth = 0.2, strokecolor = (:black, 0.2))
    Colorbar(fig[1, 2], sc, label = L"\text{Soft Dimension } d_i^{\mathrm{soft}}", width = 16, ticklabelsize = 14, labelsize = 16)
    return fig
end

function plot_soft_phase_space_grid_2d(dataset::AbstractArray{<:Real}, tau_grid::AbstractVector{<:Real}; feature_indices = [2, 3], x_label = L"T\,[\mathrm{fm}^{-1}]", y_label = L"\mathcal{A}", colormap = lpca_style, color_limits = (1.0, 2.0), figure_size = (1250, 1050), kwargs...)
    set_publication_theme()
    slice_count = length(tau_grid)
    column_count = min(3, slice_count)
    row_count = ceil(Int, slice_count / column_count)
    fig = Figure(size = (figure_size[1] + 250, figure_size[2] + 200), figure_padding = (70, 60, 50, 50))
    colgap!(fig.layout, 70); rowgap!(fig.layout, 75)
    sc_handle = nothing

    for (index, tau) in enumerate(tau_grid)
        row = div(index - 1, column_count) + 1
        col = mod1(index, column_count)
        ax = Axis(fig[row, col], title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c", titlesize = 24, xlabel = x_label, ylabel = y_label, xlabelsize = 22, ylabelsize = 22, xticklabelsize = 16, yticklabelsize = 16, xticks = LinearTicks(4), yticks = LinearTicks(4), xautolimitmargin = (0.04, 0.04), yautolimitmargin = (0.05, 0.05))
        _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
        norm_slice = apply_normalization(raw_slice, :max)
        res = compute_soft_weighted_dimension(norm_slice; tau_val = tau, kwargs...)
        sc = scatter!(ax, raw_slice[:, 1], raw_slice[:, 2]; color = res.d_soft, colormap = colormap, colorrange = color_limits, markersize = 6, strokewidth = 0.2, strokecolor = (:black, 0.2))
        isnothing(sc_handle) && (sc_handle = sc)
    end
    Colorbar(fig[1:row_count, column_count + 1], sc_handle, label = L"\text{Soft Dimension } d_i^{\mathrm{soft}}", width = 22, ticklabelsize = 16, labelsize = 20)
    return fig
end

function plot_soft_phase_space_slice_3d(dataset::AbstractArray{<:Real}, tau::Real; feature_indices = [2, 3, 4], x_label = L"T\,[\mathrm{MeV}]", y_label = L"\mathcal{A}", z_label = L"\mathcal{B}", colormap = lpca_style, color_limits = (1.0, 3.0), azimuth = 1.3, elevation = 0.15, figure_size = (950, 750), kwargs...)
    set_publication_theme()
    _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
    norm_slice = apply_normalization(raw_slice, :max)
    res = compute_soft_weighted_dimension(norm_slice; tau_val = tau, kwargs...)
    fig = Figure(size = figure_size, figure_padding = (90, 50, 60, 40))
    ax = Axis3(fig[1, 1], title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c", titlesize = 20, xlabel = x_label, ylabel = y_label, zlabel = z_label, azimuth = azimuth, elevation = elevation, xlabeloffset = 45, ylabeloffset = 45, zlabeloffset = 60)
    sc = scatter!(ax, raw_slice[:, 1], raw_slice[:, 2], raw_slice[:, 3]; color = res.d_soft, colormap = colormap, colorrange = color_limits, markersize = 6)
    Colorbar(fig[1, 2], sc, label = L"\text{Soft Dimension } d_i^{\mathrm{soft}}", width = 18, ticklabelsize = 14, labelsize = 16)
    return fig
end

function plot_soft_phase_space_grid_3d(dataset::AbstractArray{<:Real}, tau_grid::AbstractVector{<:Real}; feature_indices = [2, 3, 4], x_label = L"T\,[\mathrm{MeV}]", y_label = L"\mathcal{A}", z_label = L"\mathcal{B}", colormap = lpca_style, color_limits = (1.0, 3.0), azimuth = 1.3, elevation = 0.25, markersize = 5, kwargs...)
    set_publication_theme()
    slice_count = length(tau_grid)
    column_count = min(3, slice_count)
    row_count = ceil(Int, slice_count / column_count)
    fig = Figure(size = (720 * column_count + 200, 620 * row_count), figure_padding = (80, 70, 60, 60))
    colgap!(fig.layout, 80); rowgap!(fig.layout, 90)
    sc_handle = nothing

    for (index, tau) in enumerate(tau_grid)
        row = div(index - 1, column_count) + 1
        col = mod1(index, column_count)
        ax = Axis3(fig[row, col], title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c", titlesize = 24, xlabel = x_label, ylabel = y_label, zlabel = z_label, xlabelsize = 22, ylabelsize = 22, zlabelsize = 22, xticklabelsize = 16, yticklabelsize = 16, zticklabelsize = 16, azimuth = azimuth, elevation = elevation, xlabeloffset = 55, ylabeloffset = 55, zlabeloffset = 70, xticks = LinearTicks(4), yticks = LinearTicks(4), zticks = LinearTicks(4))
        _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
        norm_slice = apply_normalization(raw_slice, :max)
        res = compute_soft_weighted_dimension(norm_slice; tau_val = tau, kwargs...)
        sc = scatter!(ax, raw_slice[:, 1], raw_slice[:, 2], raw_slice[:, 3]; color = res.d_soft, colormap = colormap, colorrange = color_limits, markersize = markersize)
        isnothing(sc_handle) && (sc_handle = sc)
    end
    Colorbar(fig[1:row_count, column_count + 1], sc_handle, label = L"\text{Soft Dimension } d_i^{\mathrm{soft}}", width = 22, ticklabelsize = 16, labelsize = 20)
    return fig
end

function plot_pr_phase_space_slice_2d(dataset::AbstractArray{<:Real}, tau::Real; feature_indices = [2, 3], x_label = L"T\,[\mathrm{fm}^{-1}]", y_label = L"\mathcal{A}", colormap = lpca_style, color_limits = (1.0, 2.0), normalize_method = :max, return_normalized = false, figure_size = (850, 600), n_calc = 10_000, n_plot = 5_000, seed = 42, kwargs...)
    set_publication_theme()
    _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
    calc_slice = size(raw_slice, 1) > n_calc ? raw_slice[randperm(MersenneTwister(seed), size(raw_slice, 1))[1:n_calc], :] : raw_slice
    norm_slice = apply_normalization(calc_slice, normalize_method)
    res = compute_local_pr_dimension(norm_slice; kwargs...)

    pts = return_normalized ? norm_slice : calc_slice
    d_vals = res.d_pr
    if size(pts, 1) > n_plot
        p_idxs = randperm(MersenneTwister(seed + 10_000), size(pts, 1))[1:n_plot]
        pts, d_vals = pts[p_idxs, :], d_vals[p_idxs]
    end

    fig = Figure(size = figure_size, figure_padding = (35, 35, 25, 25))
    ax = Axis(fig[1, 1], title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c", titlesize = 24, xlabel = x_label, ylabel = y_label, xautolimitmargin = (0.04, 0.04), yautolimitmargin = (0.05, 0.05))
    sc = scatter!(ax, pts[:, 1], pts[:, 2]; color = d_vals, colormap = colormap, colorrange = color_limits, markersize = 7, strokewidth = 0.2, strokecolor = (:black, 0.2))
    Colorbar(fig[1, 2], sc, label = L"\text{Local PR Dimension } d_{\mathrm{PR}}(x_i)", width = 16, ticklabelsize = 14, labelsize = 16)
    return fig
end

function plot_pr_phase_space_slice_3d(dataset::AbstractArray{<:Real}, tau::Real; feature_indices = [2, 3, 4], x_label = L"T\,[\mathrm{MeV}]", y_label = L"\mathcal{A}", z_label = L"\mathcal{B}", colormap = lpca_style, color_limits = (1.0, 3.0), azimuth = -1.2, elevation = 0.2, normalize_method = :max, return_normalized = false, figure_size = (950, 750), n_calc = 10_000, n_plot = 5_000, seed = 42, kwargs...)
    set_publication_theme()
    _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
    calc_slice = size(raw_slice, 1) > n_calc ? raw_slice[randperm(MersenneTwister(seed), size(raw_slice, 1))[1:n_calc], :] : raw_slice
    norm_slice = apply_normalization(calc_slice, normalize_method)
    res = compute_local_pr_dimension(norm_slice; kwargs...)

    pts = return_normalized ? norm_slice : calc_slice
    d_vals = res.d_pr
    if size(pts, 1) > n_plot
        p_idxs = randperm(MersenneTwister(seed + 10_000), size(pts, 1))[1:n_plot]
        pts, d_vals = pts[p_idxs, :], d_vals[p_idxs]
    end

    fig = Figure(size = figure_size, figure_padding = (90, 50, 60, 40))
    ax = Axis3(fig[1, 1], title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c", titlesize = 24, xlabel = x_label, ylabel = y_label, zlabel = z_label, azimuth = azimuth, elevation = elevation, xlabeloffset = 40, ylabeloffset = 40, zlabeloffset = 50, xticks = LinearTicks(4), yticks = LinearTicks(4), zticks = LinearTicks(4))
    sc = scatter!(ax, pts[:, 1], pts[:, 2], pts[:, 3]; color = d_vals, colormap = colormap, colorrange = color_limits, markersize = 6)
    Colorbar(fig[1, 2], sc, label = L"\text{Local PR Dimension } d_{\mathrm{PR}}(x_i)", width = 18, ticklabelsize = 14, labelsize = 16)
    return fig
end

function plot_pr_phase_space_grid_2d(dataset::AbstractArray{<:Real}, tau_grid::AbstractVector{<:Real}; feature_indices = [2, 3], x_label = L"T\,[\mathrm{fm}^{-1}]", y_label = L"\mathcal{A}", colormap = lpca_style, color_limits = (1.0, 2.0), normalize_method = :max, return_normalized = false, figure_size = (1450, 1200), n_calc = 10_000, n_plot = 5_000, seed = 42, kwargs...)
    set_publication_theme()
    slice_count = length(tau_grid)
    column_count = min(3, slice_count)
    row_count = ceil(Int, slice_count / column_count)
    fig = Figure(size = (figure_size[1] + 250, figure_size[2] + 200), figure_padding = (70, 60, 50, 50))
    colgap!(fig.layout, 70); rowgap!(fig.layout, 75)
    sc_handle = nothing

    for (index, tau) in enumerate(tau_grid)
        row = div(index - 1, column_count) + 1
        col = mod1(index, column_count)
        ax = Axis(fig[row, col], title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c", titlesize = 24, xlabel = x_label, ylabel = y_label, xlabelsize = 22, ylabelsize = 22, xticklabelsize = 16, yticklabelsize = 16, xticks = LinearTicks(4), yticks = LinearTicks(4), xautolimitmargin = (0.05, 0.05), yautolimitmargin = (0.05, 0.05))

        _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
        calc_slice = size(raw_slice, 1) > n_calc ? raw_slice[randperm(MersenneTwister(seed + index), size(raw_slice, 1))[1:n_calc], :] : raw_slice
        norm_calc = apply_normalization(calc_slice, normalize_method)
        res = compute_local_pr_dimension(norm_calc; kwargs...)

        pts = return_normalized ? norm_calc : calc_slice
        plot_idxs = size(pts, 1) > n_plot ? randperm(MersenneTwister(seed + index + 10_000), size(pts, 1))[1:n_plot] : collect(1:size(pts, 1))
        sc = scatter!(ax, pts[plot_idxs, 1], pts[plot_idxs, 2]; color = res.d_pr[plot_idxs], colormap = colormap, colorrange = color_limits, markersize = 6, strokewidth = 0.2, strokecolor = (:black, 0.2))
        isnothing(sc_handle) && (sc_handle = sc)
    end
    Colorbar(fig[1:row_count, column_count + 1], sc_handle, label = L"\text{Local PR Dimension } d_{\mathrm{PR}}(x_i)", width = 22, ticklabelsize = 16, labelsize = 20)
    return fig
end

function plot_pr_phase_space_grid_3d(dataset::AbstractArray{<:Real}, tau_grid::AbstractVector{<:Real}; feature_indices = [2, 3, 4], x_label = L"T\,[\mathrm{MeV}]", y_label = L"\mathcal{A}", z_label = L"\mathcal{B}", colormap = lpca_style, color_limits = (1.0, 3.0), azimuth = 1.2, elevation = 0.25, xreversed = true, markersize = 5, normalize_method = :max, return_normalized = false, n_calc = 10_000, n_plot = 5_000, seed = 42, kwargs...)
    set_publication_theme()
    slice_count = length(tau_grid)
    column_count = min(3, slice_count)
    row_count = ceil(Int, slice_count / column_count)
    fig = Figure(size = (720 * column_count + 200, 620 * row_count), figure_padding = (80, 70, 60, 60))
    colgap!(fig.layout, 80); rowgap!(fig.layout, 90)
    sc_handle = nothing

    for (index, tau) in enumerate(tau_grid)
        row = div(index - 1, column_count) + 1
        col = mod1(index, column_count)
        ax = Axis3(fig[row, col], title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c", titlesize = 24, xlabel = x_label, ylabel = y_label, zlabel = z_label, xlabelsize = 22, ylabelsize = 22, zlabelsize = 22, xticklabelsize = 16, yticklabelsize = 16, zticklabelsize = 16, azimuth = azimuth, elevation = elevation, xreversed = xreversed, xlabeloffset = 55, ylabeloffset = 55, zlabeloffset = 70, xticks = LinearTicks(4), yticks = LinearTicks(4), zticks = LinearTicks(4))

        _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
        calc_slice = size(raw_slice, 1) > n_calc ? raw_slice[randperm(MersenneTwister(seed + index), size(raw_slice, 1))[1:n_calc], :] : raw_slice
        norm_calc = apply_normalization(calc_slice, normalize_method)
        res = compute_local_pr_dimension(norm_calc; kwargs...)

        pts = return_normalized ? norm_calc : calc_slice
        plot_idxs = size(pts, 1) > n_plot ? randperm(MersenneTwister(seed + index + 10_000), size(pts, 1))[1:n_plot] : collect(1:size(pts, 1))
        sc = scatter!(ax, pts[plot_idxs, 1], pts[plot_idxs, 2], pts[plot_idxs, 3]; color = res.d_pr[plot_idxs], colormap = colormap, colorrange = color_limits, markersize = markersize)
        isnothing(sc_handle) && (sc_handle = sc)
    end
    Colorbar(fig[1:row_count, column_count + 1], sc_handle, label = L"\text{Local PR Dimension } d_{\mathrm{PR}}(x_i)", width = 22, ticklabelsize = 16, labelsize = 20)
    return fig
end

plot_phase_space_lpca_dims(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_phase_space_lpca_dims(to_2d_local_plots(dataset), args...; kwargs...)
