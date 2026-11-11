@views function plot_phase_space_lpca_dims(
        dataset::AbstractMatrix{<:Real};
        liczba_sąsiadów::Int = 20,
        wybrane_czasy::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        tolerancja_czasowa::Real = 1.0e-3,
        tolerancja_wartości_własnych::Real = 0.01,
        metoda_regularyzacji::Union{Symbol, Function, AbstractVector{<:Real}} = :zscore,
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

function plot_pr_phase_space_slice_2d(
        dataset::AbstractArray{<:Real},
        tau::Real;
        feature_indices = [2, 3],
        x_label = L"T\,[\mathrm{fm}^{-1}]",
        y_label = L"\mathcal{A}",
        colormap = lpca_style,
        color_limits = (1.0, 2.0),
        normalize_method = :zscore,
        return_normalized = false,
        figure_size = (850, 600),
        n_calc = 10_000,
        n_plot = 5_000,
        seed = 42,
        k::Integer = 120,
        k_neighbor::Union{Nothing, Integer} = nothing
    )
    set_publication_theme()
    eff_k = !isnothing(k_neighbor) ? k_neighbor : k
    _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
    calc_slice = size(raw_slice, 1) > n_calc ? raw_slice[randperm(MersenneTwister(seed), size(raw_slice, 1))[1:n_calc], :] : raw_slice
    norm_slice = apply_normalization(calc_slice, normalize_method)
    res = compute_local_pr_dimension(norm_slice; k = eff_k)

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

function plot_pr_phase_space_slice_3d(
        dataset::AbstractArray{<:Real},
        tau::Real;
        feature_indices = [2, 3, 4],
        x_label = L"T\,[\mathrm{MeV}]",
        y_label = L"\mathcal{A}",
        z_label = L"\mathcal{B}",
        colormap = lpca_style,
        color_limits = (1.0, 3.0),
        azimuth = -1.2,
        elevation = 0.2,
        normalize_method = :zscore,
        return_normalized = false,
        figure_size = (950, 750),
        n_calc = 10_000,
        n_plot = 5_000,
        seed = 42,
        k::Integer = 120,
        k_neighbor::Union{Nothing, Integer} = nothing
    )
    set_publication_theme()
    eff_k = !isnothing(k_neighbor) ? k_neighbor : k
    _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
    calc_slice = size(raw_slice, 1) > n_calc ? raw_slice[randperm(MersenneTwister(seed), size(raw_slice, 1))[1:n_calc], :] : raw_slice
    norm_slice = apply_normalization(calc_slice, normalize_method)
    res = compute_local_pr_dimension(norm_slice; k = eff_k)

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

function plot_pr_phase_space_grid_2d(
        dataset::AbstractArray{<:Real},
        tau_grid::AbstractVector{<:Real};
        feature_indices = [2, 3],
        x_label = L"T\,[\mathrm{fm}^{-1}]",
        y_label = L"\mathcal{A}",
        colormap = lpca_style,
        color_limits = (1.0, 2.0),
        normalize_method = :zscore,
        return_normalized = false,
        figure_size = (1450, 1200),
        n_calc = 10_000,
        n_plot = 5_000,
        seed = 42,
        k::Integer = 120,
        k_neighbor::Union{Nothing, Integer} = nothing
    )
    set_publication_theme()
    eff_k = !isnothing(k_neighbor) ? k_neighbor : k
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
        res = compute_local_pr_dimension(norm_calc; k = eff_k)

        pts = return_normalized ? norm_calc : calc_slice
        plot_idxs = size(pts, 1) > n_plot ? randperm(MersenneTwister(seed + index + 10_000), size(pts, 1))[1:n_plot] : collect(1:size(pts, 1))
        sc = scatter!(ax, pts[plot_idxs, 1], pts[plot_idxs, 2]; color = res.d_pr[plot_idxs], colormap = colormap, colorrange = color_limits, markersize = 6, strokewidth = 0.2, strokecolor = (:black, 0.2))
        isnothing(sc_handle) && (sc_handle = sc)
    end
    Colorbar(fig[1:row_count, column_count + 1], sc_handle, label = L"\text{Local PR Dimension } d_{\mathrm{PR}}(x_i)", width = 22, ticklabelsize = 16, labelsize = 20)
    return fig
end

function plot_pr_phase_space_grid_3d(
        dataset::AbstractArray{<:Real},
        tau_grid::AbstractVector{<:Real};
        feature_indices = [2, 3, 4],
        x_label = L"T\,[\mathrm{MeV}]",
        y_label = L"\mathcal{A}",
        z_label = L"\mathcal{B}",
        colormap = lpca_style,
        color_limits = (1.0, 3.0),
        azimuth = 1.2,
        elevation = 0.25,
        xreversed = true,
        markersize = 5,
        normalize_method = :zscore,
        return_normalized = false,
        n_calc = 10_000,
        n_plot = 5_000,
        seed = 42,
        k::Integer = 120,
        k_neighbor::Union{Nothing, Integer} = nothing
    )
    set_publication_theme()
    eff_k = !isnothing(k_neighbor) ? k_neighbor : k
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
        res = compute_local_pr_dimension(norm_calc; k = eff_k)

        pts = return_normalized ? norm_calc : calc_slice
        plot_idxs = size(pts, 1) > n_plot ? randperm(MersenneTwister(seed + index + 10_000), size(pts, 1))[1:n_plot] : collect(1:size(pts, 1))
        sc = scatter!(ax, pts[plot_idxs, 1], pts[plot_idxs, 2], pts[plot_idxs, 3]; color = res.d_pr[plot_idxs], colormap = colormap, colorrange = color_limits, markersize = markersize)
        isnothing(sc_handle) && (sc_handle = sc)
    end
    Colorbar(fig[1:row_count, column_count + 1], sc_handle, label = L"\text{Local PR Dimension } d_{\mathrm{PR}}(x_i)", width = 22, ticklabelsize = 16, labelsize = 20)
    return fig
end

plot_phase_space_lpca_dims(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_phase_space_lpca_dims(to_2d_local_plots(dataset), args...; kwargs...)
