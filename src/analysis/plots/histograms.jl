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

"""
    _bin_pr_dimensions_stacked(d_vals, color_vals; n_bins, range_limits, n_color_bins, color_range)

Tworzy dwuwymiarowy rozkład (stacked) dla wymiaru d_PR podzielonego pionowo na n_color_bins warstw cechy.
Zwraca:
- centers_d: środki binów d_PR
- H_norm: znormalizowana macierz [n_bins, n_color_bins], gdzie suma wzdłuż warstw daje N_i / N_max <= 1.0
- bin_width_d: szerokość binu d_PR
- (low_c, high_c): zakres wartości cechy kolorującej
"""
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

function _resolve_color_feature(
        slice_data::AbstractMatrix{<:Real},
        initial_slice_data::AbstractMatrix{<:Real},
        color_by,
        custom_label = nothing
    )
    ncols = size(slice_data, 2)
    has_tau_col = ncols >= 3
    n_pts = size(slice_data, 1)
    n_pts0 = size(initial_slice_data, 1)
    is_aligned = (n_pts == n_pts0)

    if color_by in (:initial_A, :A0, :initial_anisotropy, :A, :anisotropy)
        col_idx = has_tau_col ? 3 : 2
        vals = is_aligned ? initial_slice_data[:, col_idx] : slice_data[:, col_idx]
        return vals, isnothing(custom_label) ? L"\mathcal{A}_0" : custom_label

    elseif color_by in (:current_A, :instantaneous_A, :instant_A)
        col_idx = has_tau_col ? 3 : 2
        return slice_data[:, col_idx], isnothing(custom_label) ? L"\mathcal{A}" : custom_label

    elseif color_by in (:initial_T, :T0, :initial_temperature)
        col_idx = has_tau_col ? 2 : 1
        vals = is_aligned ? initial_slice_data[:, col_idx] : slice_data[:, col_idx]
        return vals, isnothing(custom_label) ? L"T_0\,[\mathrm{fm}^{-1}]" : custom_label

    elseif color_by in (:current_T, :T, :temperature)
        col_idx = has_tau_col ? 2 : 1
        return slice_data[:, col_idx], isnothing(custom_label) ? L"T\,[\mathrm{fm}^{-1}]" : custom_label

    elseif color_by in (:initial_B, :B0)
        col_idx = has_tau_col ? 4 : 3
        data_src = is_aligned ? initial_slice_data : slice_data
        vals = ncols >= col_idx ? data_src[:, col_idx] : zeros(n_pts)
        return vals, isnothing(custom_label) ? L"\mathcal{B}_0" : custom_label

    elseif color_by in (:current_B, :B)
        col_idx = has_tau_col ? 4 : 3
        vals = ncols >= col_idx ? slice_data[:, col_idx] : zeros(n_pts)
        return vals, isnothing(custom_label) ? L"\mathcal{B}" : custom_label

    elseif color_by isa Integer
        return slice_data[:, color_by], isnothing(custom_label) ? LaTeXString("x_{$color_by}") : custom_label

    elseif color_by isa AbstractVector
        return color_by, isnothing(custom_label) ? L"\text{Feature}" : custom_label

    else
        col_idx = has_tau_col ? 3 : min(2, ncols)
        vals = is_aligned ? initial_slice_data[:, col_idx] : slice_data[:, col_idx]
        return vals, isnothing(custom_label) ? L"\mathcal{A}_0" : custom_label
    end
end

_resolve_color_feature(slice_data::AbstractMatrix{<:Real}, color_by, custom_label = nothing) =
    _resolve_color_feature(slice_data, slice_data, color_by, custom_label)

"""
    plot_pr_dimension_histogram(dataset, tau; kwargs...) -> Figure

Rysuje histogram rozkładu lokalnego wymiaru partycypacji \$d_{\\mathrm{PR}}\$ w wybranym przekroju czasowym \$\\tau\$,
z pionową segmentacją słupków (stacked barplot) i domyślnym kolorowaniem według początkowej anizotropii \$\\mathcal{A}_0\$
w palecie `lpca_style`.

### Cechy i zasada działania:
1. **Segmentacja pionowa (Stacked Bar):**
   - Funkcja `_bin_pr_dimensions_stacked` dzieli każdy słupek wymiaru \$d_{\\mathrm{PR}}\$ na \$M\$ warstw (domyślnie `n_color_bins = 8`)
     według wybranej cechy (domyślnie początkowa anizotropia \$\\mathcal{A}_0 \\equiv \\mathcal{A}(\\tau_0)\$).
   - Suma segmentów w danym binie wynosi dokładnie \$N_i / N_{\\max} \\le 1.0\$, dzięki czemu obwiednia słupków zachowuje
     dokładnie dotychczasowy profil unormowanego rozkładu gęstości.
2. **Kolorowanie z paletą `lpca_style`:**
   - Każda warstwa pobiera kolor z palety `lpca_style` proporcjonalnie do swojej wartości cechy.
   - Segmenty mają subtelny obrys (`strokewidth = 0.2`, `strokecolor = (:black, 0.25)`), zapobiegający zlewaniu się warstw.
   - Pasek `Colorbar` po prawej stronie odpowiada wartości cechy (np. \$\\mathcal{A}_0\$ z etykietą `L"\\mathcal{A}_0"`).
3. **Pełna elastyczność (`color_by`):**
   - Domyślnie: `color_by = :initial_A` (początkowa anizotropia \$\\mathcal{A}_0\$; synonimy: `:A0`, `:A`, `:anisotropy`).
   - Chwilowa anizotropia w chwili \$\\tau\$: `color_by = :current_A` (synonim: `:instantaneous_A`).
   - Temperatura: `color_by = :T` lub `color_by = :initial_T`.
   - Cecha modelu HJSW: `color_by = :B` lub `color_by = :initial_B`.
   - Jednolity histogram (bez podziału warstw): `color_by = nothing` lub `color_by = :d_pr`.

### Przykłady użycia:
```julia
using AttractorsQGP

# 1. Histogram skumulowany dla tau = 0.45 fm/c (kolorowany po A0 z lpca_style)
fig_single = plot_pr_dimension_histogram(dataset, 0.45; color_by = :A, n_color_bins = 8)

# 2. Histogram z wyłączoną segmentacją (jednolity kolor d_PR)
fig_classic = plot_pr_dimension_histogram(dataset, 0.45; color_by = nothing)
```
"""
function plot_pr_dimension_histogram(
        dataset::AbstractMatrix{<:Real},
        tau::Real;
        feature_indices::Union{AbstractVector{<:Integer}, Nothing} = nothing,
        k_neighbor::Union{Integer, Symbol} = :auto,
        normalize_method::Symbol = :zscore,
        use_density_weights::Bool = false,
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

    _, full_slice = get_tau_slice(dataset, tau; feature_cols = 1:size(dataset, 2))
    tau0 = minimum(dataset[:, 1])
    _, full_slice0 = get_tau_slice(dataset, tau0; feature_cols = 1:size(dataset, 2))

    calc_slice, calc_slice0 = if size(full_slice, 1) > n_calc
        rng = MersenneTwister(seed)
        p_idxs = randperm(rng, size(full_slice, 1))[1:n_calc]
        s = full_slice[p_idxs, :]
        s0 = (size(full_slice0, 1) == size(full_slice, 1)) ? full_slice0[p_idxs, :] : full_slice0
        (s, s0)
    else
        s0 = (size(full_slice0, 1) == size(full_slice, 1)) ? full_slice0 : full_slice
        (full_slice, s0)
    end

    feature_slice = calc_slice[:, actual_features]
    norm_slice = apply_normalization(feature_slice, normalize_method)
    eff_k = if k_neighbor === :auto
        suggest_optimal_k(size(norm_slice, 1), size(norm_slice, 2))
    else
        Int(k_neighbor)
    end
    res = compute_local_pr_dimension(norm_slice; k = eff_k, use_density_weights = use_density_weights)

    is_stacked = color_by !== nothing && color_by !== :d_pr
    cs = resolve_palette(colormap)

    fig = Figure(size = figure_size, figure_padding = (40, 40, 30, 30))
    ax = Axis(
        fig[1, 1],
        title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c",
        titlesize = 24,
        xlabel = L"\text{Local PR Dimension } d_{\mathrm{PR}}",
        ylabel = L"N / N_{\max}",
        xlabelsize = 22,
        ylabelsize = 22,
        xticklabelsize = 16,
        yticklabelsize = 16,
        limits = (color_limits[1] - 0.04, color_limits[2] + 0.04, 0.0, 1.05)
    )

    if is_stacked
        color_vals, auto_label = _resolve_color_feature(calc_slice, calc_slice0, color_by, color_label)
        c_range = if !isnothing(color_val_limits)
            color_val_limits
        elseif _is_initial_feature(color_by)
            full_vals0, _ = _resolve_color_feature(full_slice0, full_slice0, color_by)
            c_ext = extrema(filter(isfinite, full_vals0))
            c_ext[1] == c_ext[2] ? (c_ext[1] - 0.1, c_ext[2] + 0.1) : c_ext
        else
            c_ext = extrema(filter(isfinite, color_vals))
            c_ext[1] == c_ext[2] ? (c_ext[1] - 0.1, c_ext[2] + 0.1) : c_ext
        end

        centers, H_norm, bin_width, final_c_range = _bin_pr_dimensions_stacked(
            res.d_pr,
            color_vals;
            n_bins = n_bins,
            range_limits = color_limits,
            n_color_bins = n_color_bins,
            color_range = c_range
        )

        bottom = zeros(Float64, n_bins)
        for j in 1:n_color_bins
            col = ColorSchemes.get(cs, (j - 0.5) / n_color_bins)
            barplot!(
                ax,
                centers,
                H_norm[:, j];
                offset = bottom,
                width = bin_width * 0.95,
                color = col,
                strokewidth = 0.2,
                strokecolor = (:black, 0.25)
            )
            bottom .+= H_norm[:, j]
        end

        Colorbar(fig[1, 2], colormap = cs, limits = final_c_range, label = auto_label, width = 16)
    else
        centers, scaled, bin_width = _bin_pr_dimensions(res.d_pr; n_bins = n_bins, range_limits = color_limits)
        bp = barplot!(
            ax,
            centers,
            scaled;
            width = bin_width * 0.95,
            color = centers,
            colormap = colormap,
            colorrange = color_limits,
            strokewidth = 0.5,
            strokecolor = (:black, 0.4)
        )
        Colorbar(fig[1, 2], bp, label = L"d_{\mathrm{PR}}", width = 16, ticks = color_limits[1]:0.5:color_limits[2])
    end

    if show_mean
        vlines!(ax, [res.mean], color = :black, linestyle = :dash, linewidth = 2.5, label = L"\langle d_{\mathrm{PR}} \rangle = %$(round(res.mean, digits=2))")
        axislegend(ax, position = :rt, labelsize = 18)
    end

    return fig
end

"""
    plot_pr_dimension_histogram_grid(dataset, tau_grid; kwargs...) -> Figure

Tworzy siatkę histogramów rozkładu lokalnego wymiaru partycypacji \$d_{\\mathrm{PR}}\$ dla zadanych chwil czasowych `tau_grid`,
ze wspólną skalą barw i wspólnym paskiem `Colorbar` w palecie `lpca_style`.

### Cechy i zasada działania:
1. **Segmentacja pionowa (Stacked Bar):**
   - Na każdym kafelku \$\\tau\$, słupki \$d_{\\mathrm{PR}}\$ podzielone są pionowo na \$M\$ segmentów (domyślnie `n_color_bins = 8`).
   - Całkowita wysokość słupka zachowuje unormowany profil \$N_i / N_{\\max} \\le 1.0\$.
2. **Spójna skala w siatce czasowej:**
   - Zakres cechy kolorującej (np. początkowej anizotropii \$\\mathcal{A}_0\$) wyznaczany jest spójnie dla całej siatki.
   - Ten sam odcień barwy w palecie `lpca_style` oznacza dokładnie tę samą wartość cechy na każdym panelu \$\\tau\$.
   - Po prawej stronie siatki umieszczony jest jeden wspólny pasek skali `Colorbar`.
3. **Śledzenie relaksacji i basenu przyciągania:**
   - Kolorowanie po początkowej anizotropii \$\\mathcal{A}_0\$ pozwala bezpośrednio śledzić, które warunki początkowe
     najszybciej ulegają kolapsowi do atraktora (\$d_{\\mathrm{PR}} \\approx 1.0\$).
4. **Opcje konfiguracyjne:**
   - `color_by`: `:initial_A` (domyślnie), `:A`, `:current_A`, `:T`, `:B`, `nothing` lub `:d_pr`.
   - `colormap`: domyślnie `lpca_style`.
   - `n_color_bins`: liczba warstw w każdym słupku (domyślnie 8).
   - `color_val_limits`: opcjonalny ręczny zakres wartości cechy `(min, max)`.

### Przykłady użycia:
```julia
using AttractorsQGP

grid_times = [0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80, 1.00, 2.50]
fig_grid = plot_pr_dimension_histogram_grid(dataset, grid_times; color_by = :A, n_color_bins = 8)

fig_classic = plot_pr_dimension_histogram_grid(dataset, grid_times; color_by = nothing)
```
"""
function plot_pr_dimension_histogram_grid(
        dataset::AbstractMatrix{<:Real},
        tau_grid::AbstractVector{<:Real};
        feature_indices::Union{AbstractVector{<:Integer}, Nothing} = nothing,
        k_neighbor::Union{Integer, Symbol} = :auto,
        normalize_method::Symbol = :zscore,
        use_density_weights::Bool = false,
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

    effective_color_range, effective_color_label = if is_stacked
        _, default_lbl = _resolve_color_feature(full_slice0, full_slice0, color_by, color_label)
        c_range = if !isnothing(color_val_limits)
            color_val_limits
        elseif _is_initial_feature(color_by)
            full_vals0, _ = _resolve_color_feature(full_slice0, full_slice0, color_by)
            c_ext = extrema(filter(isfinite, full_vals0))
            c_ext[1] == c_ext[2] ? (c_ext[1] - 0.1, c_ext[2] + 0.1) : c_ext
        else
            all_color_vals = Float64[]
            for tg in tau_grid
                _, s = get_tau_slice(dataset, tg; feature_cols = 1:size(dataset, 2))
                vals, _ = _resolve_color_feature(s, s, color_by)
                append!(all_color_vals, filter(isfinite, vals))
            end
            c_ext = isempty(all_color_vals) ? (0.0, 1.0) : extrema(all_color_vals)
            c_ext[1] == c_ext[2] ? (c_ext[1] - 0.1, c_ext[2] + 0.1) : c_ext
        end
        (c_range, default_lbl)
    else
        (actual_color_limits, L"d_{\mathrm{PR}}")
    end

    for (idx, tau) in enumerate(tau_grid)
        row = div(idx - 1, ncols) + 1
        col = mod1(idx, ncols)

        _, full_slice = get_tau_slice(dataset, tau; feature_cols = 1:size(dataset, 2))
        calc_slice, calc_slice0 = if size(full_slice, 1) > n_calc
            rng = MersenneTwister(seed + idx)
            p_idxs = randperm(rng, size(full_slice, 1))[1:n_calc]
            s = full_slice[p_idxs, :]
            s0 = (size(full_slice0, 1) == size(full_slice, 1)) ? full_slice0[p_idxs, :] : full_slice0
            (s, s0)
        else
            s0 = (size(full_slice0, 1) == size(full_slice, 1)) ? full_slice0 : full_slice
            (full_slice, s0)
        end

        feature_slice = calc_slice[:, actual_features]
        norm_slice = apply_normalization(feature_slice, normalize_method)
        eff_k = if k_neighbor === :auto
            suggest_optimal_k(size(norm_slice, 1), size(norm_slice, 2))
            # println("")
        else
            Int(k_neighbor)
        end
        res = compute_local_pr_dimension(norm_slice; k = eff_k, use_density_weights = use_density_weights)

        ax = Axis(
            fig[row, col],
            title = L"\tau = %$(string(round(tau, digits = 2)))\,\mathrm{fm}/c",
            titlesize = 22,
            xlabel = row == nrows ? L"d_{\mathrm{PR}}" : "",
            ylabel = col == 1 ? L"N / N_{\max}" : "",
            xlabelsize = 20,
            ylabelsize = 20,
            xticklabelsize = 16,
            yticklabelsize = 16,
            limits = (actual_color_limits[1] - 0.04, actual_color_limits[2] + 0.04, 0.0, 1.05)
        )

        if is_stacked
            color_vals, _ = _resolve_color_feature(calc_slice, calc_slice0, color_by)
            centers, H_norm, bin_width, _ = _bin_pr_dimensions_stacked(
                res.d_pr,
                color_vals;
                n_bins = n_bins,
                range_limits = actual_color_limits,
                n_color_bins = n_color_bins,
                color_range = effective_color_range
            )
            bottom = zeros(Float64, n_bins)
            for j in 1:n_color_bins
                col = ColorSchemes.get(cs, (j - 0.5) / n_color_bins)
                barplot!(
                    ax,
                    centers,
                    H_norm[:, j];
                    offset = bottom,
                    width = bin_width * 0.95,
                    color = col,
                    strokewidth = 0.2,
                    strokecolor = (:black, 0.25)
                )
                bottom .+= H_norm[:, j]
            end
        else
            centers, scaled, bin_width = _bin_pr_dimensions(res.d_pr; n_bins = n_bins, range_limits = actual_color_limits)
            barplot!(
                ax,
                centers,
                scaled;
                width = bin_width * 0.95,
                color = centers,
                colormap = colormap,
                colorrange = actual_color_limits,
                strokewidth = 0.4,
                strokecolor = (:black, 0.3)
            )
        end

        if show_mean
            vlines!(ax, [res.mean], color = :black, linestyle = :dash, linewidth = 2.0)
        end
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
