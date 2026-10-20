using Makie
using ColorSchemes
using Colors
using LaTeXStrings

const JAPIDI_COLORS = [
    "#9C1313",
    "#C40A41",
    "#911465",
    "#6A0DAD",
    "#4361EE",
    "#0096C7",  # Błękit
    "#0000FF",  # Klasyczny niebieski (koniec)
]

const LPCA_COLORS = [
    "#b00020",
    "#fb8500",
    "#2ec4b6",
    "#0096c7",
    "#0000FF",
]

japidi_grid = cgrad(JAPIDI_COLORS)
japidi_stype = ColorScheme(Makie.resample_cmap(japidi_grid, 256))

lpca_style_grad = cgrad(LPCA_COLORS)
lpca_style = ColorScheme(Makie.resample_cmap(lpca_style_grad, 256))

const THEMES_DICT = Dict{Symbol, ColorScheme}(
    :japidi => japidi_stype,
    :lpca => lpca_style,
)

function resolve_palette(p::Symbol)
    if haskey(THEMES_DICT, p)
        return THEMES_DICT[p]
    elseif haskey(ColorSchemes.colorschemes, p)
        return ColorSchemes.colorschemes[p]
    else
        error("Unknown palette: :$p")
    end
end
resolve_palette(cs::ColorScheme) = cs
resolve_palette(cg::Makie.PlotUtils.ColorGradient) = ColorScheme(cg)
resolve_palette(colors::AbstractVector) = ColorScheme(colors)

function set_publication_theme(;
        cmap = lpca_style,
        n_colors::Int = 10,
        bg_color = RGBf(0.98, 0.98, 0.98),
        font = "Libertinus Serif",
        fontsize::Int = 24
    )
    cs = resolve_palette(cmap)
    scientific_palette = Makie.resample_cmap(cs, n_colors)

    return set_theme!(
        Theme(
            font = font,
            fontsize = fontsize,
            figure_padding = 20,
            Axis = (
                backgroundcolor = bg_color,
                titlesize = fontsize + 4,
                xlabelsize = fontsize + 2,
                ylabelsize = fontsize + 2,
                xticklabelsize = fontsize - 4,
                yticklabelsize = fontsize - 4,
                xgridstyle = :dash,
                ygridstyle = :dash,
                xgridcolor = RGBAf(0.8, 0.8, 0.8, 0.7),
                ygridcolor = RGBAf(0.8, 0.8, 0.8, 0.7),
                spinewidth = 1.5,
                bottomspinecolor = :black,
                leftspinecolor = :black,
                topspinecolor = :black,
                rightspinecolor = :black,
                topspinevisible = true,
                rightspinevisible = true,
                xtickalign = 1.0,
                ytickalign = 1.0,
                xticksize = 12,
                yticksize = 12,
                xtickwidth = 1.5,
                ytickwidth = 1.5,
                xtickcolor = :black,
                ytickcolor = :black,
                yminorticksvisible = true,
                xminortickalign = 1.0,
                yminortickalign = 1.0,
                xminorticksize = 6,
                yminorticksize = 6,
                xminortickwidth = 1.0,
                yminortickwidth = 1.0,
            ),
            Legend = (
                framevisible = true,
                framewidth = 1.2,
                framecolor = :black,
                backgroundcolor = RGBAf(1.0, 1.0, 1.0, 0.9),
                position = :rt,
                titlesize = fontsize - 2,
                labelsize = fontsize - 4,
                padding = (10.0, 10.0, 10.0, 10.0),
            ),
            Lines = (linewidth = 2.5,),
            Scatter = (markersize = 10,),
            Palette = (
                color = scientific_palette,
                patchcolor = scientific_palette,
            ),
        )
    )
end

"""
    set_publication_style(; cmap = lpca_style, kwargs...)

Alias for `set_publication_theme(; cmap = lpca_style, kwargs...)`.
"""
const set_publication_style = set_publication_theme

const PLOT_KEYS = Dict(
    :T => (L"T\,[\mathrm{fm}^{-1}]", (x, _) -> x[2]),
    :A => (L"\mathcal{A}", (x, _) -> x[3]),
    :B => (L"\mathcal{B}", (x, _) -> size(x, 1) >= 4 ? x[4] : 0.0),
    :tauT => (L"\tau T", (x, _) -> x[1] * x[2]),
    :Tdot => (L"\dot{T}\,[\mathrm{fm}^{-2}]", (x, _) -> (x[2] / x[1]) * (-1 / 3 + x[3] / 18)),
    :tau2Tdot => (L"\tau^2 \dot{T}", (x, _) -> x[1]^2 * ((x[2] / x[1]) * (-1 / 3 + x[3] / 18))),
    :T_norm => (L"T / T_{\mathrm{max}}", (x, slice) -> x[2] / maximum(slice[:, 2])),
    :A_norm => (L"\mathcal{A} / \mathcal{A}_{\mathrm{max}}", (x, slice) -> x[3] / maximum(slice[:, 3])),
    :B_norm => (L"\mathcal{B} / \mathcal{B}_{\mathrm{max}}", (x, slice) -> size(slice, 2) >= 4 ? x[4] / (maximum(abs.(slice[:, 4])) == 0 ? 1.0 : maximum(abs.(slice[:, 4]))) : 0.0),
    :Tdot_norm => (
        L"\dot{T} / \dot{T}_{\mathrm{max}}",
        (x, slice) -> begin
            Tdot_all = (slice[:, 2] ./ slice[:, 1]) .* (-1 / 3 .+ slice[:, 3] ./ 18)
            Tdot_current = (x[2] / x[1]) * (-1 / 3 + x[3] / 18)
            return Tdot_current / maximum(Tdot_all)
        end,
    ),
    :tauT_2020 => (L"\tau_0 T", (x, _) -> 0.2 * x[2]),
    :tau2Tdot_2020 => (L"\tau_0^2 \dot{T}", (x, _) -> 0.2^2 * ((x[2] / x[1]) * (-1 / 3 + x[3] / 18))),
    :w => (L"w = t T(t)", (x, _) -> x[2]),
)

function resolve_def(def)
    if def isa Symbol
        @assert haskey(PLOT_KEYS, def) "Unknown plot key: $def"
        return PLOT_KEYS[def]
    end
    if isa(def, Tuple) && length(def) == 2
        return (def[1], def[2])
    end
    error("Axis definition must be Symbol or Tuple(label, function).")
end

function get_data(dataset::AbstractMatrix{<:Real}, t::Real, xdef, ydef; is_attractor = false)
    xlbl, xfn = resolve_def(xdef)
    ylbl, yfn = resolve_def(ydef)
    if is_attractor
        selected = dataset
    else
        rows = findall(isapprox.(dataset[:, 1], t; atol = 1.0e-8))
        if isempty(rows)
            nearest = argmin(abs.(dataset[:, 1] .- t))
            rows = findall(isapprox.(dataset[:, 1], dataset[nearest, 1]; atol = 1.0e-8))
        end
        selected = dataset[rows, :]
    end
    x = [xfn(selected[i, :], selected) for i in 1:size(selected, 1)]
    y = [yfn(selected[i, :], selected) for i in 1:size(selected, 1)]
    return (x = x, y = y, xlabel = xlbl, ylabel = ylbl)
end

function get_limits(dataset::AbstractMatrix{<:Real}, def; times = nothing, padding = 0.05)
    lbl, fn = resolve_def(def)
    ts = isnothing(times) ? unique(dataset[:, 1]) : times
    vals = Float64[]
    for t in ts
        rows = findall(isapprox.(dataset[:, 1], t; atol = 1.0e-8))
        if isempty(rows)
            nearest = argmin(abs.(dataset[:, 1] .- t))
            rows = findall(isapprox.(dataset[:, 1], dataset[nearest, 1]; atol = 1.0e-8))
        end
        selected = dataset[rows, :]
        for i in 1:size(selected, 1)
            push!(vals, fn(selected[i, :], selected))
        end
    end
    if isempty(vals)
        return (0.0, 1.0)
    end
    val_min, val_max = minimum(vals), maximum(vals)
    dval = val_max - val_min
    if dval ≈ 0.0
        dval = val_min ≈ 0.0 ? 1.0 : abs(val_min) * 0.1
    end
    return (val_min - padding * dval, val_max + padding * dval)
end

function get_limits(dataset::AbstractMatrix{<:Real}, xdef, ydef; times = nothing, padding = 0.05)
    xlbl, xfn = resolve_def(xdef)
    ylbl, yfn = resolve_def(ydef)
    ts = isnothing(times) ? unique(dataset[:, 1]) : times
    x_vals = Float64[]
    y_vals = Float64[]
    for t in ts
        rows = findall(isapprox.(dataset[:, 1], t; atol = 1.0e-8))
        if isempty(rows)
            nearest = argmin(abs.(dataset[:, 1] .- t))
            rows = findall(isapprox.(dataset[:, 1], dataset[nearest, 1]; atol = 1.0e-8))
        end
        selected = dataset[rows, :]
        for i in 1:size(selected, 1)
            row = selected[i, :]
            push!(x_vals, xfn(row, selected))
            push!(y_vals, yfn(row, selected))
        end
    end
    if isempty(x_vals) || isempty(y_vals)
        return (0.0, 1.0, 0.0, 1.0)
    end
    xmin, xmax = minimum(x_vals), maximum(x_vals)
    ymin, ymax = minimum(y_vals), maximum(y_vals)
    dx = xmax - xmin
    dy = ymax - ymin
    if dx ≈ 0.0
        dx = xmin ≈ 0.0 ? 1.0 : abs(xmin) * 0.1
    end
    if dy ≈ 0.0
        dy = ymin ≈ 0.0 ? 1.0 : abs(ymin) * 0.1
    end
    return (
        xmin - padding * dx, xmax + padding * dx,
        ymin - padding * dy, ymax + padding * dy,
    )
end

function _split_trajectories(dataset::AbstractMatrix{<:Real})
    @assert size(dataset, 2) >= 2 "Dataset must have at least columns [tau, feature1]."
    if size(dataset, 1) == 0
        return UnitRange{Int}[]
    end
    starts = Int[1]
    for i in 2:size(dataset, 1)
        if dataset[i, 1] <= dataset[i - 1, 1]
            push!(starts, i)
        end
    end
    ranges = UnitRange{Int}[]
    for k in eachindex(starts)
        s = starts[k]
        e = k < length(starts) ? starts[k + 1] - 1 : size(dataset, 1)
        push!(ranges, s:e)
    end
    return ranges
end

function _extract_column_or_key(data_slice::AbstractMatrix{<:Real}, def::Union{Symbol, Integer, Tuple})
    if def isa Integer
        return L"\text{Kolumna } %$(def)", data_slice[:, def]
    elseif def === :tau || def === :t
        return L"\tau\,[\mathrm{fm}/c]", data_slice[:, 1]
    elseif def isa Symbol && haskey(PLOT_KEYS, def)
        lbl, fn = PLOT_KEYS[def]
        return lbl, [fn(data_slice[i, :], data_slice) for i in 1:size(data_slice, 1)]
    elseif def isa Tuple && length(def) == 2
        lbl = def[1]
        return lbl, [def[2](data_slice[i, :], data_slice) for i in 1:size(data_slice, 1)]
    else
        error("Nieobsługiwany typ definicji osi: $def. Użyj Symbolu, liczby całkowitej lub Tupli.")
    end
end

function _get_dim_color(d::Real, d_max::Real, colormap = lpca_style)
    cs = resolve_palette(colormap)
    t = d_max > 1.0 ? clamp((Float64(d) - 1.0) / (Float64(d_max) - 1.0), 0.0, 1.0) : 0.0
    return ColorSchemes.get(cs, t)
end

function to_2d_local_plots(dataset::AbstractArray{<:Real, 3})
    dataset_2d = reshape(permutedims(dataset, (2, 1, 3)), :, size(dataset, 3))
    valid_rows = .!isnan.(dataset_2d[:, 1])
    return dataset_2d[valid_rows, :]
end

get_data(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = get_data(to_2d_local_plots(dataset), args...; kwargs...)
get_limits(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = get_limits(to_2d_local_plots(dataset), args...; kwargs...)
_split_trajectories(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = _split_trajectories(to_2d_local_plots(dataset), args...; kwargs...)
