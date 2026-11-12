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
    "#0096C7",
    "#0000FF",
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
    elseif def isa Tuple && length(def) == 2
        return (def[1], def[2])
    end
    error("Axis definition must be Symbol or Tuple(label, function).")
end

function _pad_limits(vmin::Real, vmax::Real, padding::Real = 0.05)
    dv = vmax - vmin
    if dv ≈ 0.0
        dv = vmin ≈ 0.0 ? 1.0 : abs(vmin) * 0.1
    end
    return (vmin - padding * dv, vmax + padding * dv)
end

function get_data(dataset::AbstractMatrix{<:Real}, t::Real, xdef, ydef; is_attractor = false)
    xlbl, xfn = resolve_def(xdef)
    ylbl, yfn = resolve_def(ydef)
    selected = if is_attractor
        dataset
    else
        get_tau_slice(dataset, t; atol = 1.0e-8, feature_cols = 1:size(dataset, 2), nearest = true)[2]
    end
    x = [xfn(selected[i, :], selected) for i in 1:size(selected, 1)]
    y = [yfn(selected[i, :], selected) for i in 1:size(selected, 1)]
    return (x = x, y = y, xlabel = xlbl, ylabel = ylbl)
end

function get_limits(dataset::AbstractMatrix{<:Real}, def; times = nothing, padding = 0.05)
    _, fn = resolve_def(def)
    ts = isnothing(times) ? unique(dataset[:, 1]) : times
    vals = Float64[]
    for t in ts
        _, selected = get_tau_slice(dataset, t; atol = 1.0e-8, feature_cols = 1:size(dataset, 2), nearest = true)
        for i in 1:size(selected, 1)
            push!(vals, fn(selected[i, :], selected))
        end
    end
    isempty(vals) && return (0.0, 1.0)
    return _pad_limits(minimum(vals), maximum(vals), padding)
end

function get_limits(dataset::AbstractMatrix{<:Real}, xdef, ydef; times = nothing, padding = 0.05)
    xlims = get_limits(dataset, xdef; times = times, padding = padding)
    ylims = get_limits(dataset, ydef; times = times, padding = padding)
    return (xlims[1], xlims[2], ylims[1], ylims[2])
end

function _split_trajectories(dataset::AbstractMatrix{<:Real})
    @assert size(dataset, 2) >= 2 "Dataset must have at least columns [tau, feature1]."
    size(dataset, 1) == 0 && return UnitRange{Int}[]
    starts = Int[1]
    for i in 2:size(dataset, 1)
        if dataset[i, 1] <= dataset[i - 1, 1]
            push!(starts, i)
        end
    end
    return [starts[k]:(k < length(starts) ? starts[k + 1] - 1 : size(dataset, 1)) for k in eachindex(starts)]
end

function _extract_column_or_key(data_slice::AbstractMatrix{<:Real}, def::Union{Symbol, Integer, Tuple})
    if def isa Integer
        return L"\text{Kolumna } %$(def)", data_slice[:, def]
    elseif def === :tau || def === :t
        return L"\tau\,[\mathrm{fm}/c]", data_slice[:, 1]
    else
        lbl, fn = resolve_def(def)
        return lbl, [fn(data_slice[i, :], data_slice) for i in 1:size(data_slice, 1)]
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
