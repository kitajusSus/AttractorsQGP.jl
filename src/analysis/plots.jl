using ColorSchemes
using GLMakie
import GLMakie: Axis
using LaTeXStrings
# import Colors


"""
    set_publication_theme(; cmap = :devon, n_colors = 10, bg_color = RGBf(0.96, 0.96, 0.96))

`cmap` - np. :davos, :lajolla, :devon, :haline, :phase

"""
function set_publication_theme(;
        cmap = :devon,         # :davos, :lajolla, :devon, :haline, :phase
        n_colors = 25,         #
        bg_color = RGBf(0.98, 0.98, 0.98)
    )

    scientific_palette = Makie.resample_cmap(cmap, n_colors)

    return set_theme!(
        Theme(
            font = "Libertinus Serif",
            fontsize = 24,
            figure_padding = 20,

            Axis = (
                backgroundcolor = bg_color,
                titlesize = 28,
                xlabelsize = 26,
                ylabelsize = 26,
                xticklabelsize = 20,
                yticklabelsize = 20,

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

                # xminorticksvisible = true,
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
                titlesize = 22,
                labelsize = 20,
                padding = (10.0, 10.0, 10.0, 10.0),
            ),

            Lines = (
                linewidth = 2.5,
            ),
            Scatter = (
                markersize = 10,
                # strokewidth = 0.3,
                # strokecolor = :white,
            ),

            Palette = (
                color = scientific_palette,
                patchcolor = scientific_palette,
            ),
        )
    )
end


const PLOT_KEYS = Dict(
    :T => (L"T\,[\mathrm{fm}^{-1}]", (x, _) -> x[2]),
    :A => (L"\mathcal{A}", (x, _) -> x[3]),
    :B => (L"\mathcal{B}", (x, _) -> size(x, 1) >= 4 ? x[4] : 0.0),
    :tauT => (L"\tau T", (x, _) -> x[1] * x[2]),
    :Tdot => (L"\dot{T}\,[\mathrm{fm}^{-2}]", (x, _) -> (x[2] / x[1]) * (-1 / 3 + x[3] / 18)),
    :tau2Tdot => (L"\tau^2 \dot{T}", (x, _) -> x[1]^2 * ((x[2] / x[1]) * (-1 / 3 + x[3] / 18))),
    # testowanie normalizowanie
    :T_norm => (
        L"T / T_{\mathrm{max}}",
        (x, slice) -> x[2] / maximum(slice[:, 2]),
    ),
    :A_norm => (
        L"\mathcal{A} / \mathcal{A}_{\mathrm{max}}",
        (x, slice) -> x[3] / maximum(slice[:, 3]),
    ),
    :B_norm => (
        L"\mathcal{B} / \mathcal{B}_{\mathrm{max}}",
        (x, slice) -> size(slice, 2) >= 4 ? x[4] / (maximum(abs.(slice[:, 4])) == 0 ? 1.0 : maximum(abs.(slice[:, 4]))) : 0.0,
    ),
    # lambda funkcje kongo lekkie
    # Tdot / Tdot_max dla danej chwili czasu
    :Tdot_norm => (
        L"\dot{T} / \dot{T}_{\mathrm{max}}",
        (x, slice) -> begin
            Tdot_all = (slice[:, 2] ./ slice[:, 1]) .* (-1 / 3 .+ slice[:, 3] ./ 18)

            Tdot_current = (x[2] / x[1]) * (-1 / 3 + x[3] / 18)

            return Tdot_current / maximum(Tdot_all)
        end,
    ),
    # pod konkretne publikacje tutaj 2020 Hydrodynamics in Phase Space 0.22 to mój czas
    # inizjalitacji τ₀
    :tauT_2020 => (L"\tau_0 T", (x, _) -> 0.2 * x[2]),
    :tau2Tdot_2020 => (L"\tau_0^2 \dot{T}", (x, _) -> 0.2^2 * ((x[2] / x[1]) * (-1 / 3 + x[3] / 18))),

    :w => (L"w = t T(t)", (x, _) -> x[2]),
)

function resolve_def(def)
    if def isa Symbol
        @assert haskey(PLOT_KEYS, def) "Unknown plot key: $def"
        return PLOT_KEYS[def]
    end

    if isa(def, Tuple)&& length(def) == 2
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


"""
    get_limits(dataset::AbstractMatrix{<:Real}, def; times = nothing, padding = 0.05)

Calculates the padded `(min, max)` range for a single plot key definition `def`.
If `times` is `nothing`, it uses all unique `tau` values in the dataset.
"""
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

"""
    get_limits(dataset::AbstractMatrix{<:Real}, xdef, ydef; times = nothing, padding = 0.05)

Calculates the padded `(xmin, xmax, ymin, ymax)` ranges for both axis definitions `xdef` and `ydef`.
If `times` is `nothing`, it uses all unique `tau` values in the dataset.
"""
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

function plot_phase_space_grid(
        dataset::AbstractMatrix{<:Real},
        times,
        xdef,
        ydef;
        attractor = nothing,
        attractor_points::Int = 150
    )
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]

    n = length(times)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size = (400 * ncols, 350 * nrows))

    lims = get_limits(dataset, xdef, ydef; times = times)

    for (i, t) in enumerate(times)
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1

        d = get_data(dataset, t, xdef, ydef)
        if i > 3

            ax = Axis(
                fig[row, col],
                title = L"\tau = %$(round(t, digits=2))\,\mathrm{fm}/c",
                xlabel = d.xlabel,
                ylabel = d.ylabel,
                limits = lims
            )
            ylims!(ax, 0, 4)
        else
            ax = Axis(
                fig[row, col],
                title = L"\tau = %$(round(t, digits=2))\,\mathrm{fm}/c",
                xlabel = d.xlabel,
                ylabel = d.ylabel,
                limits = lims
            )
        end

        # musi być najpiew by był pod kropkami
        # jeśli jest dany jeśnie nie to nie
        if !isnothing(attractor)
            attr_line = get_attractor_line_for_frame(
                dataset,
                attractor,
                t,
                xdef,
                ydef;
                limits = [0, lims[2], lims[3], lims[4]],
                n_points = attractor_points,
            )
            xlims!(ax, 0, lims[2])
            lines!(ax, attr_line.x, attr_line.y; color = (RGBAf(176 / 255, 0, 0, 1.0), 0.86), linewidth = 5, label = "Atraktor")
        end
        scatter!(
            ax, d.x[1:1000], d.y[1:1000]; markersize = 3.5, color = (palette[2], 0.9),
            strokecolor = (palette[6], 0.01),
            strokewidth = 0.1
        )
    end
    return fig
end
function plot_phase_space_grid_3d(
        dataset::AbstractMatrix{<:Real},
        times,
        xdef,
        ydef,
        zdef;
        limits = nothing,
        attractor = nothing,
        attractor_points::Int = 150,
        azimuth::Real = 1.3,
        elevation::Real = 0.15,
        n_points_scatter::Int = 5000
    )
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]

    xlbl, xfn = resolve_def(xdef)
    ylbl, yfn = resolve_def(ydef)
    zlbl, zfn = resolve_def(zdef)

    n = length(times)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size = (480 * ncols, 420 * nrows), figure_padding = (60, 40, 50, 40))

    for (i, t) in enumerate(times)
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1

        rows = findall(isapprox.(dataset[:, 1], t; atol = 1.0e-8))
        if isempty(rows)
            nearest = argmin(abs.(dataset[:, 1] .- t))
            rows = findall(isapprox.(dataset[:, 1], dataset[nearest, 1]; atol = 1.0e-8))
        end
        selected = dataset[rows, :]

        dx = [xfn(selected[j, :], selected) for j in 1:size(selected, 1)]
        dy = [yfn(selected[j, :], selected) for j in 1:size(selected, 1)]
        dz = [zfn(selected[j, :], selected) for j in 1:size(selected, 1)]

        ax = Axis3(
            fig[row, col],
            title = L"\tau = %$(round(t, digits=2))\,\mathrm{fm}/c",
            xlabel = xlbl,
            ylabel = ylbl,
            zlabel = zlbl,
            azimuth = azimuth,
            elevation = elevation,
            xlabeloffset = 40,
            ylabeloffset = 40,
            zlabeloffset = 55
        )

        if !isnothing(limits)
            limits!(ax, limits...)
        end

        if !isnothing(attractor)
            tau_val = selected[1, 1]
            Tmin, Tmax = minimum(dataset[:, 2]) * 0.95, maximum(dataset[:, 2]) * 1.05
            T_grid = temperature_grid(Tmin, Tmax, attractor_points)
            A_attr = build_attractor_interpolant(attractor)

            attr_x = Vector{Float64}(undef, attractor_points)
            attr_y = Vector{Float64}(undef, attractor_points)
            attr_z = Vector{Float64}(undef, attractor_points)

            ncols_ds = size(dataset, 2)
            state = zeros(Float64, ncols_ds)
            state[1] = tau_val

            for (k, T) in enumerate(T_grid)
                state[2] = T
                state[3] = A_attr(tau_val * T)
                attr_x[k] = xfn(state, selected)
                attr_y[k] = yfn(state, selected)
                attr_z[k] = zfn(state, selected)
            end

            lines!(
                ax, attr_x, attr_y, attr_z;
                color = (RGBAf(176 / 255, 0, 0, 1.0), 0.9),
                linewidth = 4,
                label = "Atraktor"
            )
        end

        n_pts = min(length(dx), n_points_scatter)
        scatter!(
            ax, dx[1:n_pts], dy[1:n_pts], dz[1:n_pts];
            markersize = 10.0,
            color = (palette[10], 0.9),
            strokecolor = (palette[6], 0.01),
            strokewidth = 0.1
        )
    end

    return fig
end

function plot_attractor(
        dataset::AbstractMatrix{<:Real},
        T0_target::Union{Real, AbstractVector{<:Real}, Nothing} = nothing;
        attractor::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        tol::Real = 0.05,
        n_trajectories::Int = 50,
        group_tol::Real = 0.01
    )
    set_publication_theme()
    fig = Figure(size = (950, 600))
    ax = Axis(
        fig[1, 1],
        xlabel = L"w = \tau T",
        ylabel = L" \mathcal{A}",
        limits = (0, 3, -1, 8)
    )

    trajs = _split_trajectories(dataset)

    if T0_target isa Real
        wybrane_trajs = [tr for tr in trajs if abs(dataset[tr[1], 2] - T0_target) < tol]
    elseif T0_target isa AbstractVector
        wybrane_trajs = [tr for tr in trajs if any(abs(dataset[tr[1], 2] - t) < tol for t in T0_target)]
    else
        wybrane_trajs = trajs
    end
    sort!(wybrane_trajs, by = tr -> dataset[tr[1], 2])

    if !isnothing(attractor)
        omega_attr = attractor[:, 1] .* attractor[:, 2]
        A_attr = attractor[:, 3]

        lines!(
            ax, omega_attr, A_attr,
            color = (:red, 1),
            linewidth = 7.0,
            label = L"\text{Teoretyczny atraktor}"
        )
        axislegend(ax, position = :rt)
    end
    step = max(1, length(wybrane_trajs) ÷ n_trajectories)
    selected_trajs = wybrane_trajs[1:step:end]

    T0s = [dataset[tr[1], 2] for tr in selected_trajs]
    min_T0, max_T0 = extrema(T0s)
    if min_T0 == max_T0
        min_T0 -= 0.1
        max_T0 += 0.1
    end

    for tr in selected_trajs
        T0 = dataset[tr[1], 2]
        norm_val = (T0 - min_T0) / (max_T0 - min_T0)
        col = get(ColorSchemes.batlow, norm_val)

        omega_traj = dataset[tr, 1] .* dataset[tr, 2]
        A_traj = dataset[tr, 3]

        lines!(
            ax, omega_traj, A_traj,
            color = (col, 0.5),
            linewidth = 1.5,
        )
    end


    hlines!(ax, [0.0], color = :black, linestyle = :dash, linewidth = 1.5)

    Colorbar(fig[1, 2], colormap = :batlow, limits = (min_T0, max_T0))

    return fig
end

function plot_thermodynamics_evolution(dataset::AbstractMatrix{<:Real})
    set_publication_theme()

    palette = Makie.theme(:Palette).color[]
    trajs = _split_trajectories(dataset)
    trajektorie = trajs[1:10:end]
    fig = Figure(size = (1200, 700))
    ax1 = Axis(
        fig[1, 1],
        title = L"\text{Ewolucja Temperatury } T\,[\mathrm{fm}^{-1}]\; \text{w czasie własnym } \tau",
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"T\,[\mathrm{fm}^{-1}]",
    )

    for tr in trajektorie
        lines!(
            ax1,
            dataset[tr, 1],
            dataset[tr, 2],
            color = (palette[3], 0.2),
            linewidth = 1.5,
        )
    end

    ax2 = Axis(
        fig[2, 1],
        title = L"\text{Ewolucja Anizotropii}\; \mathcal{A(τ)}\; \text{ w czasie własnym}",
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\mathcal{A}",
    )

    for tr in trajektorie
        lines!(
            ax2,
            dataset[tr, 1],
            dataset[tr, 3],
            color = (palette[2], 0.2),
            linewidth = 1.5,
        )
    end

    hlines!(
        ax2,
        [0.0],
        color = :red,
        linestyle = :dash,
        linewidth = 2.0,
        label = L"\mathcal{A}=0\;(\text{Anizotropia} = 0)",
    )
    axislegend(ax2, position = :rt)

    linkxaxes!(ax1, ax2)
    return fig
end


function plot_phase_space_evolution(dataset::AbstractMatrix{<:Real})
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    trajs = _split_trajectories(dataset)
    trajektorie = trajs[1:10:end]
    fig = Figure(size = (950, 620))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Ewolucja w przestrzeni fazowej } (T, \mathcal{A})\; \text{w czasie własnym } \tau",
        xlabel = L"T\,[\mathrm{fm}^{-1}]",

        ylabel = L"\mathcal{A}",
    )
    for (i, tr) in enumerate(trajektorie)
        col = palette[mod1(i, length(palette))]
        lines!(
            ax,
            dataset[tr, 2],
            dataset[tr, 3],
            color = (col, 0.75),
            linewidth = 1.5,
        )
    end
    hlines!(
        ax,
        [0.0],
        color = :black,
        linestyle = :dash,
        linewidth = 2.0,
        label = L"\mathcal{A}=0\;(\text{Anizotropia} = 0)",
    )
    axislegend(ax, position = :rt)
    return fig
end


function plot_attractor_Aw_T(
        dataset::AbstractMatrix{<:Real},
        T_zadana::Real;
        attractor::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        T_tol::Real = 1.0e-3
    )
    set_publication_theme()

    T_col = dataset[:, 2]
    rows = findall(isapprox.(T_col, T_zadana; atol = T_tol))
    if isempty(rows)
        nearest_idx = argmin(abs.(T_col .- T_zadana))
        T_najblizsze = T_col[nearest_idx]
        rows = findall(isapprox.(T_col, T_najblizsze; atol = T_tol))
        println("Brak dokładnego T = $T_zadana w danych. Użyto najbliższego znalezionego: T ≈ $(round(T_najblizsze, digits = 4))")
    end
    slice = dataset[rows, :]

    omega_slice = slice[:, 1] .* slice[:, 2]
    A_slice = slice[:, 3]

    fig = Figure(size = (850, 600))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Atraktor }\mathcal{A}(\omega)\text{ dla } T = %$(round(T_zadana, digits=3))\,\mathrm{fm}^{-1}",
        xlabel = L"\text{Czas uniwersalny } \omega = \tau T",
        ylabel = L"\text{Anizotropia } \mathcal{A} (w)"
    )

    if !isnothing(attractor)
        omega_attr = attractor[:, 1] .* attractor[:, 2]
        A_attr = attractor[:, 3]
        lines!(
            ax, omega_attr, A_attr,
            color = :red,
            linewidth = 3.0,
            label = L"\text{Teoretyczny atraktor}"
        )
    end

    # Rysujemy chmurę punktów z symulacji
    palette = Makie.theme(:Palette).color[]
    scatter!(
        ax, omega_slice, A_slice,
        markersize = 8,
        color = (palette[1], 0.6),
        strokecolor = :white,
        strokewidth = 0.5,
        label = L"\text{Dane symulacji}"
    )

    axislegend(ax, position = :rb)

    return fig
end
#
#
# function plot_phase_space_evolution_3d(dataset::AbstractMatrix{<:Real})
#     set_publication_theme()
#     palette = Makie.theme(:Palette).color[]
#     trajs = _split_trajectories(dataset)
#     trajektorie = trajs[1:10:end]
#     fig = Figure(size = (950, 750))
#     ax = Axis3(
#         fig[1, 1],
#         title = L"\text{Ewolucja w przestrzeni fazowej } (T, \mathcal{A}, \tau)",
#         xlabel = L"T\,[\mathrm{fm}^{-1}]",
#         ylabel = L"\mathcal{A}",
#         zlabel = L"\tau\,[\mathrm{fm}]",
#         azimuth = 1.3π,
#         elevation = 0.15π,
#     )
#     for (i, tr) in enumerate(trajektorie)
#         col = palette[mod1(i, length(palette))]
#         lines!(
#             ax,
#             dataset[tr, 2],
#             dataset[tr, 3],
#             dataset[tr, 1],
#             color = (col, 0.7),
#             linewidth = 1.5,
#         )
#     end
#     T_range = range(
#         minimum(dataset[:, 2]),
#         maximum(dataset[:, 2]),
#         length = 2,
#     )
#     τ_range = range(
#         minimum(dataset[:, 1]),
#         maximum(dataset[:, 1]),
#         length = 2,
#     )
#     surface!(
#         ax,
#         T_range,
#         zeros(2, 2),         # A=0
#         repeat(τ_range, 1, 2)',
#         color = fill((:gray, 0.15), 2, 2),
#         transparency = true,
#     )
#     return fig
# end
#
#
function plot_pca_evr_over_time(
        dataset::AbstractMatrix{<:Real};
        n_components::Int = 2,
        method::Symbol = :minmax,
        gamma::Float64 = 1.0,
        feature_cols::AbstractVector = collect(2:size(dataset, 2)),
        plot_title::Union{String, LaTeXString} = L"\text{Explained Variance Ratio (EVR) w funkcji czasu}",
        x_label::Union{String, LaTeXString} = L"\tau\,[\mathrm{fm}/c]",
        y_label::Union{String, LaTeXString} = L"\text{EVR}",
        tau_min::Union{Real, Nothing} = nothing
    )
    set_publication_theme()

    result = run_pca_per_time(
        dataset;
        n_components = n_components,
        method = method,
        gamma = gamma,
        feature_cols = feature_cols,
    )
    taus = result.taus
    evr = result.explained_variance_ratio

    t_min = isnothing(tau_min) ? minimum(taus) : Float64(tau_min)

    fig = Figure(size = (850, 600))
    ax = Axis(
        fig[1, 1],
        title = plot_title,
        xlabel = x_label,
        ylabel = y_label,

        limits = (t_min, maximum(taus), 0, 1.05),
    )

    hlines!(ax, [1.0], color = :gray45, linestyle = :dash, label = L"100\%")

    palette = Makie.theme(:Palette).color[]
    kolory = [palette[3], palette[7], palette[3], palette[7], palette[2], palette[8]]
    for comp in 1:n_components
        vals = evr[:, comp]
        mask = .!isnan.(vals)
        if any(mask)
            lines!(
                ax,
                taus[mask],
                vals[mask],
                linewidth = 3.0,
                color = kolory[mod1(comp, length(kolory))],
                label = L"\text{PC}%$(comp)",
            )
            # if comp == 1
            #     band!(
            #         ax,
            #         taus[mask],
            #         zeros(sum(mask)),
            #         vals[mask],
            #         color = (palette[1], 0.15),
            #     )
            # end
        end
    end

    axislegend(ax, position = :rb)
    return fig
end

function plot_pca_bar_variance(
        dataset::AbstractMatrix{<:Real};
        tau::Real = 1.14,
        method::Symbol = :minmax,
        gamma::Float64 = 1.0,
        feature_cols::Union{AbstractVector{<:Integer}, Nothing} = nothing
    )
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]

    cols = isnothing(feature_cols) ? (2:size(dataset, 2)) : feature_cols
    _, Xtau = get_tau_slice(dataset, tau; atol = 1.0e-8, feature_cols = cols)

    pca_result = if method === :minmax
        run_pca(Xtau; n_components = 2)
    elseif method === :kernel
        run_pca_kernel(Xtau; n_components = 2, gamma = gamma)
    else
        error("Nieznana metoda PCA. Wybierz :minmax lub :kernel.")
    end

    evr = pca_result.explained_variance_ratio_full
    n_comp = length(evr)
    cumulative_evr = cumsum(evr)

    fig = Figure(size = (850, 550))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Analiza Wariancji — — PCA z normalizacją min-max}",
        xlabel = L"\text{Główna Składowa}",
        ylabel = L"\text{EVR [\%]}",
        xticks = (1:n_comp, [L"\mathrm{PC}_%$(i)" for i in 1:n_comp]),
        limits = (0.4, n_comp + 0.6, 0, 110)
    )

    evr_percent = evr .* 100
    barplot!(ax, 1:n_comp, evr_percent, color = palette[1], width = 0.5)

    for i in 1:n_comp
        text!(
            ax,
            i,
            evr_percent[i] + 3,
            text = string(round(evr_percent[i], digits = 1), "%"),
            align = (:center, :bottom),
            fontsize = 18,
            font = "Libertinus Serif"
        )
    end

    cum_percent = cumulative_evr .* 100
    lines!(ax, 1:n_comp, cum_percent, color = :gray35, linestyle = :dash, linewidth = 2.5)
    scatter!(ax, 1:n_comp, cum_percent, color = palette[2], markersize = 12, label = L"\text{Suma EVR}")

    axislegend(ax, position = :rc)

    return fig
end

function plot_pca_summary(
        dataset::AbstractMatrix{<:Real};
        tau::Union{Nothing, Real} = nothing,
        tau_tol::Float64 = 1.0e-8,
        tau_mode::Symbol = :nearest,
        n_components::Int = 2,
        method::Symbol = :minmax,
        gamma::Float64 = 1.0,
    )

    set_publication_theme()

    palette = Makie.theme(:Palette).color[]
    data = Matrix{Float64}(dataset)
    subtitle = L"\text{Wszystkie } \tau"

    if tau !== nothing
        τ = Float64(tau)
        τcol = data[:, 1]
        d = abs.(τcol .- τ)
        strict_mask = d .<= tau_tol

        if any(strict_mask)
            data = data[strict_mask, :]
            subtitle = L"\tau=%$(τ) \pm %$(tau_tol)"
        else
            if tau_mode === :strict
                error("No rows found for tau=$(τ) within tau_tol=$(tau_tol).")
            else
                i = argmin(d)
                τnearest = τcol[i]
                near_mask = τcol .== τnearest
                data = data[near_mask, :]
                subtitle = L"\text{najbliższe } \tau=%$(τnearest)"
            end
        end
    end

    features = data[:, 2:3]
    @assert size(features, 1) > 1 "Need at least two samples in selected tau slice."

    pca_result = if method === :minmax
        run_pca(features; n_components = n_components)
    elseif method === :kernel
        run_pca_kernel(features; n_components = n_components, gamma = gamma)
    else
        error("Unknown PCA method. Choose :minmax or :kernel.")
    end

    transformed = pca_result.transformed
    evr = pca_result.explained_variance_ratio
    n_show = min(size(transformed, 2), 2)

    fig = Figure(size = (980, 420))

    ax_proj = Axis(
        fig[1, 1],
        xlabel = L"\text{PC1}",
        ylabel = L"\text{PC2}",
        title = L"\text{Projekcja PCA } (%$subtitle)",
    )
    if n_show >= 2
        scatter!(
            ax_proj,
            transformed[:, 1],
            transformed[:, 2];
            markersize = 4.5,
            color = (palette[1], 0.75),
        )
    elseif n_show == 1
        scatter!(
            ax_proj,
            transformed[:, 1],
            zeros(size(transformed, 1));
            markersize = 4.5,
            color = (palette[1], 0.75),
        )
    end

    ax_evr = Axis(
        fig[1, 2],
        xlabel = L"\text{Główna składowa}",
        ylabel = L"\text{EVR}",
        limits = (0.5, max(length(evr), 1) + 0.5, 0, 1),
        title = L"\text{Współczynnik wariancji wyjaśnionej}",
    )
    if !isempty(evr)
        barplot!(ax_evr, 1:length(evr), evr; color = palette[2])
    end

    return fig
end
function animate_pca_evolution(
        dataset::AbstractMatrix{<:Real};
        filename::String = "pca_evolution.gif",
        fps::Int = 15,
        n_components::Int = 2,
        method::Symbol = :minmax,
        gamma::Float64 = 1.0,
        tau_tol::Float64 = 1.0e-8
    )
    set_publication_theme()

    palette = Makie.theme(:Palette).color[]
    taus = sort(unique(dataset[:, 1]))

    fig = Figure(size = (800, 600))

    title_obs = Observable(L"\text{Projekcja PCA } (\tau = %$(taus[1]))")
    ax = Axis(
        fig[1, 1],
        xlabel = L"\text{PC1}",
        ylabel = L"\text{PC2}",
        title = title_obs
    )

    pts_obs = Observable(Point2f[])

    scatter!(
        ax,
        pts_obs;
        markersize = 6.0,
        color = (palette[1], 0.75)
    )

    record(fig, filename, taus; framerate = fps) do t
        val = round(t, digits = 3)
        title_obs[] = L"\text{Projekcja PCA } (\tau = %$(val))"

        d = abs.(dataset[:, 1] .- t)
        mask = d .<= tau_tol
        data_slice = dataset[mask, :]

        features = data_slice[:, 2:3]

        if size(features, 1) > 1
            pca_result = if method === :minmax
                run_pca(features; n_components = n_components)
            elseif method === :kernel
                run_pca_kernel(features; n_components = n_components, gamma = gamma)
            else
                error("Unknown PCA method. Choose :minmax or :kernel.")
            end

            transformed = pca_result.transformed

            if size(transformed, 2) >= 2
                pts_obs[] = [Point2f(transformed[i, 1], transformed[i, 2]) for i in 1:size(transformed, 1)]
            else
                pts_obs[] = [Point2f(transformed[i, 1], 0.0) for i in 1:size(transformed, 1)]
            end

            reset_limits!(ax)
        end
    end

    return filename
end
###########################
### FUNKCJE DO LLE
###############################

"""
Analiza spektrum wartości własnych dla danego czasu 
    
    function plot_lle_spectrum_statistics(dataset; τ, k_values = 5:5:50, ile_λ = 4)

- `ile_λ` - ile pierwszych wartości własnych pokazać na wykresie (domyślnie 4) 

ale trzeba pamiętać że pierwsza wartość własna jest (i powinna być) zawsze λ₁ = 0

"""
function plot_lle_spectrum_statistics(dataset; τ, k_values = 5:5:50, ile_λ = 4)
    k_values = flatten_k_values(k_values)
    set_publication_theme()

    palette = Makie.theme(:Palette).color[]
    spectra, ks = lle_spectrum_over_k(dataset; tau = τ, k_values = k_values)
    μ, σ = spectrum_statistics(spectra)

    x = 1:ile_λ
    k_min = minimum(k_values)
    k_max = maximum(k_values)

    fig = Figure(size = (1400, 700))
    ax = Axis(
        fig[1, 1],
        title = L"\text{LLE analiza wartości własnych dla K z zakresu (%$k_min - %$k_max) (średnia } \pm \sigma \text{) } \tau=%$τ",
        xlabel = L"\text{Indeks wartości własnej } \lambda_{i}",
        ylabel = L"\text{Wartość } \lambda_{i}"
    )

    scatterlines!(
        ax, x, μ[1:ile_λ],
        linewidth = 2,
        color = (palette[5], 0.3),
        markercolor = palette[5],
        markersize = 15,
        label = L"\text{Średnia } i\text{-ta } \lambda_{i}"
    )

    band!(
        ax,
        x,
        μ[1:ile_λ] .- σ[1:ile_λ],
        μ[1:ile_λ] .+ σ[1:ile_λ],
        color = (palette[3], 0.2)
    )

    axislegend(ax, position = :lt)

    return fig
end


"""
    plot_lle_spectrum_statistics_grid(dataset, taus; k_values = 5:5:50, ile_λ = 4, atol = 1.0e-8)
"""
function plot_lle_spectrum_statistics_grid(dataset, taus; k_values = 5:5:50, ile_λ = 4, atol = 1.0e-8)
    k_values = flatten_k_values(k_values)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]

    n = length(taus)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    spectra1, ks1 = lle_spectrum_over_k(dataset; tau = taus[1], k_values = k_values, atol = atol)
    μ1, σ1 = spectrum_statistics(spectra1)

    y_min = minimum(μ1[1:ile_λ] .- σ1[1:ile_λ])
    y_max = maximum(μ1[1:ile_λ] .+ σ1[1:ile_λ])
    y_pad = (y_max - y_min) * 0.07
    lims = (0.9, ile_λ + 0.05, y_min - y_pad, y_max + y_pad)

    fig = Figure(size = (450 * ncols + 40, 350 * nrows + 120))

    k_min = minimum(k_values)
    k_max = maximum(k_values)
    title_text = L"\text{LLE analiza wartości własnych dla K z zakresu (%$k_min - %$k_max) (średnia } \pm \sigma \text{)}"
    Label(fig[1, 2:(ncols + 1)], title_text, fontsize = 36, font = :bold, padding = (0, 0, 10, 0))

    Label(fig[2:(nrows + 1), 1], L"\text{Wartość } \lambda_{i}", rotation = pi / 2, font = :bold, fontsize = 32)
    Label(fig[nrows + 2, 2:(ncols + 1)], L"\text{Indeks wartości własnej } \lambda_{i}", font = :bold, fontsize = 32)

    x = 1:ile_λ

    for (i, tau) in enumerate(taus)
        row = (i - 1) ÷ ncols + 2
        col = (i - 1) % ncols + 2
        if i == 1
            μ, σ = μ1, σ1
        else
            spectra, ks = lle_spectrum_over_k(dataset; tau = tau, k_values = k_values, atol = atol)
            μ, σ = spectrum_statistics(spectra)
        end

        ax = Axis(
            fig[row, col],
            title = L"\tau = %$tau\,\mathrm{fm}/c",
            limits = lims,
            xticks = 1:ile_λ
        )

        scatterlines!(
            ax, x, μ[1:ile_λ],
            linewidth = 2,
            color = (palette[5], 0.3),
            markercolor = palette[5],
            markersize = 15,
            label = L"\text{Średnia } i\text{-ta } \lambda_{i}"
        )

        band!(
            ax,
            x,
            μ[1:ile_λ] .- σ[1:ile_λ],
            μ[1:ile_λ] .+ σ[1:ile_λ],
            color = (palette[3], 0.2)
        )

        axislegend(ax, position = :lt)

        row_idx = (i - 1) ÷ ncols + 1
        col_idx = (i - 1) % ncols + 1
        is_bottom = (row_idx == nrows) || (i + ncols > n)
        is_left = (col_idx == 1)

        if !is_bottom
            hidexdecorations!(ax, grid = false)
        end
        if !is_left
            hideydecorations!(ax, grid = false)
        end
        println("Wyliczone wartości własne $tau: ", μ[1:ile_λ])
    end

    return fig
end

@views function plot_lle_σ_spectrum(dataset, taus; k_values = 5:5:50, ile_λ = 3, atol = 1.0e-8)
    k_values = flatten_k_values(k_values)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    n_taus = length(taus)
    sigma_matrix = fill(NaN, n_taus, ile_λ)

    for (i, tau) in enumerate(taus)
        spectra, ks = lle_spectrum_over_k(dataset; tau = tau, k_values = k_values, atol = atol)

        μ, σ = spectrum_statistics(spectra)
        N = min(ile_λ, length(σ))
        sigma_matrix[i, 1:N] .= σ[1:N] .* 10^(8)

    end

    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Zmiana } \sigma_i \text{ w funkcji } \tau \text{ dla } i \le %$ile_λ",
        xlabel = L"\tau \ [\mathrm{fm/c}]",
        ylabel = L"\sigma_i \cdot 10^{8}",

        xticks = 0.2:0.05:maximum(taus)
    )
    for j in 1:ile_λ
        xs = taus
        ys = sigma_matrix[:, j]
        lines!(ax, xs, ys, linewidth = 3, color = palette[(j + 1) * 3], label = L"\sigma_{%$j}")
        scatter!(ax, xs, ys, color = palette[j + 3], markersize = 5)
    end

    axislegend(ax, position = :rt)

    return fig
end

function plot_lle_dim(dataset::AbstractMatrix{<:Real}, k::Int, d::Int, tau::Real)
    set_publication_theme()

    palette = Makie.theme(:Palette, :color)[]
    lle_data = run_lle_per_time(dataset; k = k, d = d)
    embedding = lle_data.lle_results[tau]

    fig = Figure(size = (600, 500))
    ax = Axis(
        fig[1, 1],
        title = L"\text{LLE: } k=%$k, d=%$d, \tau=%$tau"
    )

    if d == 1
        ax.xlabel = L"\text{LLE1}"
        ax.ylabel = L"\text{Wartość stała}"
        scatter!(
            ax,
            embedding[:, 1],
            zeros(size(embedding, 1));
            markersize = 4.5,
            color = (palette[1], 0.75),
        )
    else
        ax.xlabel = L"\text{LLE1}"
        ax.ylabel = L"\text{LLE2}"
        scatter!(
            ax,
            embedding[:, 1],
            embedding[:, 2];
            markersize = 4.5,
            color = (palette[1], 0.75),
        )
    end

    return fig
end

function plot_lle_dim!(ax::Axis, dataset::AbstractMatrix{<:Real}, k::Int, d::Int, tau::Real)
    set_publication_theme()

    palette = Makie.theme(:Palette).color[]
    lle_data = run_lle_per_time(dataset; k = k, d = d)
    embedding = lle_data.lle_results[tau]

    ax.title = L"\text{LLE: } k=%$k, d=%$d, \tau=%$tau"

    if d == 1
        ax.xlabel = L"\text{Odwzorowanie 1}"
        ax.ylabel = L"\text{Wartość stała}"
        scatter!(
            ax,
            embedding[:, 1],
            zeros(size(embedding, 1));
            markersize = 4.5,
            color = (palette[1], 0.75),
        )
    else
        ax.xlabel = L"\text{Odwzorowanie LLE1}"
        ax.ylabel = L"\text{Odwzorowanie LLE2}"
        scatter!(
            ax,
            embedding[:, 1],
            embedding[:, 2];
            markersize = 4.5,
            color = (palette[1], 0.75),
        )
    end
    return nothing
end

function plot_simulation_lle(dataset::AbstractMatrix{<:Real}, k::Int, d::Int, tau_zakres)
    set_publication_theme()

    palette = Makie.theme(:Palette).color[]
    liczba_wykresow = length(tau_zakres)
    kolumny = 2
    wiersze = ceil(Int, liczba_wykresow / kolumny)

    bok_kwadratu = max(kolumny, wiersze)

    fig = Figure(size = (bok_kwadratu * 400, bok_kwadratu * 400))

    for (i, tau) in enumerate(tau_zakres)
        row = (i - 1) ÷ kolumny + 1
        col = (i - 1) % kolumny + 1

        ax = Axis(fig[row, col], aspect = 1)
        plot_lle_dim!(ax, dataset, k, d, tau)
    end

    return fig
end

function plot_lle_embedding(dataset::AbstractMatrix{<:Real}; labels = nothing, title = L"\text{Projekcja LLE}", osie = nothing)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    dim = size(dataset, 2)

    fig = Figure(size = (750, 600))

    color_param = isnothing(labels) ? (palette[1], 0.75) : labels
    cmap_param = isnothing(labels) ? :viridis : :jet

    if dim >= 3
        ax = Axis3(fig[1, 1], title = title)
        #xlabel = L"\text{LLE1}", ylabel = L"\text{LLE2}", zlabel = L"\text{LLE3}")
        scatter!(ax, dataset[:, 1], dataset[:, 2], dataset[:, 3], color = color_param, colormap = cmap_param, markersize = 6)
    elseif dim == 2
        ax = Axis(fig[1, 1], title = title, xlabel = L"\text{LLE1}", ylabel = L"\text{LLE2}")
        scatter!(ax, dataset[:, 1], dataset[:, 2], color = color_param, colormap = cmap_param, markersize = 6)
    else
        ax = Axis(fig[1, 1], title = title, xlabel = L"\text{LLE1}")
        scatter!(ax, dataset[:, 1], zeros(size(dataset, 1)), color = color_param, colormap = cmap_param, markersize = 6)
    end

    return fig
end

function plot_lle_grid(lle_results::Dict, taus; labels = nothing)
    dim = size(first(values(lle_results)), 2)
    if dim >= 3
        return plot_lle_grid_3d(lle_results, taus; labels = labels)
    else
        return plot_lle_grid_2d(lle_results, taus; labels = labels)
    end
end

function plot_lle_grid_2d(lle_results::Dict, taus; labels = nothing)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    color_param = isnothing(labels) ? (palette[1], 0.75) : labels
    cmap_param = isnothing(labels) ? :viridis : :jet

    n = length(taus)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size = (400 * ncols, 380 * nrows))

    for (i, t) in enumerate(taus)
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1
        embedding = lle_results[t]

        ax = Axis(fig[row, col], title = L"\tau = %$(round(t, digits=2))", xlabel = L"\text{LLE1}", ylabel = L"\text{LLE2}")
        scatter!(ax, embedding[:, 1], embedding[:, 2], color = color_param, colormap = cmap_param, markersize = 5)
    end

    return fig
end

function plot_lle_grid_3d(lle_results::Dict, taus; labels = nothing)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    color_param = isnothing(labels) ? (palette[1], 0.75) : labels
    cmap_param = isnothing(labels) ? :viridis : :jet

    n = length(taus)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size = (400 * ncols, 380 * nrows))

    for (i, t) in enumerate(taus)
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1
        embedding = lle_results[t]

        ax = Axis3(fig[row, col], title = L"\tau = %$(round(t, digits=2))", xlabel = L"\text{LLE1}", ylabel = L"\text{LLE2}", zlabel = L"\text{LLE3}")
        scatter!(ax, embedding[:, 1], embedding[:, 2], embedding[:, 3], color = color_param, colormap = cmap_param, markersize = 5)
    end

    return fig
end


## funkcje do PINN
function plot_pinn_deff_evolution(results::AbstractMatrix{<:Real})
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    fig = Figure(size = (850, 600))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Ewolucja wymiaru efektywnego } d_{\mathrm{eff}} \text{ w czasie } \tau",
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"d_{\mathrm{eff}}",
        limits = (minimum(results[:, 1]), maximum(results[:, 1]), 0.95, maximum(results[:, 2]) * 1.05)
    )
    lines!(ax, results[:, 1], results[:, 2], color = palette[1], linewidth = 3.0)
    return fig
end


#############
## FUNKCJE TO NN
# ###
#
#
#
#
#


#########################
# funkcje do lid mle twonn, itd
# #########################
#
#
#


function plot_lid_dimension(
        dataset::AbstractMatrix{<:Real},
        #k::Int
    )
    set_publication_theme()

    palette = Makie.theme(:Palette).color[]
    fig = Figure(size = (1200, 700))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Ewolucja estymowanego wymiaru LID w funkcji czasu dla różnych } k",
        xlabel = L"\text{Czas } \tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{estymowany wymiar } d_{\mathrm{LID}}",
    )
    k_range = 1:10:100
    for (idx, current_k) in enumerate(k_range)
        taus, lid, _, _ = scan_intrinsic_dimensions(dataset, k = current_k)

        lines!(
            ax, taus, lid,
            color = palette[idx],
            linewidth = 2.5,
            label = latexstring("k = ", current_k)
        )

        # println("τ = $(taus[idx]), LID = $(lid[idx])")
        # println("Obliczony LID dla k = $current_k: ", lid)
    end
    return fig
end


function plot_twonn(dataset::AbstractMatrix{<:Real})
    set_publication_theme()

    palette = Makie.theme(:Palette).color[]
    fig = Figure(size = (1200, 700))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Przykładowy wykres dla TWONN - Two Nearest Neighbors}",
        xlabel = L"\text{Czas } \tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{estymowany wymiar } d_{\mathrm{TWONN}}",
    )
    taus, _, twonn, _ = scan_intrinsic_dimensions(dataset)
    lines!(ax, taus, twonn, color = palette[3], linewidth = 2.5, label = L"\text{TWONN}")
    axislegend(ax, position = :rt)
    return fig
end

function plot_lle_results_for_taus(
        dataset::AbstractMatrix{<:Real},
        target_taus::Vector{Float64};
        k::Integer = 20,
        d::Integer = 2,
        atol::Real = 1.0e-3,
        feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2))
    )

    set_publication_theme()
    palette = Makie.theme(:Palette).color[]

    results = run_lle_for_selected_taus(dataset, target_taus; k = k, d = d, atol = atol, feature_cols = feature_cols)
    lle_res = results.lle_results

    n = length(target_taus)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)

    fig_width = max(1400, 450 * ncols)
    fig_height = max(700, 450 * nrows)
    fig = Figure(size = (fig_width, fig_height))

    for (i, t) in enumerate(target_taus)
        if !haskey(lle_res, t)
            continue
        end
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1

        embedding = lle_res[t]
        dim = size(embedding, 2)

        ax_title = L"\tau = %$(round(t, digits=2))"
        if dim >= 3
            ax = Axis3(fig[row, col], title = ax_title, xlabel = L"\text{LLE1}", ylabel = L"\text{LLE2}", zlabel = L"\text{LLE3}")
            scatter!(ax, embedding[:, 1], embedding[:, 2], embedding[:, 3], color = (palette[1], 0.75), markersize = 6)
        elseif dim == 2
            ax = Axis(fig[row, col], title = ax_title, xlabel = L"\text{LLE1}", ylabel = L"\text{LLE2}")
            scatter!(ax, embedding[:, 1], embedding[:, 2], color = (palette[1], 0.75), markersize = 6)
        else
            ax = Axis(fig[row, col], title = ax_title, xlabel = L"\text{LLE1}")
            scatter!(ax, embedding[:, 1], zeros(size(embedding, 1)), color = (palette[1], 0.75), markersize = 6)
        end
    end

    return fig
end

function plot_lle_spectrum_analysis(
        dataset::AbstractMatrix{<:Real},
        tau::Real;
        k_values = 5:5:50,
        atol::Real = 1.0e-8
    )

    k_values = flatten_k_values(k_values)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]

    spectra, ks = lle_spectrum_over_k(dataset; tau = tau, k_values = k_values, atol = atol)

    fig = Figure(size = (1400, 700))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Widmo LLE dla różnych } k \text{, } \tau = %$(tau)",
        xlabel = L"\text{Indeks wartości własnej}",
        ylabel = L"\lambda"
    )

    for (i, (s, k)) in enumerate(zip(spectra, ks))
        col = palette[mod1(i, length(palette))]
        lines!(ax, 1:length(s), s, label = L"k = %$k", linewidth = 2.5, color = col)
    end

    axislegend(ax, position = :lt, nbanks = 2)
    return fig
end

function plot_lle_spectrum_scan_analysis(
        dataset::AbstractMatrix{<:Real},
        taus::Vector{Float64};
        k_values = 5:5:50
    )

    set_publication_theme()
    palette = Makie.theme(:Palette).color[]

    results = scan_lle_spectrum(dataset; taus = taus, k_values = k_values)

    fig = Figure(size = (1400, 700))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Statystyki widma LLE w czasie}",
        xlabel = L"\text{Indeks wartości własnej}",
        ylabel = L"\lambda"
    )

    for (i, t) in enumerate(taus)
        if !haskey(results, t)
            continue
        end
        res = results[t]
        μ = res.mean
        σ = res.std
        x = 1:length(μ)

        col = palette[mod1(i, length(palette))]
        lines!(ax, x, μ, label = L"\tau = %$(t)", linewidth = 3, color = col)
        band!(ax, x, μ .- σ, μ .+ σ, color = (col, 0.2))
    end

    axislegend(ax, position = :rt, nbanks = 2)
    return fig
end


@views function plot_local_pca(
        dataset_loaded::AbstractMatrix{<:Real};
        n_slices::Int = 15,
        tablica_k::Vector{Int} = [10, 20, 40, 80, 160],
        feature_cols::AbstractVector{<:Integer} = 2:size(dataset_loaded, 2),
        normalize::Union{Symbol, Function} = :max,
        title::Union{String, Nothing} = nothing
    )
    set_publication_theme()

    fig = Figure(size = (950, 600))
    ax_title = title !== nothing ? title : ""
    ax = Axis(
        fig[1, 1],
        title = ax_title,
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Średni wymiar lokalny}",
        xautolimitmargin = (0.0, 0.05),
        yautolimitmargin = (0.05, 0.05)
    )

    palette = [:crimson, :dodgerblue, :forestgreen, :darkorange, :purple, :goldenrod, :darkcyan, :mediumvioletred]
    for (i, k_bazowe) in enumerate(tablica_k)

        k_drugie = k_bazowe .* 2
        tau_vals, mean_k1, std_k1 = compute_lpca(dataset_loaded, k_bazowe, n_slices; feature_cols = feature_cols, normalize = normalize)

        tau_drugie, mean_k2, std_k2 = compute_lpca(dataset_loaded, k_drugie, n_slices; feature_cols = feature_cols, normalize = normalize)
        c = palette[mod1(i, length(palette))]
        band!(
            ax,
            tau_vals,
            mean_k1,
            mean_k2;
            color = (c, 0.25)
        )

        lines!(
            ax,
            tau_vals,
            mean_k1;
            linewidth = 2.5,
            color = c,
            label = L"K = %$(k_bazowe), %$(k_drugie)"
        )
    end

    axislegend(ax, position = :rt)

    return fig
end

"""
    plot_local_pca_regularizations(
        dataset_loaded::AbstractMatrix{<:Real};
        n_slices::Int = 15,
        tablica_k::Vector{Int} = [10, 20, 40, 80],
        feature_cols::AbstractVector{<:Integer} = 2:size(dataset_loaded, 2),
        methods::Vector{Symbol} = [:max, :minmax, :zscore, :none]
    )
"""
@views function plot_local_pca_regularizations(
        dataset_loaded::AbstractMatrix{<:Real};
        n_slices::Int = 15,
        tablica_k::Vector{Int} = [10, 20, 40, 80],
        feature_cols::AbstractVector{<:Integer} = 2:size(dataset_loaded, 2),
        methods::Vector{Symbol} = [:max, :minmax, :zscore, :none]
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

        ax = Axis(
            fig[row, col],
            title = get(method_titles, m, string(m)),
            xlabel = L"\tau\,[\mathrm{fm}/c]",
            ylabel = L"\text{Średni wymiar lokalny}",
            xautolimitmargin = (0.0, 0.05),
            yautolimitmargin = (0.05, 0.05)
        )

        for (i, k_bazowe) in enumerate(tablica_k)
            k_drugie = k_bazowe * 2
            tau_vals, mean_k1, _ = compute_lpca(dataset_loaded, k_bazowe, n_slices; feature_cols = feature_cols, normalize = m)
            _, mean_k2, _ = compute_lpca(dataset_loaded, k_drugie, n_slices; feature_cols = feature_cols, normalize = m)

            c = palette[mod1(i, length(palette))]
            band!(ax, tau_vals, mean_k1, mean_k2; color = (c, 0.25))
            lines!(ax, tau_vals, mean_k1; linewidth = 2.0, color = c, label = L"K = %$(k_bazowe), %$(k_drugie)")
        end

        if row == 1 && col == n_cols
            axislegend(ax, position = :rt)
        end
    end

    return fig
end


function plot_map_lpca(
        dataset::AbstractMatrix{<:Real};
        zakres_K::AbstractVector{<:Integer} = [10, 20, 40, 80],
        n_slices::Int = 15,
        feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
    )
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]

    results = [compute_lpca(dataset, k, n_slices; feature_cols = feature_cols) for k in zakres_K]
    tau_vals = results[1][1]
    dim_matrix = reduce(hcat, [r[2] for r in results])

    fig = Figure(size = (1200, 600))
    ax = Axis(
        fig[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"K",
    )

    hm = heatmap!(
        ax, tau_vals, zakres_K, dim_matrix;
        colormap = palette,
        colorrange = (1, 3)
    )

    Colorbar(
        fig[1, 2], hm;
        label = L"\text{Local Dimension} d",
        ticks = [1, 2, 3]
    )

    return fig
end


function _extract_column_or_key(data_slice::AbstractMatrix{<:Real}, def::Union{Symbol, Integer, Tuple})
    if def isa Integer
        lbl = L"\text{Kolumna } %$(def)"
        return lbl, data_slice[:, def]
    elseif def === :tau || def === :t
        lbl = L"\tau\,[\mathrm{fm}/c]"
        return lbl, data_slice[:, 1]
    elseif def isa Symbol && haskey(PLOT_KEYS, def)
        lbl, fn = PLOT_KEYS[def]
        vals = [fn(data_slice[i, :], data_slice) for i in 1:size(data_slice, 1)]
        return lbl, vals
    elseif def isa Tuple && length(def) == 2
        lbl = def[1]
        vals = [def[2](data_slice[i, :], data_slice) for i in 1:size(data_slice, 1)]
        return lbl, vals
    else
        error("Nieobsługiwany typ definicji osi: $def. Użyj Symbolu (np. :T, :A, :B, :tau), liczby całkowitej lub Tupli.")
    end
end

"""
    plot_phase_space_lpca_dims(
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

Tworzy wykres przestrzeni fazowej (2D lub 3D), w którym punkty są kolorowane według ich
lokalnego wymiaru introwertycznego \$d_{\\text{lokalny}}\$ wykrytego przez algorytm Local PCA (LPCA).

# Parametry:
- `dataset`: Macierz danych [tau, T, A, B, ...]
- `liczba_sąsiadów`: Liczba najbliższych sąsiadów \$K\$ dla algorytmu k-NN w LPCA (domyślnie 20)
- `wybrane_czasy`: Moment(y) czasu \$\\tau\$ do analizy. Jeśli `nothing`, analizowane są wszystkie chwile czasu w danych.
- `tolerancja_czasowa`: Tolerancja dopasowania czasu \$\\tau\$ (domyślnie 1e-3)
- `tolerancja_wartości_własnych`: Próg ucinki wartości własnych \$\\lambda_i / \\sum \\lambda\$ w LPCA (domyślnie 0.01)
- `metoda_regularyzacji`: Metoda normalizacji danych per-przekrój czasu (`:max`, `:minmax`, `:zscore`, `:none`)
- `wymiary_cech`: Indeksy kolumn cech używanych do wyznaczenia wymiaru LPCA (domyślnie 2:end)
- `zmienna_x`: Zmienna na osi X (`:T`, `:A`, `:B`, `:tau`, `:w` itp.)
- `zmienna_y`: Zmienna na osi Y (`:A`, `:T`, `:B` itp.)
- `zmienna_z`: Zmienna na osi Z (jeśli podana, generowany jest wykres 3D `Axis3`)
- `rozmiar_punktu`: Rozmiar punktów na wykresie
- `przezroczystość`: Współczynnik przezroczystości punktów
- `tytuł_wykresu`: Opcjonalny tytuł wykresu
"""
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
    target_taus = if wybrane_czasy === nothing
        taus_in_data
    elseif wybrane_czasy isa Real
        [Float64(wybrane_czasy)]
    else
        Float64.(wybrane_czasy)
    end

    all_indices = Int[]
    all_dims = Float64[]

    for τ in target_taus
        rows = findall(isapprox.(dataset[:, 1], τ; atol = tolerancja_czasowa))
        if isempty(rows)
            nearest_idx = argmin(abs.(dataset[:, 1] .- τ))
            rows = findall(isapprox.(dataset[:, 1], dataset[nearest_idx, 1]; atol = tolerancja_czasowa))
        end

        X_tau = dataset[rows, wymiary_cech]
        n_points = size(X_tau, 1)
        if n_points < 2
            continue
        end

        X_norm = apply_normalization(X_tau, metoda_regularyzacji)
        d_local = dims(X_norm; k = min(liczba_sąsiadów, n_points), tol = tolerancja_wartości_własnych)

        append!(all_indices, rows)
        append!(all_dims, d_local)
    end

    if isempty(all_indices)
        error("Nie znaleziono żadnych punktów danych dla podanych kryteriów czasu.")
    end

    sub_dataset = dataset[all_indices, :]
    lbl_x, vals_x = _extract_column_or_key(sub_dataset, zmienna_x)
    lbl_y, vals_y = _extract_column_or_key(sub_dataset, zmienna_y)

    fig = Figure(size = (950, 650))
    palette = [:crimson, :dodgerblue, :forestgreen, :darkorange, :purple, :goldenrod, :darkcyan, :mediumvioletred]

    unique_dims = sort(unique(all_dims))

    if zmienna_z !== nothing
        lbl_z, vals_z = _extract_column_or_key(sub_dataset, zmienna_z)
        default_title = tytuł_wykresu !== nothing ? tytuł_wykresu : L"\text{Lokalny wymiar } d_{\mathrm{lokalny}} \text{ w przestrzeni fazowej 3D}"
        ax3 = Axis3(
            fig[1, 1],
            title = default_title,
            xlabel = lbl_x,
            ylabel = lbl_y,
            zlabel = lbl_z,
            azimuth = 1.3 * π,
            elevation = 0.15 * π
        )

        for (i, d_val) in enumerate(unique_dims)
            mask = all_dims .== d_val
            c = palette[mod1(i, length(palette))]
            scatter!(
                ax3,
                vals_x[mask],
                vals_y[mask],
                vals_z[mask];
                color = (c, przezroczystość),
                markersize = rozmiar_punktu,
                strokewidth = 0.2,
                strokecolor = :white,
                label = L"d_{\mathrm{lokalny}} = %$(Int(round(d_val)))"
            )
        end
        axislegend(ax3, position = :rt)
    else
        default_title = tytuł_wykresu !== nothing ? tytuł_wykresu : L"\text{Lokalny wymiar } d_{\mathrm{lokalny}} \text{ w przestrzeni fazowej}"
        ax2 = Axis(
            fig[1, 1],
            title = default_title,
            xlabel = lbl_x,
            ylabel = lbl_y,
            xautolimitmargin = (0.05, 0.05),
            yautolimitmargin = (0.05, 0.05)
        )

        for (i, d_val) in enumerate(unique_dims)
            mask = all_dims .== d_val
            c = palette[mod1(i, length(palette))]
            scatter!(
                ax2,
                vals_x[mask],
                vals_y[mask];
                color = (c, przezroczystość),
                markersize = rozmiar_punktu,
                strokewidth = 0.2,
                strokecolor = :white,
                label = L"d_{\mathrm{lokalny}} = %$(Int(round(d_val)))"
            )
        end
        axislegend(ax2, position = :rt)
    end

    return fig
end

const plot_phase_space_local_pca_dimensions = plot_phase_space_lpca_dims


function to_2d_local_plots(dataset::AbstractArray{<:Real, 3})
    dataset_2d = reshape(permutedims(dataset, (2, 1, 3)), :, size(dataset, 3))
    valid_rows = .!isnan.(dataset_2d[:, 1])
    return dataset_2d[valid_rows, :]
end


function get_data(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return get_data(to_2d_local_plots(dataset), args...; kwargs...)
end
function get_limits(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return get_limits(to_2d_local_plots(dataset), args...; kwargs...)
end
function _split_trajectories(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return _split_trajectories(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_phase_space_grid(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_phase_space_grid(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_phase_space_grid_3d(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_phase_space_grid_3d(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_attractor(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_attractor(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_thermodynamics_evolution(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_thermodynamics_evolution(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_phase_space_evolution(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_phase_space_evolution(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_attractor_Aw_T(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_attractor_Aw_T(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_phase_space_evolution_3d(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_phase_space_evolution_3d(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_pca_evr_over_time(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_pca_evr_over_time(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_pca_bar_variance(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_pca_bar_variance(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_pca_summary(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_pca_summary(to_2d_local_plots(dataset), args...; kwargs...)
end
function animate_pca_evolution(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return animate_pca_evolution(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_lle_dim(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_lle_dim(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_lle_dim!(ax::Axis, dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_lle_dim!(ax, to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_simulation_lle(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_simulation_lle(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_lle_embedding(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_lle_embedding(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_pinn_deff_evolution(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_pinn_deff_evolution(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_lid_dimension(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_lid_dimension(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_twonn(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_twonn(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_lle_results_for_taus(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_lle_results_for_taus(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_lle_spectrum_analysis(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_lle_spectrum_analysis(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_lle_spectrum_scan_analysis(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_lle_spectrum_scan_analysis(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_local_pca(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_local_pca(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_local_pca_regularizations(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_local_pca_regularizations(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_map_lpca(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_map_lpca(to_2d_local_plots(dataset), args...; kwargs...)
end
function plot_phase_space_lpca_dims(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return plot_phase_space_lpca_dims(to_2d_local_plots(dataset), args...; kwargs...)
end

# ==============================================================================
# Publication LPCA Visualizations: K-Dependency, Normalization, Invariance, Stats
# ==============================================================================

"""
    plot_k_dependency_bands(k_results; figure_size)
    plot_k_dependency_bands(dataset; k_pairs, tau_values, ...)

Plots average local dimension trajectories with shaded uncertainty bands between k_base and k_expanded.
"""
function plot_k_dependency_bands(
    k_results::AbstractVector{<:NamedTuple};
    figure_size::Tuple{Integer, Integer} = (900, 560)
)
    set_publication_theme()

    figure = Figure(size = figure_size)
    axis = Axis(
        figure[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )

    palette = [:crimson, :dodgerblue, :forestgreen, :darkorange, :purple, :goldenrod, :darkcyan]

    for (index, item) in enumerate(k_results)
        color = palette[mod1(index, length(palette))]
        k_base = item.k_base
        k_expanded = item.k_expanded
        tau_values = item.tau_values
        mean_k1 = item.mean_dimension_k1
        mean_k2 = item.mean_dimension_k2

        band!(
            axis,
            tau_values,
            mean_k1,
            mean_k2;
            color = (color, 0.25)
        )

        lines!(
            axis,
            tau_values,
            mean_k1;
            linewidth = 2.5,
            color = color,
            label = L"K = %$(k_base), %$(k_expanded)"
        )
    end

    axislegend(axis, position = :rt)
    return figure
end

function plot_k_dependency_bands(
    dataset::AbstractArray{<:Real};
    k_pairs = [(5, 10), (10, 20), (20, 40), (40, 80)],
    tau_values = range(0.2, 5.0, length = 15),
    kwargs...
)
    results = evaluate_k_dependency(dataset, k_pairs, tau_values)
    return plot_k_dependency_bands(results; kwargs...)
end

"""
    plot_normalization_multipanel(method_results; methods_to_plot)
    plot_normalization_multipanel(dataset; methods, k_pairs, tau_values, ...)

Creates a multi-panel figure (2x2 grid) comparing normalization methods.
"""
function plot_normalization_multipanel(
    method_results::Dict{Symbol, Vector{NamedTuple}};
    methods_to_plot::AbstractVector{Symbol} = [:none, :max, :minmax, :zscore]
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

        axis = Axis(
            figure[row, col],
            xlabel = L"\tau\,[\mathrm{fm}/c]",
            ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
            xautolimitmargin = (0.0, 0.04),
            yautolimitmargin = (0.05, 0.05)
        )

        results_for_method = method_results[method]
        for (k_index, item) in enumerate(results_for_method)
            color = palette[mod1(k_index, length(palette))]
            band!(
                axis,
                item.tau_values,
                item.mean_dimension_k1,
                item.mean_dimension_k2;
                color = (color, 0.25)
            )
            lines!(
                axis,
                item.tau_values,
                item.mean_dimension_k1;
                linewidth = 2.0,
                color = color,
                label = L"K = %$(item.k_base), %$(item.k_expanded)"
            )
        end

        if row == 1 && col == column_count
            axislegend(axis, position = :rt)
        end
    end

    return figure
end

function plot_normalization_multipanel(
    dataset::AbstractArray{<:Real};
    methods::AbstractVector{Symbol} = [:none, :max, :minmax, :zscore],
    k_pairs = [(5, 10), (10, 20), (20, 40), (40, 80)],
    tau_values = range(0.2, 5.0, length = 15),
    kwargs...
)
    results = compare_normalization_methods(dataset, methods, k_pairs, tau_values)
    return plot_normalization_multipanel(results; methods_to_plot = methods, kwargs...)
end

"""
    plot_normalization_direct_overlay(dataset, normalization_methods, k_neighbor, tau_values; ...)
"""
function plot_normalization_direct_overlay(
    dataset::AbstractMatrix{<:Real},
    normalization_methods::AbstractVector{Symbol},
    k_neighbor::Integer,
    tau_values::AbstractVector{<:Real};
    feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
    tolerance::Real = 0.01
)
    set_publication_theme()

    figure = Figure(size = (900, 560))
    axis = Axis(
        figure[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )

    labels = Dict(
        :none => L"\text{Raw Units}",
        :max => L"\text{Abs-Max Scaling } ([-1, 1])",
        :minmax => L"\text{Min-Max Scaling } ([0, 1])",
        :zscore => L"\text{Z-Score Standardization}"
    )
    palette = [:crimson, :dodgerblue, :forestgreen, :darkorange]

    for (index, method) in enumerate(normalization_methods)
        means = Float64[]
        for tau in tau_values
            _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
            normalized_slice = apply_normalization(raw_slice, method)
            dims_i = dims(normalized_slice; k = k_neighbor, tol = tolerance)
            push!(means, mean(dims_i))
        end

        color = palette[mod1(index, length(palette))]
        lines!(
            axis,
            tau_values,
            means;
            linewidth = 3.0,
            color = color,
            label = get(labels, method, string(method))
        )
        scatter!(
            axis,
            tau_values,
            means;
            markersize = 8,
            color = color
        )
    end

    axislegend(axis, position = :rt)
    return figure
end
plot_normalization_direct_overlay(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) =
    plot_normalization_direct_overlay(to_2d_local_plots(dataset), args...; kwargs...)

"""
    plot_coordinate_invariance(invariance_result_normalized, invariance_result_raw; figure_size)
    plot_coordinate_invariance(variant_datasets, k_neighbor, tau_values; kwargs...)
"""
function plot_coordinate_invariance(
    invariance_result_normalized::NamedTuple,
    invariance_result_raw::NamedTuple;
    figure_size::Tuple{Integer, Integer} = (1100, 520)
)
    set_publication_theme()

    figure = Figure(size = figure_size)
    tau_values = invariance_result_normalized.tau_values

    axis_left = Axis(
        figure[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )

    axis_right = Axis(
        figure[1, 2],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )

    is_hjsw = invariance_result_normalized.model == :hjsw
    variants = is_hjsw ? [
        (:physical, L"\text{Physical: }(T, \mathcal{A}, \mathcal{B})", :dodgerblue, :solid, 3.0),
        (:dimensionless, L"\text{Dimensionless: }(w, \mathcal{A}, \mathcal{B})", :crimson, :dash, 2.5),
        (:scaled_10x, L"\text{Rescaled: }(10T, \mathcal{A}, \mathcal{B})", :forestgreen, :dot, 2.5)
    ] : [
        (:physical, L"\text{Physical: }(T, \mathcal{A})", :dodgerblue, :solid, 3.0),
        (:dimensionless, L"\text{Dimensionless: }(w, \mathcal{A})", :crimson, :dash, 2.5),
        (:scaled_10x, L"\text{Rescaled: }(10T, \mathcal{A})", :forestgreen, :dot, 2.5)
    ]

    for (key, label_text, color, line_style, width) in variants
        lines!(
            axis_left,
            tau_values,
            invariance_result_normalized.curves[key];
            color = color,
            linestyle = line_style,
            linewidth = width,
            label = label_text
        )

        lines!(
            axis_right,
            tau_values,
            invariance_result_raw.curves[key];
            color = color,
            linestyle = line_style,
            linewidth = width,
            label = label_text
        )
    end

    axislegend(axis_left, position = :rt)
    axislegend(axis_right, position = :rt)

    return figure
end

function plot_coordinate_invariance(
    variant_datasets::NamedTuple,
    k_neighbor::Integer,
    tau_values::AbstractVector{<:Real};
    kwargs...
)
    res_norm = test_coordinate_invariance(variant_datasets, k_neighbor, tau_values; normalize_method = :max)
    res_raw = test_coordinate_invariance(variant_datasets, k_neighbor, tau_values; normalize_method = :none)
    return plot_coordinate_invariance(res_norm, res_raw; kwargs...)
end

function plot_coordinate_invariance_single(
    invariance_result::NamedTuple;
    figure_size::Tuple{Integer, Integer} = (900, 560)
)
    set_publication_theme()

    figure = Figure(size = figure_size)
    tau_values = invariance_result.tau_values

    axis = Axis(
        figure[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )

    is_hjsw = invariance_result.model == :hjsw
    variants = is_hjsw ? [
        (:physical, L"\text{Physical: }(T, \mathcal{A}, \mathcal{B})", :dodgerblue, :solid, 3.0),
        (:dimensionless, L"\text{Dimensionless: }(w, \mathcal{A}, \mathcal{B})", :crimson, :dash, 2.5),
        (:scaled_10x, L"\text{Rescaled: }(10T, \mathcal{A}, \mathcal{B})", :forestgreen, :dot, 2.5)
    ] : [
        (:physical, L"\text{Physical: }(T, \mathcal{A})", :dodgerblue, :solid, 3.0),
        (:dimensionless, L"\text{Dimensionless: }(w, \mathcal{A})", :crimson, :dash, 2.5),
        (:scaled_10x, L"\text{Rescaled: }(10T, \mathcal{A})", :forestgreen, :dot, 2.5)
    ]

    for (key, label_text, color, line_style, width) in variants
        lines!(
            axis,
            tau_values,
            invariance_result.curves[key];
            color = color,
            linestyle = line_style,
            linewidth = width,
            label = label_text
        )
    end

    axislegend(axis, position = :rt)
    return figure
end

"""
    plot_dimension_distribution(distribution_result; figure_size)
    plot_dimension_distribution(dataset, k_neighbor, tau_values; kwargs...)

Generates a 2-panel figure:
Left panel: Mean <d> vs Median with [Q25, Q75] interquartile band.
Right panel: Evolution of discrete probabilities P(d = m) over tau.
"""
function plot_dimension_distribution(
    distribution_result::NamedTuple;
    figure_size::Tuple{Integer, Integer} = (1100, 500)
)
    set_publication_theme()

    figure = Figure(size = figure_size)
    tau_values = distribution_result.tau_values
    embedding_dim = distribution_result.embedding_dimension

    axis_left = Axis(
        figure[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Local Dimension } d",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )

    axis_right = Axis(
        figure[1, 2],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Fraction of Points } P(d)",
        limits = (nothing, nothing, 0.0, 1.05),
        xautolimitmargin = (0.0, 0.04)
    )

    band!(
        axis_left,
        tau_values,
        distribution_result.q25_dimension,
        distribution_result.q75_dimension;
        color = (:dodgerblue, 0.25),
        label = L"\text{IQR (25\% -- 75\%)}"
    )

    lines!(
        axis_left,
        tau_values,
        distribution_result.mean_dimension;
        color = :crimson,
        linewidth = 3.0,
        label = L"\text{Mean } \langle d \rangle"
    )

    lines!(
        axis_left,
        tau_values,
        distribution_result.median_dimension;
        color = :navy,
        linewidth = 2.5,
        linestyle = :dash,
        label = L"\text{Median } d_{1/2}"
    )

    axislegend(axis_left, position = :rt)

    dim_colors = [:forestgreen, :darkorange, :purple, :dodgerblue]
    for dim_val in 1:embedding_dim
        color = dim_colors[mod1(dim_val, length(dim_colors))]
        fractions = distribution_result.dimension_fractions[:, dim_val]
        lines!(
            axis_right,
            tau_values,
            fractions;
            color = color,
            linewidth = 2.8,
            label = L"P(d = %$dim_val)"
        )
        scatter!(
            axis_right,
            tau_values,
            fractions;
            color = color,
            markersize = 7
        )
    end

    axislegend(axis_right, position = :rt)

    return figure
end

function plot_dimension_distribution(
    dataset::AbstractArray{<:Real},
    k_neighbor::Integer,
    tau_values::AbstractVector{<:Real};
    kwargs...
)
    res = analyze_dimension_distribution(dataset, k_neighbor, tau_values; kwargs...)
    return plot_dimension_distribution(res)
end

function plot_dimension_mean_single(
    distribution_result::NamedTuple;
    figure_size::Tuple{Integer, Integer} = (900, 560)
)
    set_publication_theme()

    figure = Figure(size = figure_size)
    tau_values = distribution_result.tau_values

    axis = Axis(
        figure[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Local Dimension } d",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )

    band!(
        axis,
        tau_values,
        distribution_result.q25_dimension,
        distribution_result.q75_dimension;
        color = (:dodgerblue, 0.25),
        label = L"\text{IQR (25\% -- 75\%)}"
    )

    lines!(
        axis,
        tau_values,
        distribution_result.mean_dimension;
        color = :crimson,
        linewidth = 3.0,
        label = L"\text{Mean } \langle d \rangle"
    )

    lines!(
        axis,
        tau_values,
        distribution_result.median_dimension;
        color = :navy,
        linewidth = 2.5,
        linestyle = :dash,
        label = L"\text{Median } d_{1/2}"
    )

    axislegend(axis, position = :rt)
    return figure
end

function plot_dimension_populations_single(
    distribution_result::NamedTuple;
    figure_size::Tuple{Integer, Integer} = (900, 560)
)
    set_publication_theme()

    figure = Figure(size = figure_size)
    tau_values = distribution_result.tau_values
    embedding_dim = distribution_result.embedding_dimension

    axis = Axis(
        figure[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Fraction of Points } P(d)",
        limits = (nothing, nothing, 0.0, 1.05),
        xautolimitmargin = (0.0, 0.04)
    )

    dim_colors = [:forestgreen, :darkorange, :purple, :dodgerblue]
    for dim_val in 1:embedding_dim
        color = dim_colors[mod1(dim_val, length(dim_colors))]
        fractions = distribution_result.dimension_fractions[:, dim_val]
        lines!(
            axis,
            tau_values,
            fractions;
            color = color,
            linewidth = 2.8,
            label = L"P(d = %$dim_val)"
        )
        scatter!(
            axis,
            tau_values,
            fractions;
            color = color,
            markersize = 7
        )
    end

    axislegend(axis, position = :rt)
    return figure
end

"""
    plot_tolerance_sensitivity(sensitivity_result; palette, figure_size)
    plot_tolerance_sensitivity(mis_result, hjsw_result; palette, figure_size)
"""
function plot_tolerance_sensitivity(
    sensitivity_result::NamedTuple;
    palette = [:dodgerblue, :forestgreen, :darkorange, :crimson],
    figure_size::Tuple{Integer, Integer} = (900, 560)
)
    set_publication_theme()
    fig = Figure(size = figure_size)
    ax = Axis(
        fig[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )
    tau_grid = sensitivity_result.tau_values
    for (t_idx, tol_val) in enumerate(sensitivity_result.tolerances)
        color = palette[mod1(t_idx, length(palette))]
        means = sensitivity_result.results[Float64(tol_val)]
        lines!(ax, tau_grid, means; color = color, linewidth = 2.5, label = L"\mathrm{tol} = %$(tol_val)")
    end
    axislegend(ax, position = :rt)
    return fig
end

function plot_tolerance_sensitivity(
    mis_result::NamedTuple,
    hjsw_result::NamedTuple;
    palette = [:dodgerblue, :forestgreen, :darkorange, :crimson],
    figure_size::Tuple{Integer, Integer} = (1100, 520)
)
    set_publication_theme()
    fig = Figure(size = figure_size)
    ax_mis = Axis(
        fig[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )
    ax_hjsw = Axis(
        fig[1, 2],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Mean Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )
    tau_grid = mis_result.tau_values
    for (t_idx, tol_val) in enumerate(mis_result.tolerances)
        color = palette[mod1(t_idx, length(palette))]
        lines!(ax_mis, tau_grid, mis_result.results[Float64(tol_val)]; color = color, linewidth = 2.5, label = L"\mathrm{tol} = %$(tol_val)")
        lines!(ax_hjsw, tau_grid, hjsw_result.results[Float64(tol_val)]; color = color, linewidth = 2.5, label = L"\mathrm{tol} = %$(tol_val)")
    end
    axislegend(ax_mis, position = :rt)
    return fig
end

"""
    plot_colored_phase_space_slice_2d(slice_data; x_col_idx, y_col_idx, x_label, y_label, attractor_curve, figure_size)
"""
function plot_colored_phase_space_slice_2d(
    slice_data::PointwiseDimensionSlice;
    x_col_idx::Integer = 1,
    y_col_idx::Integer = 2,
    x_label::LaTeXString = L"T\,[\mathrm{fm}^{-1}]",
    y_label::LaTeXString = L"\mathcal{A}",
    attractor_curve::Union{NamedTuple, Nothing} = nothing,
    figure_size::Tuple{Integer, Integer} = (800, 600)
)
    set_publication_theme()

    tau_str = string(round(slice_data.tau, digits = 2))
    figure = Figure(size = figure_size, figure_padding = (35, 35, 25, 25))
    axis = Axis(
        figure[1, 1],
        title = L"\tau = %$(tau_str)\,\mathrm{fm}/c",
        titlesize = 20,
        xlabel = x_label,
        ylabel = y_label,
        xautolimitmargin = (0.04, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )

    x_values = slice_data.coordinates[:, x_col_idx]
    y_values = slice_data.coordinates[:, y_col_idx]
    dim_values = slice_data.dimensions

    palette = Dict(
        1.0 => (:crimson, L"d = 1"),
        2.0 => (:dodgerblue, L"d = 2")
    )

    if !isnothing(attractor_curve)
        lines!(
            axis,
            attractor_curve.x,
            attractor_curve.y;
            color = (:black, 0.6),
            linewidth = 3.0,
            linestyle = :dash,
            label = L"\text{Attractor}"
        )
    end

    for target_dim in (2.0, 1.0)
        mask = isapprox.(dim_values, target_dim; atol = 0.1)
        if any(mask)
            color, label_text = palette[target_dim]
            scatter!(
                axis,
                x_values[mask],
                y_values[mask];
                color = (color, 0.85),
                markersize = 8,
                strokewidth = 0.3,
                strokecolor = (:black, 0.3),
                label = label_text
            )
        end
    end

    axislegend(axis, position = :rt)
    return figure
end

"""
    plot_colored_phase_space_grid_2d(dataset, tau_grid; feature_indices, x_label, y_label, k_neighbor, tolerance, normalize_method, figure_size)
"""
function plot_colored_phase_space_grid_2d(
    dataset::AbstractArray{<:Real},
    tau_grid::AbstractVector{<:Real};
    feature_indices::AbstractVector{<:Integer} = [2, 3],
    x_label::LaTeXString = L"T\,[\mathrm{fm}^{-1}]",
    y_label::LaTeXString = L"\mathcal{A}",
    k_neighbor::Integer = 24,
    tolerance::Real = 0.01,
    normalize_method::Symbol = :max,
    figure_size::Tuple{Integer, Integer} = (1200, 1050)
)
    set_publication_theme()

    slice_count = length(tau_grid)
    column_count = min(3, slice_count)
    row_count = ceil(Int, slice_count / column_count)

    figure = Figure(size = figure_size, figure_padding = (35, 35, 25, 25))

    for (index, tau) in enumerate(tau_grid)
        row = div(index - 1, column_count) + 1
        col = mod1(index, column_count)

        tau_str = string(round(tau, digits = 2))
        axis = Axis(
            figure[row, col],
            title = L"\tau = %$(tau_str)\,\mathrm{fm}/c",
            titlesize = 17,
            xlabel = x_label,
            ylabel = y_label,
            xautolimitmargin = (0.04, 0.04),
            yautolimitmargin = (0.05, 0.05)
        )

        slice_data = compute_pointwise_dimensions(
            dataset,
            tau;
            feature_indices = feature_indices,
            k_neighbor = k_neighbor,
            tolerance = tolerance,
            normalize_method = normalize_method
        )

        x_vals = slice_data.coordinates[:, 1]
        y_vals = slice_data.coordinates[:, 2]
        d_vals = slice_data.dimensions

        mask_d2 = isapprox.(d_vals, 2.0; atol = 0.1)
        mask_d1 = isapprox.(d_vals, 1.0; atol = 0.1)

        if any(mask_d2)
            scatter!(
                axis,
                x_vals[mask_d2],
                y_vals[mask_d2];
                color = (:dodgerblue, 0.75),
                markersize = 6,
                strokewidth = 0.2,
                strokecolor = (:black, 0.2),
                label = L"d = 2"
            )
        end

        if any(mask_d1)
            scatter!(
                axis,
                x_vals[mask_d1],
                y_vals[mask_d1];
                color = (:crimson, 0.85),
                markersize = 6,
                strokewidth = 0.2,
                strokecolor = (:black, 0.2),
                label = L"d = 1"
            )
        end

        if row == 1 && col == column_count
            axislegend(axis, position = :rt)
        end
    end

    return figure
end

"""
    plot_colored_phase_space_grid_hjsw_projections(dataset, tau_grid; k_neighbor, tolerance, normalize_method, figure_size)
"""
function plot_colored_phase_space_grid_hjsw_projections(
    dataset::AbstractArray{<:Real},
    tau_grid::AbstractVector{<:Real};
    k_neighbor::Integer = 24,
    tolerance::Real = 0.01,
    normalize_method::Symbol = :max,
    figure_size::Tuple{Integer, Integer} = (1200, 1050)
)
    set_publication_theme()

    slice_count = length(tau_grid)
    column_count = min(3, slice_count)
    row_count = ceil(Int, slice_count / column_count)

    figure = Figure(size = figure_size, figure_padding = (35, 35, 25, 25))

    for (index, tau) in enumerate(tau_grid)
        row = div(index - 1, column_count) + 1
        col = mod1(index, column_count)

        tau_str = string(round(tau, digits = 2))
        axis = Axis(
            figure[row, col],
            title = L"\tau = %$(tau_str)\,\mathrm{fm}/c",
            titlesize = 17,
            xlabel = L"\mathcal{A}",
            ylabel = L"\mathcal{B}",
            xautolimitmargin = (0.04, 0.04),
            yautolimitmargin = (0.05, 0.05)
        )

        slice_data = compute_pointwise_dimensions(
            dataset,
            tau;
            feature_indices = [2, 3, 4],
            k_neighbor = k_neighbor,
            tolerance = tolerance,
            normalize_method = normalize_method
        )

        a_vals = slice_data.coordinates[:, 2]
        b_vals = slice_data.coordinates[:, 3]
        d_vals = slice_data.dimensions

        mask_d3 = isapprox.(d_vals, 3.0; atol = 0.1)
        mask_d2 = isapprox.(d_vals, 2.0; atol = 0.1)
        mask_d1 = isapprox.(d_vals, 1.0; atol = 0.1)

        if any(mask_d3)
            scatter!(
                axis,
                a_vals[mask_d3],
                b_vals[mask_d3];
                color = (:dodgerblue, 0.60),
                markersize = 5,
                label = L"d = 3"
            )
        end

        if any(mask_d2)
            scatter!(
                axis,
                a_vals[mask_d2],
                b_vals[mask_d2];
                color = (:darkorange, 0.80),
                markersize = 5,
                label = L"d = 2"
            )
        end

        if any(mask_d1)
            scatter!(
                axis,
                a_vals[mask_d1],
                b_vals[mask_d1];
                color = (:crimson, 0.90),
                markersize = 6,
                label = L"d = 1"
            )
        end

        if row == 1 && col == column_count
            axislegend(axis, position = :rt)
        end
    end

    return figure
end

"""
    plot_colored_phase_space_slice_hjsw_3d(slice_data; x_label, y_label, z_label, azimuth, elevation, figure_size)
"""
function plot_colored_phase_space_slice_hjsw_3d(
    slice_data::PointwiseDimensionSlice;
    x_label::LaTeXString = L"T\,[\mathrm{MeV}]",
    y_label::LaTeXString = L"\mathcal{A}",
    z_label::LaTeXString = L"\mathcal{B}",
    azimuth::Real = 1.3,
    elevation::Real = 0.15,
    figure_size::Tuple{Integer, Integer} = (900, 750)
)
    set_publication_theme()

    tau_str = string(round(slice_data.tau, digits = 2))
    figure = Figure(size = figure_size, figure_padding = (100, 60, 70, 50))
    axis = Axis3(
        figure[1, 1],
        title = L"\tau = %$(tau_str)\,\mathrm{fm}/c",
        titlesize = 20,
        xlabel = x_label,
        ylabel = y_label,
        zlabel = z_label,
        azimuth = azimuth,
        elevation = elevation,
        xlabeloffset = 45,
        ylabeloffset = 45,
        zlabeloffset = 60
    )

    t_vals = slice_data.coordinates[:, 1]
    a_vals = slice_data.coordinates[:, 2]
    b_vals = slice_data.coordinates[:, 3]
    d_vals = slice_data.dimensions

    mask_d3 = isapprox.(d_vals, 3.0; atol = 0.1)
    mask_d2 = isapprox.(d_vals, 2.0; atol = 0.1)
    mask_d1 = isapprox.(d_vals, 1.0; atol = 0.1)

    if any(mask_d3)
        scatter!(
            axis,
            t_vals[mask_d3],
            a_vals[mask_d3],
            b_vals[mask_d3];
            color = (:dodgerblue, 0.50),
            markersize = 6,
            label = L"d = 3"
        )
    end

    if any(mask_d2)
        scatter!(
            axis,
            t_vals[mask_d2],
            a_vals[mask_d2],
            b_vals[mask_d2];
            color = (:darkorange, 0.75),
            markersize = 6,
            label = L"d = 2"
        )
    end

    if any(mask_d1)
        scatter!(
            axis,
            t_vals[mask_d1],
            a_vals[mask_d1],
            b_vals[mask_d1];
            color = (:crimson, 0.90),
            markersize = 8,
            label = L"d = 1"
        )
    end

    axislegend(axis, position = :rt)
    return figure
end

"""
    plot_parameterization_focus_2x2(sweep_result; colormap, figure_size, max_tau, draw_tunnel)
"""
function plot_parameterization_focus_2x2(
    sweep_result::NamedTuple;
    colormap::Symbol = :phase,
    figure_size::Tuple{Integer, Integer} = (1180, 880),
    max_tau::Real = 7.0,
    draw_tunnel::Bool = true
)
    set_publication_theme()

    fig = Figure(
        size = figure_size,
        figure_padding = (25, 25, 25, 25)
    )

    k_values = sweep_result.k_values
    tau_indices = findall(t -> t <= max_tau + 1e-5, sweep_result.tau_values)
    tau_plot = sweep_result.tau_values[tau_indices]
    k_min = minimum(k_values)
    k_max = maximum(k_values)
    is_hjsw = sweep_result.model == :hjsw

    panel_defs = is_hjsw ? [
        (:physical, L"\mathbf{(a)}\quad (T, \mathcal{A}, \mathcal{B})", :solid, 1, 1),
        (:dimensionless, L"\mathbf{(b)}\quad (w, \mathcal{A}, \mathcal{B})", :dash, 1, 2),
        (:scaled_10x, L"\mathbf{(c)}\quad (10T, \mathcal{A}, \mathcal{B})", :dot, 2, 1),
        (:mixed_scaled, L"\mathbf{(d)}\quad (10w, 2\mathcal{A}, \mathcal{B})", :dashdot, 2, 2)
    ] : [
        (:physical, L"\mathbf{(a)}\quad (T, \mathcal{A})", :solid, 1, 1),
        (:dimensionless, L"\mathbf{(b)}\quad (w, \mathcal{A})", :dash, 1, 2),
        (:scaled_10x, L"\mathbf{(c)}\quad (10T, \mathcal{A})", :dot, 2, 1),
        (:mixed_scaled, L"\mathbf{(d)}\quad (10w, 2\mathcal{A})", :dashdot, 2, 2)
    ]

    axes_list = Axis[]

    for (focused_key, panel_title, _, row, col) in panel_defs
        ax = Axis(
            fig[row, col],
            xlabel = row == 2 ? L"\tau\,[\mathrm{fm}/c]" : "",
            ylabel = col == 1 ? L"\text{Mean Local Dimension } \langle d \rangle" : "",
            title = panel_title,
            titlesize = 18,
            xautolimitmargin = (0.02, 0.04),
            yautolimitmargin = (0.05, 0.05)
        )
        push!(axes_list, ax)

        for (bg_key, _, bg_style, _, _) in panel_defs
            if bg_key != focused_key && haskey(sweep_result.curves, bg_key)
                if draw_tunnel
                    band!(
                        ax,
                        tau_plot,
                        sweep_result.envelopes[bg_key].min[tau_indices],
                        sweep_result.envelopes[bg_key].max[tau_indices];
                        color = (:gray85, 0.30)
                    )
                end

                for k in k_values
                    lines!(
                        ax,
                        tau_plot,
                        sweep_result.curves[bg_key][k][tau_indices];
                        color = (:gray65, 0.35),
                        linewidth = 0.9,
                        linestyle = bg_style
                    )
                end
            end
        end

        if draw_tunnel && haskey(sweep_result.envelopes, focused_key)
            band!(
                ax,
                tau_plot,
                sweep_result.envelopes[focused_key].min[tau_indices],
                sweep_result.envelopes[focused_key].max[tau_indices];
                color = (:gray80, 0.18)
            )
        end

        if haskey(sweep_result.curves, focused_key)
            for k in k_values
                norm_val = k_max > k_min ? (k - k_min) / (k_max - k_min) : 0.5
                color_val = cgrad(colormap)[norm_val]
                lines!(
                    ax,
                    tau_plot,
                    sweep_result.curves[focused_key][k][tau_indices];
                    color = color_val,
                    linewidth = 1.8,
                    linestyle = :solid
                )
            end
        end

        xlims!(ax, minimum(tau_plot), max_tau)
        ylims!(ax, 0.95, is_hjsw ? 3.05 : 2.05)
    end

    linkaxes!(axes_list...)

    Colorbar(
        fig[1:2, 3],
        limits = (k_min, k_max),
        colormap = colormap,
        label = L"\text{Nearest Neighbors } K",
        width = 18,
        ticklabelsize = 14,
        labelsize = 16
    )

    return fig
end

"""
    plot_parameterization_focus_tripanel(sweep_result; colormap, figure_size, max_tau, draw_tunnel)
"""
function plot_parameterization_focus_tripanel(
    sweep_result::NamedTuple;
    colormap::Symbol = :phase,
    figure_size::Tuple{Integer, Integer} = (1550, 460),
    max_tau::Real = 7.0,
    draw_tunnel::Bool = true
)
    set_publication_theme()

    fig = Figure(
        size = figure_size,
        figure_padding = (30, 25, 25, 25)
    )

    k_values = sweep_result.k_values
    tau_indices = findall(t -> t <= max_tau + 1e-5, sweep_result.tau_values)
    tau_plot = sweep_result.tau_values[tau_indices]
    k_min = minimum(k_values)
    k_max = maximum(k_values)
    is_hjsw = sweep_result.model == :hjsw

    panel_defs = is_hjsw ? [
        (:physical, L"\mathbf{(a)}\quad (T, \mathcal{A}, \mathcal{B})", :solid),
        (:dimensionless, L"\mathbf{(b)}\quad (w, \mathcal{A}, \mathcal{B})", :dash),
        (:scaled_10x, L"\mathbf{(c)}\quad (10T, \mathcal{A}, \mathcal{B})", :dot)
    ] : [
        (:physical, L"\mathbf{(a)}\quad (T, \mathcal{A})", :solid),
        (:dimensionless, L"\mathbf{(b)}\quad (w, \mathcal{A})", :dash),
        (:scaled_10x, L"\mathbf{(c)}\quad (10T, \mathcal{A})", :dot)
    ]

    axes_list = Axis[]

    for (p_idx, (focused_key, panel_title, _)) in enumerate(panel_defs)
        ax = Axis(
            fig[1, p_idx],
            xlabel = L"\tau\,[\mathrm{fm}/c]",
            ylabel = p_idx == 1 ? L"\text{Mean Local Dimension } \langle d \rangle" : "",
            title = panel_title,
            titlesize = 18,
            xautolimitmargin = (0.02, 0.04),
            yautolimitmargin = (0.05, 0.05)
        )
        push!(axes_list, ax)

        for (bg_key, _, bg_style) in panel_defs
            if bg_key != focused_key && haskey(sweep_result.curves, bg_key)
                if draw_tunnel
                    band!(
                        ax,
                        tau_plot,
                        sweep_result.envelopes[bg_key].min[tau_indices],
                        sweep_result.envelopes[bg_key].max[tau_indices];
                        color = (:gray85, 0.35)
                    )
                end

                for k in k_values
                    lines!(
                        ax,
                        tau_plot,
                        sweep_result.curves[bg_key][k][tau_indices];
                        color = (:gray65, 0.40),
                        linewidth = 1.0,
                        linestyle = bg_style
                    )
                end
            end
        end

        if draw_tunnel && haskey(sweep_result.envelopes, focused_key)
            band!(
                ax,
                tau_plot,
                sweep_result.envelopes[focused_key].min[tau_indices],
                sweep_result.envelopes[focused_key].max[tau_indices];
                color = (:gray80, 0.20)
            )
        end

        if haskey(sweep_result.curves, focused_key)
            for k in k_values
                norm_val = k_max > k_min ? (k - k_min) / (k_max - k_min) : 0.5
                color_val = cgrad(colormap)[norm_val]
                lines!(
                    ax,
                    tau_plot,
                    sweep_result.curves[focused_key][k][tau_indices];
                    color = color_val,
                    linewidth = 2.2,
                    linestyle = :solid
                )
            end
        end

        xlims!(ax, minimum(tau_plot), max_tau)
    end

    linkaxes!(axes_list...)

    Colorbar(
        fig[1, 4],
        limits = (k_min, k_max),
        colormap = colormap,
        label = L"\text{Nearest Neighbors } K",
        width = 18,
        ticklabelsize = 14,
        labelsize = 16
    )

    return fig
end

"""
    plot_soft_weighted_dimension(scan_result; compare_hard = true, figure_size = (900, 560))
    plot_soft_weighted_dimension(dataset; tau_values, kwargs...)

Visualizes the continuous soft-weighted local dimension trajectory ⟨d⟩_W(τ)
with a ±1σ_W weighted uncertainty band and optional comparison to hard-threshold LPCA.
"""
function plot_soft_weighted_dimension(
    scan_result::NamedTuple;
    compare_hard::Bool = true,
    figure_size::Tuple{Integer, Integer} = (900, 560)
)
    set_publication_theme()

    fig = Figure(size = figure_size)
    ax = Axis(
        fig[1, 1],
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{Local Dimension } \langle d \rangle",
        xautolimitmargin = (0.0, 0.04),
        yautolimitmargin = (0.05, 0.05)
    )

    tau_vals = scan_result.tau_values
    mean_w = scan_result.mean_dims
    std_w = scan_result.std_dims

    band!(
        ax,
        tau_vals,
        mean_w .- std_w,
        mean_w .+ std_w;
        color = (:dodgerblue, 0.25),
        label = L"\pm 1\sigma_W \text{ (Weighted Band)}"
    )

    lines!(
        ax,
        tau_vals,
        mean_w;
        color = :dodgerblue,
        linewidth = 3.0,
        label = L"\langle d \rangle_W \text{ (Soft-Weighted)}"
    )

    if compare_hard && hasfield(typeof(scan_result), :unweighted_hard_means)
        lines!(
            ax,
            tau_vals,
            scan_result.unweighted_hard_means;
            color = :crimson,
            linewidth = 2.0,
            linestyle = :dash,
            label = L"\langle d \rangle \text{ (Standard Hard LPCA)}"
        )
    end

    axislegend(ax, position = :rt)
    return fig
end

function plot_soft_weighted_dimension(
    dataset::AbstractArray{<:Real};
    tau_values = range(0.2, 5.0, length = 20),
    kwargs...
)
    res = scan_soft_weighted_dimension(dataset, tau_values; kwargs...)
    return plot_soft_weighted_dimension(res)
end


