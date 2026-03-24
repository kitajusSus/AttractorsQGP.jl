using CairoMakie
using LaTeXStrings
using ColorSchemes

"""
Apply the default Makie theme used by this package.

Returns `nothing` and updates Makie's global active theme.

```julia
set_publication_theme()
```
"""
function set_publication_theme()
    set_theme!(
        Theme(
            font = "TeX Gyre Heros",
            fontsize = 16,
            figure_padding = 14,
            Axis = (
                titlesize = 18,
                xlabelsize = 18,
                ylabelsize = 18,
                xticklabelsize = 14,
                yticklabelsize = 14,
                backgroundcolor = RGBf(0.94, 0.94, 0.94),
                xgridstyle = :dash,
                ygridstyle = :dash,
                xgridcolor = RGBAf(0.80, 0.80, 0.80, 0.65),
                ygridcolor = RGBAf(0.80, 0.80, 0.80, 0.65),
                spinewidth = 1.2,
                xtickwidth = 1.2,
                ytickwidth = 1.2,
                topspinevisible = true,
                rightspinevisible = true,
            ),
            Legend = (
                framevisible = true,
                framewidth = 1.0,
                backgroundcolor = RGBf(0.94, 0.94, 0.94),
                position = :rt,
            ),
            Palette = (color = Makie.wong_colors(),),
        ),
    )
end

const PLOT_KEYS = Dict(
    :T => (L"T\,[\mathrm{fm}^{-1}]", x -> x[2]),
    :A => (L"\mathcal{A}", x -> x[3]),
    :tauT => (L"\tau T", x -> x[1] * x[2]),
    :tau2A => (L"\tau^2 \mathcal{A}", x -> x[1]^2 * x[3]),
)

"""
Resolve an axis definition used by plotting helpers.

Accepted formats:
- `Symbol` key from `PLOT_KEYS` (for example `:T`, `:A`, `:tauT`)
- `(label, fn)` tuple where `fn(row)` computes axis value from one dataset row

Returns `(label, fn)`.

```julia
resolve_def(:T)
resolve_def(("custom", row -> row[2] / row[1]))
```
"""
function resolve_def(def)
    if def isa Symbol
        @assert haskey(PLOT_KEYS, def) "Unknown plot key: $def"
        return PLOT_KEYS[def]
    end

    if def isa Tuple && length(def) == 2
        return (def[1], def[2])
    end

    error("Axis definition must be Symbol or Tuple(label, function).")
end

"""
Extract x/y arrays for a selected simulation time.

The dataset must have columns `[tau, T, A]`.
If exact `t` is not present, the nearest available `tau` slice is used.

Returns a named tuple `(x, y, xlabel, ylabel)`.

```julia
get_data(dataset, 0.6, :tauT, :A)
```
"""
function get_data(dataset::AbstractMatrix{<:Real}, t::Real, xdef, ydef)
    @assert size(dataset, 2) == 3 "Dataset must have columns [tau, T, A]."

    xlbl, xfn = resolve_def(xdef)
    ylbl, yfn = resolve_def(ydef)

    rows = findall(isapprox.(dataset[:, 1], t; atol = 1e-8))
    if isempty(rows)
        nearest = argmin(abs.(dataset[:, 1] .- t))
        rows = findall(isapprox.(dataset[:, 1], dataset[nearest, 1]; atol = 1e-8))
    end

    selected = dataset[rows, :]
    x = [xfn(selected[i, :]) for i = 1:size(selected, 1)]
    y = [yfn(selected[i, :]) for i = 1:size(selected, 1)]

    return (x = x, y = y, xlabel = xlbl, ylabel = ylbl)
end

"""
Split a stacked dataset into trajectory row ranges.

Assumes rows are ordered by time inside each trajectory and that a new trajectory
starts when `tau` stops increasing.

Returns a vector of `UnitRange{Int}` that can be used for plotting each line.

```julia
ranges = _split_trajectories(dataset)
```
"""
function _split_trajectories(dataset::AbstractMatrix{<:Real})
    @assert size(dataset, 2) == 3 "Dataset must have columns [tau, T, A]."
    if size(dataset, 1) == 0
        return UnitRange{Int}[]
    end

    starts = Int[1]
    for i = 2:size(dataset, 1)
        if dataset[i, 1] <= dataset[i - 1, 1]
            push!(starts, i)
        end
    end
 # to  definicja typu dla ranges to samo co ranges = [1:0.1:15...]
    ranges = UnitRange{Int}[]
    for k in eachindex(starts)
        s = starts[k]
        e = k < length(starts) ? starts[k + 1] - 1 : size(dataset, 1)
        push!(ranges, s:e)
    end
    return ranges
end

"""
Create a grid of phase-space snapshots for selected times.

`times` is an iterable of requested `tau` values.
`xdef` and `ydef` can be symbols from `PLOT_KEYS` or `(label, fn)` tuples.

Returns a `Figure`.


> example for repl
```julia
plot_phase_space_grid(dataset, [0.3, 0.5, 0.7], :tauT, :A)
```
"""
function plot_phase_space_grid(dataset::AbstractMatrix{<:Real}, times, xdef, ydef)
    set_publication_theme()

    n = length(times)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size = (360 * ncols, 290 * nrows))

    for (i, t) in enumerate(times)
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1
        d = get_data(dataset, t, xdef, ydef)

        ax = Axis(
            fig[row, col],
            title = L"\tau = %$(round(t, digits=2))\,\mathrm{fm}/c",
            xlabel = d.xlabel,
            ylabel = d.ylabel,
        )

        scatter!(ax, d.x, d.y; markersize = 2.4, color = :midnightblue, alpha = 0.72)
    end

    fig
end

"""
Plot time evolution of temperature `T` and anisotropy `A` for all trajectories.

The input dataset must contain stacked trajectories in columns `[tau, T, A]`.

Returns a `Figure` with two linked x-axes.

```julia
plot_thermodynamics_evolution(dataset)
```
"""
function plot_thermodynamics_evolution(dataset::AbstractMatrix{<:Real})
    set_publication_theme()
    trajs = _split_trajectories(dataset)

    fig = Figure(size = (950, 620))
    ax1 = Axis(
        fig[1, 1],
        title = L"\text{Ewolucja Temperatury } T\,[\mathrm{fm}^{-1}]\; \text{w czasie własnym } \tau",
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"T\,[\mathrm{fm}^{-1}]",
    )

    for tr in trajs
        lines!(
            ax1,
            dataset[tr, 1],
            dataset[tr, 2],
            color = (:dodgerblue, 0.20),
            linewidth = 1.2,
        )
    end

    ax2 = Axis(
        fig[2, 1],
        title = L"\text{Ewolucja Anizotropii}\; \mathcal{A(τ)}\; \text{ w czasie własnym}",
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"\mathcal{A}",
    )

    for tr in trajs
        lines!(
            ax2,
            dataset[tr, 1],
            dataset[tr, 3],
            color = (:dodgerblue, 0.20),
            linewidth = 1.2,
        )
    end

    hlines!(
        ax2,
        [0.0],
        color = :red,
        linestyle = :dash,
        linewidth = 1.8,
        label = L"\mathcal{A}=0\;(\text{Anizotropia} = 0)",
    )
    axislegend(ax2, position = :rt)

    linkxaxes!(ax1, ax2)
    return fig
end

"""
Plot explained variance ratio (EVR) over time.

PCA is computed independently for each `tau` slice via `run_pca_per_time`.

Returns a `Figure`.

```julia
plot_pca_evr_over_time(dataset; n_components=2, method=:minmax, feature_cols=[2, 3])
```
"""
function plot_pca_evr_over_time(
    dataset::AbstractMatrix{<:Real};
    n_components::Int = 2,
    method::Symbol = :minmax,
    gamma::Float64 = 1.0,
    feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
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

    fig = Figure(size = (900, 460))
    ax = Axis(
        fig[1, 1],
        title = "Explained Variance  (EVR) w funkcji czasu",
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = "EVR",
        limits = (minimum(taus), maximum(taus), 0, 1),
    )

    hlines!(ax, [1.0], color = :gray45, linestyle = :dash, label = "100%")

    palette = Makie.wong_colors()
    for comp = 1:n_components
        vals = evr[:, comp]
        mask = .!isnan.(vals)
        if any(mask)
            lines!(
                ax,
                taus[mask],
                vals[mask],
                linewidth = 2.8,
                color = palette[min(comp, end)],
                label = "PC$(comp)",
            )
            if comp == 1
                band!(
                    ax,
                    taus[mask],
                    zeros(sum(mask)),
                    vals[mask],
                    color = (palette[1], 0.10),
                )
            end
        end
    end

    axislegend(ax, position = :rb)
    fig
end

"""
    plot_pca_summary(
        dataset::AbstractMatrix{<:Real};
        tau::Union{Nothing,Real}=nothing,
        tau_tol::Float64=1e-8,
        tau_mode::Symbol=:nearest,
        n_components::Int=2,
        method::Symbol=:minmax,
        gamma::Float64=1.0,
    )

DONT USE THIS FUNCTION DIRECTLY. Use `plot_pca_evr_over_time` instead.
This helper plots a PCA summary for a single dataset slice.

Dataset columns are expected as `[tau, T, A, ...]`.
If `tau` is provided, rows are selected according to `tau_mode`:

- `:strict`  -> use rows with `abs(tau - tau) <= tau_tol`
- `:nearest` -> if strict match exists, use it; otherwise use nearest available tau

If `tau === nothing`, all rows are used (legacy behavior).

Left panel: projection on principal components.
Right panel: explained variance ratio (EVR).

`method` can be `:minmax` (standard PCA pipeline) or `:kernel`.

Returns a `Figure`.
"""
function plot_pca_summary(
    dataset::AbstractMatrix{<:Real};
    tau::Union{Nothing,Real} = nothing,
    tau_tol::Float64 = 1e-8,
    tau_mode::Symbol = :nearest,
    n_components::Int = 2,
    method::Symbol = :minmax,
    gamma::Float64 = 1.0,
)
    @assert size(dataset, 2) >= 3 "Dataset must contain at least [tau, T, A]."
    @assert tau_tol >= 0 "tau_tol must be >= 0."
    @assert tau_mode in (:strict, :nearest) "tau_mode must be :strict or :nearest."

    set_publication_theme()

    data = Matrix{Float64}(dataset)
    subtitle = "all τ"

    if tau !== nothing
        τ = Float64(tau)
        τcol = data[:, 1]
        d = abs.(τcol .- τ)
        strict_mask = d .<= tau_tol

        if any(strict_mask)
            data = data[strict_mask, :]
            subtitle = "τ=$(τ) ± $(tau_tol)"
        else
            if tau_mode === :strict
                error("No rows found for tau=$(τ) within tau_tol=$(tau_tol).")
            else
                i = argmin(d)
                τnearest = τcol[i]
                near_mask = τcol .== τnearest
                data = data[near_mask, :]
                @warn "No exact tau slice for tau=$(τ) within tau_tol=$(tau_tol). Using nearest tau=$(τnearest)."
                subtitle = "requested τ=$(τ), using nearest τ=$(τnearest)"
            end
        end
    end

    features = data[:, 2:3]  # [T, A]
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
        xlabel = "PC1",
        ylabel = "PC2",
        title = "PCA projection ($subtitle)",
    )
    if n_show >= 2
        scatter!(
            ax_proj,
            transformed[:, 1],
            transformed[:, 2];
            markersize = 4.5,
            color = (:midnightblue, 0.75),
        )
    elseif n_show == 1
        scatter!(
            ax_proj,
            transformed[:, 1],
            zeros(size(transformed, 1));
            markersize = 4.5,
            color = (:midnightblue, 0.75),
        )
    end

    ax_evr = Axis(
        fig[1, 2],
        xlabel = "Principal component",
        ylabel = "EVR",
        limits = (0.5, max(length(evr), 1) + 0.5, 0, 1),
        title = "Explained variance ratio",
    )
    if !isempty(evr)
        barplot!(ax_evr, 1:length(evr), evr; color = :slateblue3)
    end

    return fig

end



function plot_lle_dim(dataset::AbstractMatrix{<:Real}, k::Int, d::Int, tau::Real)
    set_publication_theme()

    lle_data = run_lle_per_time(dataset; k = k, d = d)

    if !haskey(lle_data.lle_results, tau)
        error("Wartość tau = $tau nie została znaleziona w wynikach LLE.")
    end

    embedding = lle_data.lle_results[tau]

    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1], title = "LLE: k=$k neighbors, d=$d dimensions, tau=$tau")

    if d == 1
        ax.xlabel = "LLE1"
        ax.ylabel = "Wartość stała"
        scatter!(
            ax,
            embedding[:, 1],
            zeros(size(embedding, 1));
            markersize = 4.5,
            color = (:midnightblue, 0.75),
        )
    else
        ax.xlabel = "LLE1"
        ax.ylabel = "LLE2"
        scatter!(
            ax,
            embedding[:, 1],
            embedding[:, 2];
            markersize = 4.5,
            color = (:midnightblue, 0.75),
        )
    end

    return fig
end

"""
    function plot_lle_dim!(ax::Axis, dataset::AbstractMatrix{<:Real}, k::Int, d::Int, tau::Real)
Helper function to plot LLE embedding for a single tau slice on an existing axis.
Dataset columns are expected as `[tau, T, A, ...]`.
The LLE embedding is computed via `run_lle_per_time` and plotted as a scatter
plot. The axis title is set to indicate the LLE parameters.

"""
function plot_lle_dim!(ax::Axis, dataset::AbstractMatrix{<:Real}, k::Int, d::Int, tau::Real)
    lle_data = run_lle_per_time(dataset; k = k, d = d)
    embedding = lle_data.lle_results[tau]

    ax.title = L"\text{LLE: } k=%$k, d=%$d, \tau=%$tau"

    if d == 1
        ax.xlabel = "Odwzorowanie 1"
        ax.ylabel = "Wartość stała"
        scatter!(
            ax,
            embedding[:, 1],
            zeros(size(embedding, 1));
            markersize = 4.5,
            color = (:midnightblue, 0.75),
        )
    else
        ax.xlabel = "Odwzorowanie LLE1"
        ax.ylabel = "Odwzorowanie LLE2"
        scatter!(
            ax,
            embedding[:, 1],
            embedding[:, 2];
            markersize = 4.5,
            color = (:midnightblue, 0.75),
        )
    end
    return nothing
end

function plot_simulation_lle(dataset::AbstractMatrix{<:Real}, k::Int, d::Int, tau_zakres)
    set_publication_theme()

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
function animate_pca_evolution(
    dataset::AbstractMatrix{<:Real};
    filename::String = "pca_evolution.gif",
    fps::Int = 15,
    n_components::Int = 2,
    method::Symbol = :minmax,
    gamma::Float64 = 1.0,
    tau_tol::Float64 = 1e-8
)
    set_publication_theme()

    taus = sort(unique(dataset[:, 1]))

    fig = Figure(size = (800, 600))

    title_obs = Observable("PCA projection (tau = $(taus[1]))")
    ax = Axis(
        fig[1, 1],
        xlabel = "PC1",
        ylabel = "PC2",
        title = title_obs
    )

    pts_obs = Observable(Point2f[])

    scatter!(
        ax,
        pts_obs;
        markersize = 6.0,
        color = (:midnightblue, 0.75)
    )

    record(fig, filename, taus; framerate = fps) do t
        title_obs[] = "PCA projection (tau = $(round(t, digits=3)))"

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

            autolimits!(ax)
        end
    end

    return filename
end
