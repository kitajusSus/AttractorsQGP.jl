using CairoMakie
using LaTeXStrings
using ColorSchemes

"""
theme of all those Makie plots.
"""
function set_publication_theme()
    set_theme!(Theme(
        font="TeX Gyre Heros",
        fontsize=16,
        Axis=(
            titlesize=18,
            xlabelsize=18,
            ylabelsize=18,
            xticklabelsize=14,
            yticklabelsize=14,
            backgroundcolor=:white,
            xgridstyle=:dash,
            ygridstyle=:dash,
            xgridcolor=RGBAf(0.8, 0.8, 0.8, 0.5),
            ygridcolor=RGBAf(0.8, 0.8, 0.8, 0.5),
            spinewidth=1.2,
            xtickwidth=1.2,
            ytickwidth=1.2,
            topspinevisible=true,
            rightspinevisible=true,
        ),
        Legend=(
            framevisible=true,
            framewidth=0.8,
            backgroundcolor=:white,
            position=:rt,
        ),
        Palette=(
            color=Makie.wong_colors(),
        ),
    ))
end

const PLOT_KEYS = Dict(
    :T => (L"T", x -> x[2]),
    :A => (L"\\mathcal{A}", x -> x[3]),
    :tauT => (L"\\tau T", x -> x[1] * x[2]),
    :tau2A => (L"\\tau^2 \\mathcal{A}", x -> x[1]^2 * x[3]),
)

"""
Resolve axis definition from Symbol or Tuple(label, fn).
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
Get x/y data for a requested simulation time.
Uses nearest available row in dataset.
"""
function get_data(dataset::AbstractMatrix{<:Real}, t::Real, xdef, ydef)
    @assert size(dataset, 2) == 3 "Dataset must have columns [tau, T, A]."

    xlbl, xfn = resolve_def(xdef)
    ylbl, yfn = resolve_def(ydef)

    rows = findall(isapprox.(dataset[:, 1], t; atol=1e-8))
    if isempty(rows)
        nearest = argmin(abs.(dataset[:, 1] .- t))
        rows = [nearest]
    end

    selected = dataset[rows, :]
    x = [xfn(selected[i, :]) for i in 1:size(selected, 1)]
    y = [yfn(selected[i, :]) for i in 1:size(selected, 1)]

    return (x=x, y=y, xlabel=xlbl, ylabel=ylbl)
end

"""
Grid of phase-space snapshots for selected times.
"""
function plot_phase_space_grid(dataset::AbstractMatrix{<:Real}, times, xdef, ydef)
    set_publication_theme()

    n = length(times)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size=(360 * ncols, 320 * nrows))

    for (i, t) in enumerate(times)
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1
        d = get_data(dataset, t, xdef, ydef)

        ax = Axis(fig[row, col],
            title=L"\\tau = %$(round(t, digits=3))",
            xlabel=d.xlabel,
            ylabel=d.ylabel,
        )

        scatter!(ax, d.x, d.y;
            markersize=7,
            color=:dodgerblue,
            strokecolor=:black,
            strokewidth=0.4,
            alpha=0.8,
        )
    end

    fig
end

"""
Plot trajectory evolution in time for T and A.
"""
function plot_thermodynamics_evolution(dataset::AbstractMatrix{<:Real})
    set_publication_theme()

    τ = dataset[:, 1]
    T = dataset[:, 2]
    A = dataset[:, 3]

    fig = Figure(size=(900, 560))

    ax1 = Axis(fig[1, 1],
        title=L"Ewolucja termodynamiczna",
        xlabel=L"\\tau",
        ylabel=L"T",
    )
    lines!(ax1, τ, T, color=:firebrick, linewidth=2.5, label="T(τ)")
    axislegend(ax1, position=:rt)

    ax2 = Axis(fig[2, 1],
        xlabel=L"\\tau",
        ylabel=L"\\mathcal{A}",
    )
    lines!(ax2, τ, A, color=:midnightblue, linewidth=2.5, label="A(τ)")
    hlines!(ax2, [0.0], color=:gray50, linestyle=:dash)
    axislegend(ax2, position=:rt)

    linkxaxes!(ax1, ax2)

    fig
end

"""
Simple PCA visualization matching project style.
"""
function plot_pca_summary(data::AbstractMatrix{<:Real}; n_components::Int=2)
    set_publication_theme()

    pca = run_pca(data; n_components=n_components)
    ratio = pca.explained_variance_ratio
    transformed = pca.transformed

    fig = Figure(size=(1000, 420))

    ax1 = Axis(fig[1, 1],
        title="Explained variance",
        xlabel="Component",
        ylabel="EVR",
        limits=(0.5, length(ratio) + 0.5, 0, 1),
    )
    barplot!(ax1, 1:length(ratio), ratio, color=collect(ColorSchemes.viridis[range(0.2, 0.9, length=length(ratio))]))

    ax2 = Axis(fig[1, 2],
        title="PCA projection",
        xlabel="PC1",
        ylabel="PC2",
    )

    if size(transformed, 2) >= 2
        scatter!(ax2, transformed[:, 1], transformed[:, 2],
            markersize=8,
            color=1:size(transformed, 1),
            colormap=:magma,
        )
    else
        scatter!(ax2, transformed[:, 1], zeros(size(transformed, 1)),
            markersize=8,
            color=:dodgerblue,
        )
    end

    fig
end
