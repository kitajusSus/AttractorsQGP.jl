"""
    plot_phase_space_grid(dataset, times, xdef, ydef; attractor=nothing, attractor_points=500, show_legend=false, kwargs...) -> Figure

Rysuje siatkę przekrojów czasowych przestrzeni fazowej dla wybranych chwil `times`.
Obsługuje zarówno macierze 2D (`AbstractMatrix`), jak i tablice 3D symulacji (`AbstractArray{<:Real, 3}`).

# Argumenty pozycyjne:
- `dataset`: Zbiór danych (tablica 3D `[trajektorie, kroki czasowe, zmienne]` lub macierz 2D `[punkty, kolumny]`).
- `times`: Wektor chwil czasowych \$\\tau\$ dla poszczególnych paneli siatki (np. `[0.2, 0.4, 0.6, 1.0]`).
- `xdef`: Zmienna na osi X (np. `:T`, `:tauT`, `:w`, `:T_norm`).
- `ydef`: Zmienna na osi Y (np. `:A`, `:B`, `:A_norm`, `:Tdot`).

# Dostępne definicje osi (`xdef`, `ydef`):
- `:T`: Temperatura \$T\\,[\\mathrm{fm}^{-1}]\$
- `:A`: Anizotropia ciśnień \$\\mathcal{A}\$
- `:B`: Mod niehydrodynamiczny \$\\mathcal{B}\$ (dla modeli takich jak HJSW)
- `:tauT` / `:w`: Skalowana zmienna atraktora \$w = \\tau T\$
- `:Tdot`: Pochodna temperatury \$\\dot{T}\\,[\\mathrm{fm}^{-2}]\$
- `:T_norm`, `:A_norm`, `:B_norm`: Zmienne znormalizowane do wartości maksymalnych

# Argumenty nazwane (`kwargs`):
- `attractor = nothing`: Rysuje teoretyczną krzywą atraktora (`true`, obiekt `Attractor` lub ścieżka).
- `attractor_points::Int = 500`: Liczba punktów na krzywej atraktora.
- `show_legend::Bool = false`: Czy wyświetlić legendę na wykresie.

# Przykład:
```julia
times = [0.2, 0.4, 0.6, 1.0]
# Wykres (T, A):
fig = plot_phase_space_grid(mis, times, :T, :A; attractor = true)

# Wykres w zmiennych atraktora (tau*T, A):
fig = plot_phase_space_grid(mis, times, :tauT, :A; attractor = true)
```
"""
function plot_phase_space_grid(
        dataset::AbstractMatrix{<:Real},
        times,
        xdef,
        ydef;
        attractor = nothing,
        Attractor = nothing,
        attractor_points::Int = 500,
        show_legend::Bool = false,
        kwargs...
    )
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]

    n = length(times)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size = (480 * ncols + 60, 420 * nrows + 40), figure_padding = (60, 50, 40, 40))
    colgap!(fig.layout, 55)
    rowgap!(fig.layout, 60)

    lims = get_limits(dataset, xdef, ydef; times = times)

    raw_attr = isnothing(Attractor) ? attractor : Attractor
    attr_data = _resolve_attractor_data(raw_attr, dataset)

    for (i, t) in enumerate(times)
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1
        d = get_data(dataset, t, xdef, ydef)

        ax = Axis(
            fig[row, col],
            title = L"\tau = %$(round(t, digits=2))\,\mathrm{fm}/c",
            xlabel = d.xlabel,
            ylabel = d.ylabel,
            xlabelsize = 30,
            ylabelsize = 30,
            xticklabelsize = 20,
            yticklabelsize = 20,
            limits = lims
        )
        if i > 3 && ydef === :A
            ylims!(ax, 0, 4)
        end

        if !isnothing(attr_data)
            attr_line = get_attractor_line_for_frame(
                dataset,
                attr_data,
                t,
                xdef,
                ydef;
                limits = [0, lims[2], lims[3], lims[4]],
                n_points = attractor_points,
            )
            xlims!(ax, 0, lims[2])
            lines!(ax, attr_line.x, attr_line.y; color = (RGBAf(176 / 255, 0, 0, 1.0), 0.86), linewidth = 5, label = "Atraktor")
            if show_legend
                axislegend(ax, position = :rt)
            end
        end
        scatter!(
            ax, d.x[1:min(10000, length(d.x))], d.y[1:min(10000, length(d.y))];
            # ax, d.x, d.y;
            markersize = 3.5, color = (palette[10], 0.9),
            strokecolor = (palette[9], 0.01), strokewidth = 0.2
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
        Attractor = nothing,
        attractor_points::Int = 150,
        azimuth::Real = -1.2,
        elevation::Real = 0.25,
        n_points_scatter::Int = 5000,
        show_legend::Bool = false,
        kwargs...
    )
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]

    xlbl, xfn = resolve_def(xdef)
    ylbl, yfn = resolve_def(ydef)
    zlbl, zfn = resolve_def(zdef)

    raw_attr = isnothing(Attractor) ? attractor : Attractor
    attr_data = _resolve_attractor_data(raw_attr, dataset)

    n = length(times)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size = (680 * ncols + 150, 580 * nrows), figure_padding = (80, 70, 60, 60))
    colgap!(fig.layout, 75)
    rowgap!(fig.layout, 85)

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
            titlesize = 22,
            xlabel = xlbl, ylabel = ylbl, zlabel = zlbl,
            xlabelsize = 20, ylabelsize = 20, zlabelsize = 20,
            xticklabelsize = 16, yticklabelsize = 16, zticklabelsize = 16,
            azimuth = azimuth, elevation = elevation,
            xlabeloffset = 55, ylabeloffset = 55, zlabeloffset = 70,
            xticks = LinearTicks(4), yticks = LinearTicks(4), zticks = LinearTicks(4)
        )
        !isnothing(limits) && limits!(ax, limits...)

        if !isnothing(attr_data)
            tau_val = selected[1, 1]
            Tmin, Tmax = minimum(dataset[:, 2]) * 0.95, maximum(dataset[:, 2]) * 1.05
            T_grid = temperature_grid(Tmin, Tmax, attractor_points)
            A_attr = build_attractor_interpolant(attr_data)
            attr_x = Vector{Float64}(undef, attractor_points)
            attr_y = Vector{Float64}(undef, attractor_points)
            attr_z = Vector{Float64}(undef, attractor_points)
            state = zeros(Float64, size(dataset, 2))
            state[1] = tau_val
            for (k, T) in enumerate(T_grid)
                state[2] = T
                state[3] = A_attr(tau_val * T)
                attr_x[k] = xfn(state, selected)
                attr_y[k] = yfn(state, selected)
                attr_z[k] = zfn(state, selected)
            end
            lines!(ax, attr_x, attr_y, attr_z; color = (RGBAf(176 / 255, 0, 0, 1.0), 0.9), linewidth = 4, label = "Atraktor")
            if show_legend
                axislegend(ax, position = :rt)
            end
        end

        n_pts = min(length(dx), n_points_scatter)
        scatter!(
            ax, dx[1:n_pts], dy[1:n_pts], dz[1:n_pts];
            markersize = 10.0, color = (palette[10], 0.9),
            strokecolor = (palette[6], 0.01), strokewidth = 0.1
        )
    end
    return fig
end

function plot_attractor(
        dataset::AbstractMatrix{<:Real},
        T0_target::Union{Real, AbstractVector{<:Real}, Nothing} = nothing;
        attractor = nothing,
        Attractor = nothing,
        tol::Real = 0.05,
        n_trajectories::Int = 50,
        group_tol::Real = 0.01,
        kwargs...
    )
    set_publication_theme()
    fig = Figure(size = (950, 600))
    ax = Axis(fig[1, 1], xlabel = L"w = \tau T", ylabel = L"\mathcal{A}", limits = (0, 3, -1, 8))

    trajs = _split_trajectories(dataset)
    wybrane_trajs = if T0_target isa Real
        [tr for tr in trajs if abs(dataset[tr[1], 2] - T0_target) < tol]
    elseif T0_target isa AbstractVector
        [tr for tr in trajs if any(abs(dataset[tr[1], 2] - t) < tol for t in T0_target)]
    else
        trajs
    end
    sort!(wybrane_trajs, by = tr -> dataset[tr[1], 2])

    raw_attr = isnothing(Attractor) ? attractor : Attractor
    attr_data = _resolve_attractor_data(raw_attr, dataset)

    if !isnothing(attr_data)
        omega_attr = attr_data[:, 1] .* attr_data[:, 2]
        A_attr = attr_data[:, 3]
        lines!(ax, omega_attr, A_attr, color = (:red, 1), linewidth = 7.0, label = L"\text{Teoretyczny atraktor}")
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
        lines!(ax, omega_traj, A_traj, color = (col, 0.5), linewidth = 1.5)
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
    ax1 = Axis(fig[1, 1], title = L"\text{Ewolucja Temperatury } T\,[\mathrm{fm}^{-1}]\; \text{w czasie własnym } \tau", xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"T\,[\mathrm{fm}^{-1}]")
    for tr in trajektorie
        lines!(ax1, dataset[tr, 1], dataset[tr, 2], color = (palette[3], 0.2), linewidth = 1.5)
    end

    ax2 = Axis(fig[2, 1], title = L"\text{Ewolucja Anizotropii}\; \mathcal{A}(\tau)\; \text{ w czasie własnym}", xlabel = L"\tau\,[\mathrm{fm}/c]", ylabel = L"\mathcal{A}")
    for tr in trajektorie
        lines!(ax2, dataset[tr, 1], dataset[tr, 3], color = (palette[2], 0.2), linewidth = 1.5)
    end
    hlines!(ax2, [0.0], color = :red, linestyle = :dash, linewidth = 2.0, label = L"\mathcal{A}=0\;(\text{Anizotropia} = 0)")
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
    ax = Axis(fig[1, 1], title = L"\text{Ewolucja w przestrzeni fazowej } (T, \mathcal{A})\; \text{w czasie własnym } \tau", xlabel = L"T\,[\mathrm{fm}^{-1}]", ylabel = L"\mathcal{A}")
    for (i, tr) in enumerate(trajektorie)
        col = palette[mod1(i, length(palette))]
        lines!(ax, dataset[tr, 2], dataset[tr, 3], color = (col, 0.75), linewidth = 1.5)
    end
    hlines!(ax, [0.0], color = :black, linestyle = :dash, linewidth = 2.0, label = L"\mathcal{A}=0\;(\text{Anizotropia} = 0)")
    axislegend(ax, position = :rt)
    return fig
end

function plot_attractor_Aw_T(
        dataset::AbstractMatrix{<:Real},
        T_zadana::Real;
        attractor = nothing,
        Attractor = nothing,
        T_tol::Real = 1.0e-3,
        kwargs...
    )
    set_publication_theme()
    T_col = dataset[:, 2]
    rows = findall(isapprox.(T_col, T_zadana; atol = T_tol))
    if isempty(rows)
        nearest_idx = argmin(abs.(T_col .- T_zadana))
        T_najblizsze = T_col[nearest_idx]
        rows = findall(isapprox.(T_col, T_najblizsze; atol = T_tol))
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

    raw_attr = isnothing(Attractor) ? attractor : Attractor
    attr_data = _resolve_attractor_data(raw_attr, dataset)

    if !isnothing(attr_data)
        omega_attr = attr_data[:, 1] .* attr_data[:, 2]
        lines!(ax, omega_attr, attr_data[:, 3], color = :red, linewidth = 3.0, label = L"\text{Teoretyczny atraktor}")
    end
    palette = Makie.theme(:Palette).color[]
    scatter!(ax, omega_slice, A_slice, markersize = 8, color = (palette[1], 0.6), strokecolor = :white, strokewidth = 0.5, label = L"\text{Dane symulacji}")
    axislegend(ax, position = :rb)
    return fig
end

plot_phase_space_grid(dataset::AbstractArray{<:Real, 3}, times, xdef, ydef; kwargs...) = plot_phase_space_grid(to_2d_local_plots(dataset), times, xdef, ydef; kwargs...)
plot_phase_space_grid(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_phase_space_grid(to_2d_local_plots(dataset), args...; kwargs...)

plot_phase_space_grid_3d(dataset::AbstractArray{<:Real, 3}, times, xdef, ydef, zdef; kwargs...) = plot_phase_space_grid_3d(to_2d_local_plots(dataset), times, xdef, ydef, zdef; kwargs...)
plot_phase_space_grid_3d(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_phase_space_grid_3d(to_2d_local_plots(dataset), args...; kwargs...)
plot_attractor(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_attractor(to_2d_local_plots(dataset), args...; kwargs...)
plot_thermodynamics_evolution(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_thermodynamics_evolution(to_2d_local_plots(dataset), args...; kwargs...)
plot_phase_space_evolution(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_phase_space_evolution(to_2d_local_plots(dataset), args...; kwargs...)
plot_attractor_Aw_T(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_attractor_Aw_T(to_2d_local_plots(dataset), args...; kwargs...)
