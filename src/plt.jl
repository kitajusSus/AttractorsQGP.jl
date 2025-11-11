include("lib.jl")
include("pca.jl")

module modPlots

using GLMakie
using LaTeXStrings
using Printf
using ..modHydroSim
using ..modPCA

set_theme!(theme_latexfonts())

export plot_phase_space_snapshot,
    plot_anisotropy_evolution,
    animate_phase_space_evolution,
    plot_phase_space_grid,
    plot_explained_variance_evolution,
    visualize_pca_static_grid,
    plot_loadings_evolution,
    plot_pca_snapshot

"""
Rysuje dane przestrzeni fazowej dla jednego momentu czasu `t`
na podanej osi `ax`. Używa `extract_phase_space_slice` do pobrania danych.
"""
function plot_phase_space_snapshot!(
    ax::Axis,
    simres::modHydroSim.SimResult,
    t::Float64,
)
    u_vals, du_vals, mask = modHydroSim.extract_phase_space_slice(simres, t)

    ax.title = "τ = $(round(t, digits=2)) fm/c"

    if !any(mask)
        text!(ax, "Brak danych", position=(0.5, 0.5), align=(:center, :center))
        @warn "Brak poprawnych danych dla τ = $t. Oś będzie pusta."
        return ax
    end

    t0 = simres.settings.tspan[1]

    T_vals = u_vals[1][mask]
    dT_vals = du_vals[1][mask]
    points = Point2f.(t0 .* T_vals, (t0^2) .* dT_vals)

    scatter!(
        ax,
        points,
        markersize=4,
        strokewidth=0,
        alpha=0.7,
        color=:blue,
    )

    return ax
end

"""
Generuje siatkę statycznych wykresów przestrzeni fazowej
dla podanej listy czasów `times`.

Wywołuje `plot_phase_space_snapshot!` dla każdego wykresu.
"""
function plot_phase_space_grid(
    simres::modHydroSim.SimResult,
    times::AbstractVector{<:Real};
    layout=nothing,
    fig_size=(800, 700),
)
    println("plt.plot_phase_space_grid")

    n_plots = length(times)
    if isnothing(layout)
        cols = ceil(Int, sqrt(n_plots))
        rows = ceil(Int, n_plots / cols)
        layout = (rows, cols)
    end

    fig = Figure(size=fig_size)
    println("Obliczanie globalnych limitów dla siatki...")
    xlims, ylims = _calculate_global_limits(simres, simres.settings.tspan)
    if isnothing(xlims)
        println("Warning: Nie można ustalić limitów.")
        xlims = (nothing, nothing)
        ylims = (nothing, nothing)
    end

    fig[0, 1:layout[2]] = Label(
        fig,
        "Phase Space Evolution [$(simres.settings.theory)]",
        fontsize=18,
        font=:bold
    )

    plot_count = 1
    for r in 1:layout[1], c in 1:layout[2]
        if plot_count > n_plots
            break
        end

        t = times[plot_count]

        ax = Axis(
            fig[r+1, c], #
            xlabel=(r == layout[1]) ? L"\tau_0 T" : "",
            ylabel=(c == 1) ? L"\tau_0^2 \dot{T}" : "",
            limits=(xlims, ylims),
            xticklabelsize=12,
            yticklabelsize=12,
            xticklabelsvisible=(r == layout[1]),
            yticklabelsvisible=(c == 1),
        )

        plot_phase_space_snapshot!(ax, simres, t)

        plot_count += 1
    end

    println("Gotowe. Zwracam figurę.")
    return fig
end

function plot_anisotropy_evolution(simres::SimResult)
    settings = simres.settings
    fig = Figure(size=(1000, 750))
    ax = Axis(
        fig[1, 1],
        title="Anisotropy Evolution A(τ) for $(settings.theory)",
        xlabel=L"\tau \text{ [fm/c]}",
        ylabel=L"A(\tau)",
        xticklabelsize=14,
        yticklabelsize=14
    )

    for sol in simres.solutions
        t_vals = sol.t
        A_vals = [u[2] for u in sol.u]

        valid_indices = findall(isfinite.(A_vals))
        if !isempty(valid_indices)
            lines!(ax, t_vals[valid_indices], A_vals[valid_indices], lw=1.5, alpha=0.4)
        end
    end
    return fig
end

function _calculate_global_limits(simres::SimResult, t_range)
    t0 = simres.settings.tspan[1]
    all_x = Float64[]
    all_y = Float64[]

    t_samples = range(t_range..., length=20)
    for t_sample in t_samples
        u_vals, du_vals, mask = extract_phase_space_slice(simres, t_sample)

        if !any(mask)
            continue
        end

        T_vals = u_vals[1][mask]
        dT_vals = du_vals[1][mask]

        append!(all_x, t0 .* T_vals)
        append!(all_y, (t0^2) .* dT_vals)
    end

    if isempty(all_x) || isempty(all_y)
        return (nothing, nothing)
    end

    xlims = (minimum(all_x), maximum(all_x))
    ylims = (minimum(all_y), maximum(all_y))

    x_pad = (xlims[2] - xlims[1]) * 0.05
    y_pad = (ylims[2] - ylims[1]) * 0.05

    return (
        (xlims[1] - x_pad, xlims[2] + x_pad),
        (ylims[1] - y_pad, ylims[2] + y_pad),
    )
end




function animate_phase_space_evolution(
    simres::SimResult;
    output_filename="phase_space_evolution.gif",
    t_steps=100,
    fps=20,
)

    println("plt.animate_phase_space_evolution")
    t_span = simres.settings.tspan
    t_range = range(t_span..., length=t_steps)
    t0 = t_span[1]

    println("Pre-calculating axis limits for animation...")
    xlims, ylims = _calculate_global_limits(simres, t_span)
    if isnothing(xlims)
        println("Warning: Could not determine data limits. Animation might fail.")
        xlims = (0, 1)
        ylims = (-1, 1)
    end

    fig = Figure(size=(800, 650))

    t_observable = Observable(t_range[1])
    ax_title = @lift(
        "Phase Space at τ = $(round($t_observable, digits=2)) fm/c [$(simres.settings.theory)]"
    )
    ax = Axis(
        fig[1, 1],
        xlabel=L"\tau_0 T",
        ylabel=L"\tau_0^2 \dot{T}",
        limits=(xlims, ylims),
        title=ax_title,
        xticklabelsize=14,  # Zwiększa cyfry na osi X
        yticklabelsize=14   # Zwiększa cyfry na osi Y
    )

    points = @lift begin
        u_vals, du_vals, mask = extract_phase_space_slice(simres, $t_observable)
        T_vals = u_vals[1][mask]
        dT_vals = du_vals[1][mask]
        Point2f.(t0 .* T_vals, (t0^2) .* dT_vals)
    end

    scatter!(
        ax,
        points,
        markersize=4,
        strokewidth=0,
        alpha=0.7,
        color=:blue,
    )

    println("Recording animation to $output_filename...")
    record(fig, output_filename, t_range; framerate=fps) do t
        t_observable[] = t
        if t == 0.55
            display(fig)
            readline()
        end
    end

    println("Animation saved successfully.")
    return fig
end


"""
Rysuje dane przestrzeni fazowej dla jednego momentu czasu `t`
na podanej osi `ax`.
"""
function plot_phase_space_at_time!(
    ax::Axis,
    simres::SimResult,
    t::Float64,
)
    t0 = simres.settings.tspan[1]

    # Ustaw tytuł dla tej konkretnej osi
    ax.title = "τ = $(round(t, digits=2)) fm/c"

    # Wyekstrahuj i oblicz punkty
    u_vals, du_vals, mask = extract_phase_space_slice(simres, t)
    T_vals = u_vals[1][mask]
    dT_vals = du_vals[1][mask]
    points = Point2f.(t0 .* T_vals, (t0^2) .* dT_vals)

    # Narysuj punkty
    scatter!(
        ax,
        points,
        markersize=4,
        strokewidth=0,
        alpha=0.7,
        color=:blue,
    )

    return ax # Zwróć oś
end


"""
Generuje siatkę statycznych wykresów przestrzeni fazowej
dla podanej listy czasów `times`.
"""
function plot_phase_space_grid(
    simres::SimResult,
    times::AbstractVector{<:Real};
    layout=nothing, # np. (2, 2) dla 4 wykresów
    fig_size=(800, 700),
)
    println("plt.plot_phase_space_grid")
    t_span = simres.settings.tspan
    t0 = t_span[1]

    # --- Ustal układ siatki ---
    n_plots = length(times)
    if isnothing(layout)
        # Automatycznie ustal układ (np. dąż do kwadratu)
        cols = ceil(Int, sqrt(n_plots))
        rows = ceil(Int, n_plots / cols)
        layout = (rows, cols)
    end

    fig = Figure(size=fig_size)

    # --- Oblicz globalne limity dla osi (używa istniejącej funkcji) ---
    println("Obliczanie globalnych limitów dla siatki...")
    xlims, ylims = _calculate_global_limits(simres, t_span)
    if isnothing(xlims)
        println("Warning: Nie można ustalić limitów. Wykresy mogą się różnić.")
        xlims = (0, 1)
        ylims = (-1, 1)
    end

    # --- Tytuł główny figury ---
    fig[0, 1:layout[2]] = Label(
        fig,
        "Phase Space Evolution [$(simres.settings.theory)]",
        fontsize=18,
        font=:bold
    )

    # --- Tworzenie siatki wykresów ---
    plot_count = 1
    for r in 1:layout[1], c in 1:layout[2]
        if plot_count > n_plots
            break # Zakończ, jeśli narysowano wszystkie wykresy
        end

        t = times[plot_count]

        # Stwórz oś (Axis)
        ax = Axis(
            fig[r+1, c], # +1 dla przesunięcia przez tytuł główny
            xlabel=(r == layout[1]) ? L"\tau_0 T" : "", # Etykieta X tylko na dole
            ylabel=(c == 1) ? L"\tau_0^2 \dot{T}" : "",  # Etykieta Y tylko po lewej
            limits=(xlims, ylims),
            xticklabelsize=12,
            yticklabelsize=12,
            xticklabelsvisible=(r == layout[1]), # Ukryj etykiety pośrednie
            yticklabelsvisible=(c == 1),
        )

        # Użyj funkcji pomocniczej do narysowania danych
        plot_phase_space_at_time!(ax, simres, t)

        plot_count += 1
    end

    println("Gotowe. Zwracam figurę.")
    return fig
end

# ==============================================================================
# KONIEC NOWYCH FUNKCJI
# ==============================================================================


function plot_explained_variance_evolution(
    pca_results::Vector{PCAResultAtTime};
    info_text::String="",
)
    if isempty(pca_results)
        @warn "Brak wyników PCA do narysowania wykresu wariancji."
        return Figure()
    end
    # Zwiększa cyfry na osi Y
    taus = [res.tau for res in pca_results]
    n_components = maximum(length(res.explained_variance_ratio) for res in pca_results)

    fig = Figure(size=(1000, 600))
    ax = Axis(
        fig[1, 1],
        title="Ewolucja wariancji wyjaśnionej przez komponenty PCA",
        subtitle=info_text,
        xlabel=L"\tau \text{ [fm/c]}",
        ylabel="Proporcja wyjaśnionej wariancji//Explained Variance EVR",
        limits=(nothing, (0, 1.1)),
        xticklabelsize=14,  # Zwiększa cyfry na osi X
        yticklabelsize=14   # Zwiększa cyfry na osi Y
    )

    for i = 1:n_components
        variance_data = [
            length(res.explained_variance_ratio) >= i ? res.explained_variance_ratio[i] : NaN
            for res in pca_results
        ]
        lines!(ax, taus, variance_data, label="PC $i", linewidth=2.5)
    end

    if n_components > 1
        cumulative_variance = [sum(res.explained_variance_ratio) for res in pca_results]
        lines!(
            ax,
            taus,
            cumulative_variance,
            label="Suma",
            linestyle=:dash,
            color=:black,
            linewidth=2.0
        )
    end

    hlines!(ax, [1.0], linestyle=:dot, color=:grey, alpha=0.7)
    axislegend(ax, position=:rc)
    return fig
end

"""
    plot_pca_snapshot!(ax, pca_result, initial_temps_for_mask; ...)

Wewnętrzna funkcja rysująca pojedynczy wykres PCA na podanej osi (Axis).
"""
function plot_pca_snapshot!(
    ax::Axis,
    pca_result::PCAResultAtTime,
    initial_temps_for_mask::Vector{Float64};
    colormap=:plasma,
    markersize=4,
)
    data = pca_result.transformed_data

    if size(data, 2) < 2
        text!(ax, "Zbyt mało komponentów (N < 2)", position=(0, 0), align=(:center, :center))
        @warn "Pominięto rysowanie dla tau=$(pca_result.tau), liczba komponentów < 2"
        return
    end

    pc1_data = data[:, 1]
    pc2_data = data[:, 2]

    if length(initial_temps_for_mask) == size(data, 1)
        scatter!(
            ax,
            pc1_data,
            pc2_data,
            color=initial_temps_for_mask,
            colormap=colormap,
            markersize=markersize,
            alpha=0.8,
            strokewidth=0,
        )
    else
        @warn "Niezgodność liczby temperatur i punktów danych dla tau=$(pca_result.tau). Używanie domyślnego koloru."
        scatter!(
            ax,
            pc1_data,
            pc2_data,
            color=:blue,
            markersize=markersize,
            alpha=0.8,
            strokewidth=0,
        )
    end
end

"""
    plot_pca_snapshot(sim_result, t, feature_indices, pca_method_params; ...)

Funkcja "publiczna": Uruchamia PCA dla pojedynczego czasu `t` i zwraca gotowy wykres (`Figure`).
"""
function plot_pca_snapshot(
    sim_result::modHydroSim.SimResult,
    t::Float64,
    feature_indices::Vector{Int},
    pca_method_params::Dict;
    info_text::String="",
    n_components::Int=2
)
    println("Uruchamianie PCA dla pojedynczego czasu τ = $t...")
    pca_result = modPCA.run_pca_at_time(
        sim_result,
        t,
        feature_indices,
        n_components,
        pca_method_params
    )

    if isnothing(pca_result)
        @error "Nie można wygenerować PCA dla tau=$t. Zwracanie pustego wykresu."
        return Figure()
    end

    initial_states_raw = [sol.u[1] for sol in sim_result.solutions]
    all_initial_temps = [s[1] for s in initial_states_raw]
    valid_initial_temps = all_initial_temps[pca_result.valid_mask]

    total_var = sum(pca_result.explained_variance_ratio) * 100
    ax_title = "τ = $(round(t, digits=2)) fm/c (Var: $(round(total_var, digits=1))%)"

    fig = Figure(size=(800, 700))
    fig[1, 1] = Label(fig, "Wizualizacja PCA (pojedynczy czas)", fontsize=24, tellwidth=false)
    fig[2, 1] = Label(fig, info_text, fontsize=14, tellwidth=false, padding=(0, 0, 5, 10))

    ax = Axis(
        fig[3, 1],
        title=ax_title,
        xlabel="PC 1",
        ylabel="PC 2",
        xticklabelsize=16,  # Zwiększa cyfry na osi X
        yticklabelsize=16   # Zwiększa cyfry na osi Y
    )

    plot_pca_snapshot!(ax, pca_result, valid_initial_temps)

    if length(all_initial_temps) > 0
        Colorbar(fig[3, 2], colormap=:plasma, label=L"T_0 \text{ [MeV]}")
    end

    return fig
end


function visualize_pca_static_grid(
    pca_results::Vector{PCAResultAtTime},
    sim_result::modHydroSim.SimResult,
    num_plots::Int;
    info_text::String="",
)
    if isempty(pca_results)
        @warn "Brak wyników PCA do wizualizacji."
        return Figure()
    end

    initial_states_raw = [sol.u[1] for sol in sim_result.solutions]
    all_initial_temps = [s[1] for s in initial_states_raw]

    total_steps = length(pca_results)
    indices_to_plot =
        unique(round.(Int, range(1, stop=total_steps, length=num_plots)))

    n_cols = ceil(Int, sqrt(length(indices_to_plot)))
    n_rows = ceil(Int, length(indices_to_plot) / n_cols)

    fig = Figure(size=(500 * n_cols, 450 * n_rows))
    fig[1, 1:n_cols] = Label(fig, "Wizualizacja PCA w przestrzeni komponentów", fontsize=24, tellwidth=false)
    fig[2, 1:n_cols] = Label(fig, info_text, fontsize=14, tellwidth=false, padding=(0, 0, 5, 10))

    all_pc1 = filter(isfinite, vcat([res.transformed_data[:, 1] for res in pca_results if size(res.transformed_data, 2) >= 1]...))
    all_pc2 = filter(isfinite, vcat([res.transformed_data[:, 2] for res in pca_results if size(res.transformed_data, 2) >= 2]...))

    global_xlims = (nothing, nothing)
    global_ylims = (nothing, nothing)

    if !isempty(all_pc1) && !isempty(all_pc2)
        min_pc1, max_pc1 = minimum(all_pc1), maximum(all_pc1)
        min_pc2, max_pc2 = minimum(all_pc2), maximum(all_pc2)
        x_margin = (max_pc1 - min_pc1) * 0.05
        y_margin = (max_pc2 - min_pc2) * 0.05
        global_xlims = (min_pc1 - x_margin, max_pc1 + x_margin)
        global_ylims = (min_pc2 - y_margin, max_pc2 + y_margin)
    else
        @warn "Nie można ustalić globalnych limitów dla siatki PCA. Wykres może być pusty."
    end

    for (i, res_idx) in enumerate(indices_to_plot)
        res = pca_results[res_idx]
        valid_initial_temps = all_initial_temps[res.valid_mask]

        row = (i - 1) ÷ n_cols + 3
        col = (i - 1) % n_cols + 1

        total_var = sum(res.explained_variance_ratio) * 100
        ax_title = "τ = $(round(res.tau, digits=2)) fm/c (Var: $(round(total_var, digits=1))%)"

        ax = Axis(
            fig[row, col],
            title=ax_title,
            xlabel="PC 1",
            ylabel="PC 2",
            limits=(global_xlims, global_ylims),
            xticklabelsize=14,
            yticklabelsize=14
        )

        plot_pca_snapshot!(ax, res, valid_initial_temps)
    end

    if length(all_initial_temps) > 0
        Colorbar(fig[3:n_rows+2, n_cols+1], colormap=:plasma, label=L"T_0 \text{ [MeV]}")
    end

    return fig
end


function plot_loadings_evolution(
    pca_results::Vector{PCAResultAtTime},
    selected_feature_names::Vector{String};
    info_text::String="",
)
    if isempty(pca_results)
        @warn "Brak wyników PCA do narysowania wykresu ładunków."
        return Figure()
    end

    taus = [res.tau for res in pca_results]
    n_components = 0
    if !isempty(pca_results)
        n_components = maximum(size(res.principal_components, 2) for res in pca_results if !isempty(res.principal_components))
    end
    n_features = length(selected_feature_names)

    fig = Figure(size=(1000, 400 * n_components))
    fig[1, 1] = Label(fig, "Ewolucja ładunków (Loadings) PCA", fontsize=24, tellwidth=false)
    fig[2, 1] = Label(fig, info_text, fontsize=14, tellwidth=false, padding=(0, 0, 5, 10))

    for i = 1:n_components
        ax = Axis(
            fig[i+2, 1],
            title="Komponent PC$i",
            xlabel=L"\tau \text{ [fm/c]}",
            ylabel="Wartość ładunku",
            limits=(nothing, (-1.1, 1.1)),
        )

        for j = 1:n_features
            loading_data = [
                (size(res.principal_components, 1) >= j && size(res.principal_components, 2) >= i) ?
                res.principal_components[j, i] : NaN
                for res in pca_results
            ]

            lines!(ax, taus, loading_data, label=selected_feature_names[j], linewidth=2.5)
        end
        hlines!(ax, [0.0], linestyle=:dot, color=:grey, alpha=0.7)
        axislegend(ax, position=:rc, patchsize=(30, 30))
    end

    return fig
end

end # koniec modułu modPlots

