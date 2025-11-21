include("lib.jl")
include("pca.jl")

module modPlots

using GLMakie
using LaTeXStrings
using ..modHydroSim
using ..modPCA

set_theme!(theme_latexfonts())

export plot_phase_space_snapshot, plot_phase_space_grid,
       plot_explained_variance_evolution, visualize_pca_static_grid,
       plot_loadings_evolution, plot_pca_snapshot, animate_phase_space_evolution,
       plot_thermodynamics_evolution,
       AVAILABLE_PLOT_KEYS, get_axis_label

# --- Definitions of available variables ---
const AVAILABLE_PLOT_KEYS = Dict(
    :T => (label=L"T \text{ [MeV]}", desc="Temperatura (MeV)"),
    :A => (label=L"A", desc="Anizotropia"),
    :dT => (label=L"\dot{T}", desc="Pochodna Temp."),
    :dA => (label=L"\dot{A}", desc="Pochodna Aniz."),
    :tau0_T => (label=L"\tau_0 T", desc="Skalowana Temp. (Attractor)"),
    :tau0sq_dT => (label=L"\tau_0^2 \dot{T}", desc="Skalowana Poch. (Attractor)"),
    :tau_T => (label=L"\tau T", desc="w = tau * T")
)

function get_axis_label(key::Symbol)
    return haskey(AVAILABLE_PLOT_KEYS, key) ? AVAILABLE_PLOT_KEYS[key].label : string(key)
end

# Helper to get values for a specific key
function _get_values_for_key(simres, t, key::Symbol)
    u, du, mask = modHydroSim.extract_phase_space_slice(simres, t)
    if !any(mask)
        return Float64[], mask
    end

    t0 = simres.settings.tspan[1]
    MeV_conversion = 197.327

    vals = if key == :T
        u[1] .* MeV_conversion
    elseif key == :A
        u[2]
    elseif key == :dT
        du[1] .* MeV_conversion
    elseif key == :dA
        du[2]
    elseif key == :tau0_T
        t0 .* u[1]
    elseif key == :tau0sq_dT
        (t0^2) .* du[1]
    elseif key == :tau_T
        t .* u[1]
    else
        error("Unknown plot key: $key")
    end

    return vals, mask
end

function _calc_limits(simres, trange, x_key, y_key)
    all_x, all_y = Float64[], Float64[]
    steps = range(trange..., length=15)

    for t in steps
        vx, mx = _get_values_for_key(simres, t, x_key)
        vy, my = _get_values_for_key(simres, t, y_key)
        if any(mx)
            append!(all_x, vx[mx])
            append!(all_y, vy[my])
        end
    end

    if isempty(all_x) || isempty(all_y)
        return (nothing, nothing)
    end
    return (minimum(all_x), maximum(all_x)), (minimum(all_y), maximum(all_y))
end

# --- Snapshot Drawing ---
function plot_phase_space_snapshot!(ax, simres, t, x_key, y_key)
    vx, mx = _get_values_for_key(simres, t, x_key)
    vy, my = _get_values_for_key(simres, t, y_key)

    if !any(mx)
        text!(ax, "Brak danych", position=(0,0))
        return
    end

    scatter!(ax, vx[mx], vy[my],
             markersize=4, color=:blue, alpha=0.6)
end

function plot_phase_space_grid(simres, times, x_key, y_key; layout=nothing)
    n = length(times)
    cols = ceil(Int, sqrt(n))
    rows = ceil(Int, n/cols)
    fig = Figure(size=(400*cols, 350*rows))

    xlims, ylims = _calc_limits(simres, simres.settings.tspan, x_key, y_key)
    xl, yl = get_axis_label(x_key), get_axis_label(y_key)

    Label(fig[0, :], "Ewolucja chmury punktów: $yl vs $xl", fontsize=20, font=:bold)

    for (i, t) in enumerate(times)
        r, c = (i-1) ÷ cols + 1, (i-1) % cols + 1
        ax = Axis(fig[r, c], title="τ=$(round(t, digits=2)) fm/c", xlabel=xl, ylabel=yl)

        if !isnothing(xlims)
            wx, wy = xlims[2]-xlims[1], ylims[2]-ylims[1]
            limits!(ax, xlims[1]-0.05wx, xlims[2]+0.05wx, ylims[1]-0.05wy, ylims[2]+0.05wy)
        end
        plot_phase_space_snapshot!(ax, simres, t, x_key, y_key)
    end
    return fig
end

function plot_phase_space_snapshot(simres, t, x_key, y_key)
    fig = Figure(size=(800, 600))
    xl, yl = get_axis_label(x_key), get_axis_label(y_key)
    ax = Axis(fig[1,1], title="Snapshot τ=$(round(t, digits=2)) fm/c", xlabel=xl, ylabel=yl)
    plot_phase_space_snapshot!(ax, simres, t, x_key, y_key)
    return fig
end

# --- Animation ---
function animate_phase_space_evolution(simres, x_key, y_key; output_filename="anim.gif", fps=20)
    times = range(simres.settings.tspan..., length=100)
    xlims, ylims = _calc_limits(simres, simres.settings.tspan, x_key, y_key)
    xl, yl = get_axis_label(x_key), get_axis_label(y_key)

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1,1], title="Ewolucja", xlabel=xl, ylabel=yl)
    if !isnothing(xlims)
        wx, wy = xlims[2]-xlims[1], ylims[2]-ylims[1]
        limits!(ax, xlims[1]-0.05wx, xlims[2]+0.05wx, ylims[1]-0.05wy, ylims[2]+0.05wy)
    end

    t_obs = Observable(times[1])
    pts = @lift begin
        vx, mx = _get_values_for_key(simres, $t_obs, x_key)
        vy, my = _get_values_for_key(simres, $t_obs, y_key)
        Point2f.(vx[mx], vy[mx])
    end

    scatter!(ax, pts, color=:blue, markersize=4)
    record(fig, output_filename, times; framerate=fps) do t
        t_obs[] = t
    end
    println("Zapisano: $output_filename")
end

# --- Thermo Evolution Plot (Option 5) ---
function plot_thermodynamics_evolution(simres)
    fig = Figure(size=(1200, 600))

    ax_T = Axis(fig[1, 1], title="Ewolucja Temperatury T(τ) (Wszystkie punkty)", xlabel=L"\tau \text{ [fm/c]}", ylabel=L"T \text{ [MeV]}")
    ax_A = Axis(fig[1, 2], title="Ewolucja Anizotropii A(τ) (Wszystkie punkty)", xlabel=L"\tau \text{ [fm/c]}", ylabel=L"A")

    # Plot ALL trajectories (no step skipping)

    for sol in simres.solutions
        if isempty(sol.t) || any(isnan, sol.u[end]); continue; end
        ts = sol.t
        Ts = [u[1] for u in sol.u]
        As = [u[2] for u in sol.u]

        lines!(ax_T, ts, Ts, alpha=0.3, linewidth=0.5, color=(:red, 0.5))
        lines!(ax_A, ts, As, alpha=0.3, linewidth=0.5, color=(:blue, 0.5))
    end
    return fig
end

# --- PCA Functions ---
function plot_explained_variance_evolution(results; info_text="")
    fig = Figure(size=(800,500))
    ax = Axis(fig[1,1], title="Wariancja wyjaśniona $info_text", xlabel=L"\tau", ylabel="EVR")
    taus = [r.tau for r in results]
    if isempty(taus); return fig; end
    n_comp = length(results[1].explained_variance_ratio)
    for i in 1:n_comp
        lines!(ax, taus, [r.explained_variance_ratio[i] for r in results], label="PC$i")
    end
    lines!(ax, taus, [sum(r.explained_variance_ratio) for r in results], label="Suma", linestyle=:dash)
    axislegend(ax)
    return fig
end

function visualize_pca_static_grid(results, simres, n_plots; info_text="")
    idxs = round.(Int, range(1, length(results), length=n_plots))
    cols = ceil(Int, sqrt(n_plots))
    rows = ceil(Int, n_plots/cols)
    fig = Figure(size=(300*cols, 300*rows))
    Label(fig[0, :], info_text, fontsize=14)
    t0_temps = [sol.u[1][1] for sol in simres.solutions]
    for (i, idx) in enumerate(idxs)
        res = results[idx]
        r, c = (i-1)÷cols + 1, (i-1)%cols + 1
        ax = Axis(fig[r,c], title="τ=$(round(res.tau, digits=2))", xlabel="PC1", ylabel="PC2")
        if size(res.transformed_data, 2) >= 2
            colors = t0_temps[res.valid_mask]
            scatter!(ax, res.transformed_data[:,1], res.transformed_data[:,2],
                     color=colors, colormap=:plasma, markersize=4)
        end
    end
    return fig
end

function plot_loadings_evolution(results, names; info_text="")
    fig = Figure(size=(800, 600))
    n_comps = size(results[1].principal_components, 2)
    taus = [r.tau for r in results]
    for i in 1:min(n_comps, 2)
        ax = Axis(fig[i, 1], title="Loadings PC$i", xlabel=L"\tau")
        for (j, name) in enumerate(names)
            vals = [r.principal_components[j, i] for r in results]
            lines!(ax, taus, vals, label=string(name))
        end
        axislegend(ax)
    end
    return fig
end

function plot_pca_snapshot(simres, t, idxs, params; info_text="", n_components=2)
    res = modPCA.run_pca_at_time(simres, t, idxs, n_components, params)
    if isnothing(res); return Figure(); end
    fig = Figure()
    ax = Axis(fig[1,1], title="PCA τ=$t", xlabel="PC1", ylabel="PC2")
    t0_temps = [s.u[1][1] for s in simres.solutions]
    colors = t0_temps[res.valid_mask]
    scatter!(ax, res.transformed_data[:,1], res.transformed_data[:,2],
             color=colors, colormap=:plasma)
    return fig
end

end
