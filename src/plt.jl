include("lib.jl")
include("pca.jl")

module modPlots

using GLMakie
using LaTeXStrings
using ..modHydroSim
using ..modPCA

set_theme!(theme_latexfonts())

export plot_phase_space_grid, plot_phase_space_snapshot,
       plot_thermodynamics_evolution, animate_phase_space_evolution,
       plot_explained_variance_evolution, visualize_pca_static_grid,
       plot_loadings_evolution

const PLOT_KEYS = Dict(
    :T => (L"T", u->u[1], (u,du,t,t0)->u[1]),
    :A => (L"A", u->u[2], (u,du,t,t0)->u[2]),
    :dT => (L"\dot{T}", nothing, (u,du,t,t0)->du[1]),
    :dA => (L"\dot{A}", nothing, (u,du,t,t0)->du[2]),
    :tau0_T => (L"\tau_0 T", nothing, (u,du,t,t0)->t0*u[1]),
    :tau0sq_dT => (L"\tau_0^2 \dot{T}", nothing, (u,du,t,t0)->t0^2*du[1]),
    :tau_T => (L"\tau T", nothing, (u,du,t,t0)->t*u[1])
)

function get_data(simres, t, key)
    u, du, mask = modHydroSim.extract_phase_space_slice(simres, t)
    if !any(mask); return Float64[], mask; end
    t0 = simres.settings.tspan[1]

    func = PLOT_KEYS[key][3]
    vals = func(u, du, t, t0)
    return vals, mask
end

function _limits(simres, ts, kx, ky)
    X, Y = Float64[], Float64[]
    for t in range(ts..., length=10)
        vx, mx = get_data(simres, t, kx)
        vy, my = get_data(simres, t, ky)
        if any(mx)
            append!(X, vx[mx])
            append!(Y, vy[my])
        end
    end
    isempty(X) ? (nothing, nothing) : ((minimum(X), maximum(X)), (minimum(Y), maximum(Y)))
end

function plot_phase_space_grid(simres, times, kx, ky)
    n = length(times)
    c = ceil(Int, sqrt(n))
    r = ceil(Int, n/c)
    fig = Figure(size=(400*c, 350*r))

    xl, yl = _limits(simres, simres.settings.tspan, kx, ky)
    lbl_x, lbl_y = PLOT_KEYS[kx][1], PLOT_KEYS[ky][1]

    for (i, t) in enumerate(times)
        row, col = (i-1)÷c + 1, (i-1)%c + 1
        ax = Axis(fig[row, col], title="τ=$(round(t, digits=2))", xlabel=lbl_x, ylabel=lbl_y)
        !isnothing(xl) && limits!(ax, xl, yl)

        vx, mx = get_data(simres, t, kx)
        vy, my = get_data(simres, t, ky)
        any(mx) && scatter!(ax, vx[mx], vy[my], markersize=4, color=:blue, alpha=0.6)
    end
    return fig
end

function plot_phase_space_snapshot(simres, t, kx, ky)
    fig = Figure()
    ax = Axis(fig[1,1], title="τ=$t", xlabel=PLOT_KEYS[kx][1], ylabel=PLOT_KEYS[ky][1])
    vx, mx = get_data(simres, t, kx)
    vy, my = get_data(simres, t, ky)
    any(mx) && scatter!(ax, vx[mx], vy[my], markersize=4, color=:blue)
    return fig
end

function animate_phase_space_evolution(simres, kx, ky; out="anim.gif")
    ts = range(simres.settings.tspan..., length=100)
    xl, yl = _limits(simres, simres.settings.tspan, kx, ky)
    fig = Figure()
    ax = Axis(fig[1,1], title="Ewolucja", xlabel=PLOT_KEYS[kx][1], ylabel=PLOT_KEYS[ky][1])
    !isnothing(xl) && limits!(ax, xl, yl)

    t_node = Observable(ts[1])
    pts = @lift begin
        vx, mx = get_data(simres, $t_node, kx)
        vy, my = get_data(simres, $t_node, ky)
        Point2f.(vx[mx], vy[my])
    end
    scatter!(ax, pts, color=:blue, markersize=4)
    record(fig, out, ts) do t; t_node[] = t; end
end

function plot_thermodynamics_evolution(simres)
    fig = Figure(size=(1000, 500))
    ax1 = Axis(fig[1,1], title="T(τ)", xlabel=L"\tau", ylabel=L"T")
    ax2 = Axis(fig[1,2], title="A(τ)", xlabel=L"\tau", ylabel=L"A")

    for sol in simres.solutions
        isempty(sol.t) && continue
        lines!(ax1, sol.t, [u[1] for u in sol.u], alpha=0.3, color=(:red, 0.5))
        lines!(ax2, sol.t, [u[2] for u in sol.u], alpha=0.3, color=(:blue, 0.5))
    end
    return fig
end

function plot_explained_variance_evolution(res)
    fig = Figure()
    ax = Axis(fig[1,1], title="PCA Variance", xlabel=L"\tau", ylabel="EVR")
    ts = [r.tau for r in res]
    for i in 1:length(res[1].explained_variance_ratio)
        lines!(ax, ts, [r.explained_variance_ratio[i] for r in res], label="PC$i")
    end
    axislegend(ax)
    return fig
end

function visualize_pca_static_grid(res, simres, n)
    idx = round.(Int, range(1, length(res), length=n))
    c = ceil(Int, sqrt(n))
    r = ceil(Int, n/c)
    fig = Figure(size=(300*c, 300*r))

    t0 = [s.u[1][1] for s in simres.solutions]

    for (i, id) in enumerate(idx)
        rr = res[id]
        row, col = (i-1)÷c + 1, (i-1)%c + 1
        ax = Axis(fig[row,col], title="τ=$(round(rr.tau, digits=2))", xlabel="PC1", ylabel="PC2")
        scatter!(ax, rr.transformed_data[:,1], rr.transformed_data[:,2],
                 color=t0[rr.valid_mask], colormap=:plasma, markersize=4)
    end
    return fig
end

function plot_loadings_evolution(res, names)
    fig = Figure()
    ts = [r.tau for r in res]
    for i in 1:2
        ax = Axis(fig[i,1], title="Loadings PC$i", xlabel=L"\tau")
        for (j, n) in enumerate(names)
            lines!(ax, ts, [r.principal_components[j,i] for r in res], label=n)
        end
        axislegend(ax)
    end
    return fig
end

end
