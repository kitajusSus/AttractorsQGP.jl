module modPlots

using CairoMakie
using LaTeXStrings
using ColorSchemes
using ..modHydroSim
using ..modPCA

# --- KONFIGURACJA STYLU PUBLIKACYJNEGO ---
function set_publication_theme()
    set_theme!(Theme(
        font = "TeX Gyre Heros", # Lub "Latin Modern Roman"
        fontsize = 16,
        Axis = (
            titlesize = 18,
            xlabelsize = 18,
            ylabelsize = 18,
            xticklabelsize = 14,
            yticklabelsize = 14,
            backgroundcolor = :white,
            xgridstyle = :dash,
            ygridstyle = :dash,
            xgridcolor = RGBAf(0.8, 0.8, 0.8, 0.5),
            ygridcolor = RGBAf(0.8, 0.8, 0.8, 0.5),
            spinewidth = 1.2,
            xtickwidth = 1.2,
            ytickwidth = 1.2,
            topspinevisible = true,
            rightspinevisible = true
        ),
        Legend = (
            framevisible = true,
            framewidth = 0.8,
            backgroundcolor = :white, # POPRAWKA: bgcolor -> backgroundcolor
            position = :rt
        ),
        Palette = (
            color = Makie.wong_colors(),
        )
    ))
end

# Uruchomienie stylu przy ładowaniu
set_publication_theme()

export plot_phase_space_grid, plot_phase_space_snapshot,
       plot_thermodynamics_evolution, animate_phase_space_evolution,
       plot_explained_variance_evolution, visualize_pca_static_grid,
       plot_loadings_evolution

# --- DEFINICJE STANDARDOWE ---
const PLOT_KEYS = Dict(
    :T => (L"T \, [\text{fm}^{-1}]", u->u[1], (u,du,t,t0)->u[1]),
    :A => (L"\mathcal{A}", u->u[2], (u,du,t,t0)->u[2]),
    :dT => (L"\dot{T} \, [\text{fm}^{-2}]", nothing, (u,du,t,t0)->du[1]),
    :dA => (L"\dot{\mathcal{A}}", nothing, (u,du,t,t0)->du[2]),
    :tau0_T => (L"\tau_0 T", nothing, (u,du,t,t0)->t0*u[1]),
    :tau0sq_dT => (L"\tau_0^2 \dot{T}", nothing, (u,du,t,t0)->t0^2*du[1]),
    :tau_T => (L"\tau T", nothing, (u,du,t,t0)->t*u[1]),
    :tauSq_dT => (L"\tau^2 \dot{T}", nothing, (u,du,t,t0)->t^2*du[1])
)

# --- MECHANIZM ROZWIĄZYWANIA DEFINICJI (Symbol lub Krotka) ---
function resolve_def(input)
    if input isa Symbol
        if !haskey(PLOT_KEYS, input)
            error("Klucz :$input nie istnieje w PLOT_KEYS. Użyj krotki (Label, Func) dla własnej definicji.")
        end
        return PLOT_KEYS[input]
    elseif input isa Tuple && length(input) == 2
        # Wersja skrócona: ("Label", func(u,du,t,t0)) -> dodajemy nothing w środek
        return (input[1], nothing, input[2])
    elseif input isa Tuple && length(input) == 3
        return input
    else
        error("Niepoprawna definicja osi. Oczekiwano Symbol lub Tuple.")
    end
end

# --- POBIERANIE DANYCH ---
function get_data(simres, t, key_or_def)
    u, du, mask = modHydroSim.extract_phase_space_slice(simres, t)
    if !any(mask); return Float64[], mask; end
    t0 = simres.settings.tspan[1]

    # Rozwiązujemy definicję (czy to Symbol czy manualna funkcja)
    def = resolve_def(key_or_def)
    func = def[3]

    vals = func(u, du, t, t0)
    return vals, mask
end

function _limits(simres, ts, def_x, def_y)
    X, Y = Float64[], Float64[]
    for t in range(ts..., length=10)
        # Przekazujemy już rozwiązane definicje
        vx, mx = get_data(simres, t, def_x)
        vy, my = get_data(simres, t, def_y)
        if any(mx)
            append!(X, vx[mx])
            append!(Y, vy[my])
        end
    end
    if isempty(X)
        return (nothing, nothing)
    else
        dx, dy = maximum(X) - minimum(X), maximum(Y) - minimum(Y)
        return (minimum(X) - 0.05*dx, maximum(X) + 0.05*dx),
               (minimum(Y) - 0.05*dy, maximum(Y) + 0.05*dy)
    end
end

# --- FUNKCJE RYSOWANIA ---

function plot_phase_space_grid(simres, times, def_input_x, def_input_y;
                               colorize_by_t0=true,
                               rasterize_factor=5,
                               shared_axes=false)

    # Rozwiązujemy definicje (Symbol -> Definicja lub Tuple -> Definicja)
    def_x = resolve_def(def_input_x)
    def_y = resolve_def(def_input_y)
    lbl_x, lbl_y = def_x[1], def_y[1]

    n = length(times)
    c = n > 3 ? 3 : n
    r = ceil(Int, n/c)

    fig = Figure(size=(350*c, 300*r * 0.9))

    xl, yl = nothing, nothing
    if shared_axes
        xl, yl = _limits(simres, simres.settings.tspan, def_x, def_y)
    end

    t0_vals = [s.u[1][1] for s in simres.solutions]
    axes = []

    for (i, t) in enumerate(times)
        row, col = (i-1)÷c + 1, (i-1)%c + 1

        # Jeśli osie nie są wspólne, pokazujemy etykiety wszędzie
        show_xlabel = shared_axes ? (i > (r-1)*c) : true
        show_ylabel = shared_axes ? (col == 1) : true

        ax = Axis(fig[row, col],
                  title=L"\tau = %$(round(t, digits=2)) \, \text{fm}/c",
                  xlabel=show_xlabel ? lbl_x : "",
                  ylabel=show_ylabel ? lbl_y : "")

        push!(axes, ax)

        if shared_axes && !isnothing(xl)
            limits!(ax, xl, yl)
        end

        vx, mx = get_data(simres, t, def_x)
        vy, my = get_data(simres, t, def_y)

        if any(mx)
            color_data = colorize_by_t0 ? t0_vals[mx] : :midnightblue
            scatter!(ax, vx[mx], vy[my],
                     markersize=3,
                     color=color_data, colormap=:magma, alpha=0.7,
                     rasterize=rasterize_factor)
        end
    end

    if colorize_by_t0
        Colorbar(fig[:, c+1], label=L"T(\tau_0)", colormap=:magma,
                 limits=(minimum(t0_vals), maximum(t0_vals)))
    end

    if shared_axes
        linkaxes!(axes...)
    end

    return fig
end

function plot_phase_space_snapshot(simres, t, kx, ky)
    def_x, def_y = resolve_def(kx), resolve_def(ky)
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1,1],
              title=L"\text{Snapshot at } \tau = %$t \, \text{fm}/c",
              xlabel=def_x[1],
              ylabel=def_y[1])

    vx, mx = get_data(simres, t, def_x)
    vy, my = get_data(simres, t, def_y)

    if any(mx)
        scatter!(ax, vx[mx], vy[my], markersize=3, color=(:royalblue, 0.6), rasterize=5)
    end
    return fig
end

# --- PRZYWRÓCONA FUNKCJA (Brakowało jej w Twoim kodzie) ---
function visualize_pca_static_grid(res, simres, n)
    idx = round.(Int, range(1, length(res), length=n))
    c = ceil(Int, sqrt(n))
    r = ceil(Int, n/c)
    fig = Figure(size=(300*c + 100, 300*r))

    t0 = [s.u[1][1] for s in simres.solutions]

    for (i, id) in enumerate(idx)
        rr = res[id]
        row, col = (i-1)÷c + 1, (i-1)%c + 1
        ax = Axis(fig[row,col],
                  title=L"\tau = %$(round(rr.tau, digits=2))",
                  xlabel=L"\text{PC}_1",
                  ylabel=L"\text{PC}_2")

        sc = scatter!(ax, rr.transformed_data[:,1], rr.transformed_data[:,2],
                 color=t0[rr.valid_mask],
                 colormap=:viridis,
                 markersize=3,
                 rasterize=4)

        if i == n
            Colorbar(fig[:, c+1], sc, label=L"T(\tau_0)")
        end
    end
    return fig
end

function plot_explained_variance_evolution(res)
    fig = Figure(size=(800, 500))
    ax = Axis(fig[1,1],
              title="Wariancja wyjaśniona (EVR) w funkcji czasu",
              xlabel=L"\tau \, [\text{fm}/c]",
              ylabel=L"\text{EVR}",
        limits = (0.2,1.5,0.0,1.0)
              )



    ts = [r.tau for r in res]
    hlines!(ax, [1.0], color=:gray50, linestyle=:dash, label="100%")
    colors = Makie.wong_colors()

    for i in 1:length(res[1].explained_variance_ratio)
        vals = [r.explained_variance_ratio[i] for r in res]
        lines!(ax, ts, vals, label="PC$i", linewidth=3, color=colors[i])

        if i == 1
            band!(ax, ts, zeros(length(ts)), vals, color=(colors[i], 0.1))
        end
    end

    axislegend(ax, position=:rb)
    return fig
end

function plot_loadings_evolution(res, names)
    fig = Figure(size=(800, 600))
    ts = [r.tau for r in res]
    colors = Makie.wong_colors()

    for i in 1:2
        ax = Axis(fig[i,1],
                  title="Wkłady cech do PC$i",
                  xlabel=i==2 ? L"\tau \, [\text{fm}/c]" : "",
                  ylabel="Wartość ładunku")

        hlines!(ax, [0.0], color=:black, linewidth=1)

        for (j, n) in enumerate(names)
            # Obsługa manualnych nazw lub symboli
            key_sym = Symbol(n)
            lbl = ""
            if haskey(PLOT_KEYS, key_sym)
                # Makie czasami ma problem z LaTeXStrings w legendzie przy skomplikowanych typach
                # Tutaj konwertujemy na string, jeśli trzeba, ale L"" zazwyczaj działa
                lbl = string(n)
            else
                lbl = string(n)
            end

            lines!(ax, ts, [r.principal_components[j,i] for r in res],
                   label=lbl, linewidth=2.5, color=colors[j])
        end
        if i == 1
            axislegend(ax, "Zmienne", position=:rt)
        end
    end
    return fig
end

function animate_phase_space_evolution(simres, kx, ky; out="anim.gif")
    # Rozwiązujemy definicje
    def_x, def_y = resolve_def(kx), resolve_def(ky)
    ts = range(simres.settings.tspan..., length=100)

    xl, yl = _limits(simres, simres.settings.tspan, def_x, def_y)

    fig = Figure(size=(800,600))
    ax = Axis(fig[1,1],
              title="Ewolucja w czasie rzeczywistym",
              xlabel=def_x[1],
              ylabel=def_y[1])

    !isnothing(xl) && limits!(ax, xl, yl)

    t_node = Observable(ts[1])

    pts = @lift begin
        vx, mx = get_data(simres, $t_node, def_x)
        vy, my = get_data(simres, $t_node, def_y)
        Point2f.(vx[mx], vy[my])
    end

    scatter!(ax, pts, color=:dodgerblue, markersize=3, alpha=0.6)


    label_text = @lift(latexstring("\$\\tau = $(round($t_node, digits=2))\$"))

    text!(ax, 0.05, 0.95, text=label_text,
          space=:relative, fontsize=22, align=(:left, :top))

    record(fig, out, ts; framerate=15) do t
        t_node[] = t
    end
end

end # Koniec modułu
