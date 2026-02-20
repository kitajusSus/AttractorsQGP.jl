include("lib.jl")
include("pca.jl")
include("plt.jl")

using .modHydroSim
using .modPCA
using .modPlots
using GLMakie
using CSV
using DataFrames

const FEATS = [("T", :T), ("A", :A), ("dT", :dTdτ), ("dA", :dAdτ), ("τ₀T", :tau0_T), ("τ₀²dT", :tau0sq_dTdτ)]
const AXES = [(:T, "T"), (:A, "A"), (:dT, "dT"), (:dA, "dA"), (:tau0_T, "τ₀T"), (:tau0sq_dT, "τ₀²dT"), (:tau, "τ")]

function prompt_file(dir="datasets")
    !isdir(dir) && return nothing
    files = readdir(dir)
    for (i, f) in enumerate(files)
        println("[$i] $f")
    end
    try
        return joinpath(dir, files[parse(Int64, readline())])
    catch
        return nothing
    end
end

function prompt_axis(lbl)
    println(lbl)
    for (i, v) in enumerate(AXES)
        println("[$i] $(v[2])")
    end
    return AXES[parse(Int64, readline())][1]
end

function save_csv(simres, name)
    df = DataFrame(Run=Int[], Tau=Float64[], T=Float64[], A=Float64[])
    for (i, s) in enumerate(simres.solutions)
        isempty(s.t) && continue
        append!(df, DataFrame(Run=i, Tau=s.t, T=[u[1] for u in s.u], A=[u[2] for u in s.u]))
    end
    CSV.write(name, df)
    println("Saved $name")
end

function kadr(simres, t)
    vals, _, mask = modHydroSim.extract_phase_space_slice(simres, Float64(t))

    if !any(mask)
        println("Brak danych dla czasu t=$t (symulacja mogła się skończyć wcześniej)")
        return
    end

    Ts = vals[1][mask]
    As = vals[2][mask]

    fig = Figure(size=(800, 600))

    ax = Axis(fig[1, 1],
        title="Niezależne stany ewolucji plazmy dla różnych par warunków początkowych  w  [T(τ),A(τ)] τ = $t fm/c ",
        xlabel="Temperatura T [MeV]",
        ylabel="Anizotropia A",
        titlesize=24,
        xlabelsize=18,
        ylabelsize=18
    )

    scatter!(ax, Ts, As,
        markersize=8,
        color=:blue,
        strokewidth=1,
        strokecolor=:black,
        label="Stan Ewolucji dla danych warunków"
    )

    axislegend(ax)

    display(fig)
end

function plot_universal_grid(simres, times, x_key, y_key)

    x_label = try
        modPlots.resolve_def(x_key)[1]
    catch
        string(x_key)
    end
    y_label = try
        modPlots.resolve_def(y_key)[1]
    catch
        string(y_key)
    end

    n = length(times)
    n_cols = ceil(Int, sqrt(n))
    n_rows = ceil(Int, n / n_cols)

    fig = Figure(size=(400 * n_cols, 350 * n_rows))

    Label(fig[0, :], "Zależność $y_label od $x_label", fontsize=20, font=:bold)

    axes_list = []

    for (i, t) in enumerate(times)
        row = (i - 1) ÷ n_cols + 1
        col = (i - 1) % n_cols + 1

        ax = Axis(fig[row, col],
            title="τ = $t fm/c",
            xlabel=x_label,
            ylabel=y_label,
            xlabelsize=16,
            ylabelsize=16
        )
        push!(axes_list, ax)

        vx, mask_x = modPlots.get_data(simres, Float64(t), x_key)
        vy, mask_y = modPlots.get_data(simres, Float64(t), y_key)

        mask = mask_x .& mask_y

        if any(mask)
            scatter!(ax, vx[mask], vy[mask],
                markersize=5,
                color=:dodgerblue,
                strokewidth=0.5,
                strokecolor=:black,
                alpha=0.7
            )
        else
            text!(ax, 0.5, 0.5, text="Brak danych", space=:relative, align=(:center, :center))
        end

        if row < n_rows
            hidexdecorations!(ax, grid=false)
        end
        if col > 1
            hideydecorations!(ax, grid=false)
        end
    end

    # linkaxes!(axes_list...)

    display(fig)
    return fig
end
function kadr_grid(simres, times)
    T_min, T_max = simres.settings.T_range
    A_min, A_max = simres.settings.A_range

    n = length(times)
    n_cols = ceil(Int, sqrt(n))
    n_rows = ceil(Int, n / n_cols)

    fig = Figure(size=(350 * n_cols, 300 * n_rows))

    axes_list = []

    for (i, t) in enumerate(times)
        row = (i - 1) ÷ n_cols + 1
        col = (i - 1) % n_cols + 1

        ax = Axis(fig[row, col],
            title="τ = $t fm/c",
            xlabel="T [MeV]",
            ylabel="A",
            #limits=(T_min, T_max, A_min, A_max), # <--- TO JEST KLUCZ DO STAŁYCH OSI
            xlabelsize=14,
            ylabelsize=14
        )

        if row < n_rows
            hidexdecorations!(ax, grid=false) # Ukryj X (liczy i label) jeśli nie jest na dole
        end
        if col > 1
            hideydecorations!(ax, grid=false) # Ukryj Y (liczby i label) jeśli nie jest z lewej
        end

        push!(axes_list, ax)

        vals, _, mask = modHydroSim.extract_phase_space_slice(simres, Float64(t))
        if any(mask)
            Ts = vals[1][mask]
            As = vals[2][mask]
            scatter!(ax, Ts, As, markersize=4, color=:blue, strokewidth=0.5, strokecolor=:black)
        else
            text!(ax, (T_min + T_max) / 2, (A_min + A_max) / 2, text="Brak danych", align=(:center, :center))
        end
    end

    linkaxes!(axes_list...)

    Label(fig[0, :], "Ewolucja w przestrzeni fazowej", fontsize=18, font=:bold)

    display(fig)
end

function plot_trajectories(simres, var_key)
    vars = Dict(
        :T => (1, L"T \, [\text{fm}^{-1}]"),
        :A => (2, L"\mathcal{A}")
    )

    if !haskey(vars, var_key)
        println("Nieznana zmienna. Dostępne opcje: :T, :A")
        return nothing
    end

    idx, y_label = vars[var_key]

    fig = Figure(size=(900, 600), fontsize=20)
    ax = Axis(fig[1, 1],
        title=L"Zbieganie trajektorii %$(y_label) \text{ do atraktora}",
        xlabel=L"\tau \, [\text{fm}/c]",
        ylabel=y_label,
        limits=(0.2, 1.2, 1, 10),
        titlesize=24,
        xlabelsize=22,
        ylabelsize=22
    )

    n_sols = length(simres.solutions)
    step = max(1, n_sols ÷ 500)

    println("Rysowanie $n_sols trajektorii (krok: $step)...")

    for i in 1:step:n_sols
        sol = simres.solutions[i]
        if !isempty(sol.t)
            values = [u[idx] for u in sol.u]

            lines!(ax, sol.t, values,
                color=(:dodgerblue, 0.2),
                linewidth=1.5
            )
        end
    end

    if var_key == :A
        hlines!(ax, [0.0], color=:red, linestyle=:dash, linewidth=2, label=L"\mathcal{A}=0 \text{ (Izotropia)}")
        axislegend(ax)
    end

    # ax.xscale = log10

    return fig
end

function main()
    file = prompt_file()
    isnothing(file) && return

    sets = load_simulation_settings(file)
    sim = run_simulation(settings=sets, ic_file=file)

    while true
        println("\n[1] PCA Custom")
        println("[2] Wykresy Grid")
        println("[3] Animacja")
        println("[5] Ewolucja T,A + Chmura A vs T + CSV")
        println("[6] kadry dla danego czasu")
        println("[7] kadr grid ")
        println("[10] Ewolucja dla A lub T w τ")
        println("[q] Quit")

        c = readline()
        if c == "1"
            println("Cechy (indeksy):")
            for (i, f) in enumerate(FEATS)
                println("[$i] $(f[1])")
            end
            ids = [parse(Int64, x) for x in split(readline(), ",")]
            fs = [FEATS[i][2] for i in ids]
            println("Kroki: ")
            n = parse(Int64, readline())
            res = modPCA.run_pca_over_time(sim, fs, n, 2, Dict(:method => :standardize))
            save("pca_var.png", modPlots.plot_explained_variance_evolution(res))
            save("pca_grid.png", modPlots.visualize_pca_static_grid(res, sim, 50))
            # save("pca_load.png", modPlots.plot_loadings_evolution(res, [FEATS[i][1] for i in ids]))
        elseif c == "2"
            x, y = prompt_axis("X:"), prompt_axis("Y:")
            println("Czasy: ")
            ts = [parse(Float64, t) for t in split(readline())]
            save("grid.png", modPlots.plot_phase_space_grid(sim, ts, x, y))
        elseif c == "3"
            x, y = prompt_axis("X:"), prompt_axis("Y:")
            modPlots.animate_phase_space_evolution(sim, x, y)
        elseif c == "4"
            println("\n--- Uniwersalny Wykres ---")
            x = prompt_axis("Wybierz zmienną na Oś X:")
            y = prompt_axis("Wybierz zmienną na Oś Y:")
            println("Podaj czasy oddzielone spacją (np. 0.2 0.6 1.0): ")
            input_t = readline()
            if !isempty(input_t)
                ts = [parse(Float64, t) for t in split(input_t)]
                fig = plot_universal_grid(sim, ts, x, y)
                println("Zapisać wykres? [t/n]")
                if readline() == "t"
                    save("universal_plot_$(x)_vs_$(y).png", fig)
                    println("Zapisano jako universal_plot_$(x)_vs_$(y).png")
                end
            end

        elseif c == "5"
            println("Rysuje trajektorie...")
            save("traj.png", modPlots.plot_thermodynamics_evolution(sim))
            println("Zapisuje CSV...")
            save_csv(sim, "data.csv")
            println("Chmura A vs T. Czasy: ")
            ts = [parse(Float64, t) for t in split(readline())]
            save("cloud_grid.png", modPlots.plot_phase_space_grid(sim, ts, :T, :A))
            for t in ts
                save("cloud_$t.png", modPlots.plot_phase_space_snapshot(sim, t, :T, :A))
            end
        elseif c == "6"
            println("Podaj czas (tau): ")
            t_val = parse(Float64, readline())
            kadr(sim, t_val)

        elseif c == "7"
            println("Podaj czasy oddzielone spacją (np. 0.5 1.0 2.0 5.0): ")
            ts = [parse(Float64, t) for t in split(readline())]
            kadr_grid(sim, ts)
        elseif c == "10" # <--- OBSŁUGA NOWEJ OPCJI
            println("Wybierz zmienną do wykreślenia:")
            println("[1] Temperatura T(τ)")
            println("[2] Anizotropia A(τ)")
            choice = readline()

            var_sym = choice == "1" ? :T : (choice == "2" ? :A : nothing)

            if !isnothing(var_sym)
                fig = plot_trajectories(sim, var_sym)
                display(fig)
                println("Zapisać wykres? [t/n]")
                if readline() == "t"
                    save("trajectory_$(var_sym).png", fig)
                    println("Zapisano jako trajectory_$(var_sym).png")
                end
            else
                println("Niepoprawny wybór.")
            end


        elseif c == "q"
            break
        end
    end
end
