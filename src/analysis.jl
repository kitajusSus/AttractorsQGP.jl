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
const AXES = [(:T, "T"), (:A, "A"), (:dT, "dT"), (:dA, "dA"), (:tau0_T, "τ₀T"), (:tau0sq_dT, "τ₀²dT")]

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

    # Wyciągamy T i A tylko dla aktywnych symulacji
    Ts = vals[1][mask]
    As = vals[2][mask]

    # 2. Tworzenie ŁADNEGO wykresu (jeden duży widok)
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

        elseif c == "q"
            break
        end
    end
end
