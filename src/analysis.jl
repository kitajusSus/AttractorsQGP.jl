include("lib.jl")
include("pca.jl")
include("plt.jl")

using .modHydroSim
using .modPCA
using .modPlots
using GLMakie
using Dates
using DataFrames
using CSV

# --- Mapowanie dla PCA ---
const FEATURES_MAP = [
    ("T", :T), ("A", :A),
    ("dT", :dTdτ), ("dA", :dAdτ),
    ("tau0*T", :tau0_T), ("tau0^2*dT", :tau0sq_dTdτ)
]

# --- Mapowanie dla Wykresów ---
const PLOT_VARS = [
    (:T, "T"), (:A, "A"),
    (:dT, "dT"), (:dA, "dA"),
    (:tau0_T, "tau0*T"), (:tau0sq_dT, "tau0^2*dT"),
    (:tau_T, "tau*T")
]

function prompt_dataset(dir="datasets")
    if !isdir(dir); println("Brak folderu $dir"); return nothing; end
    files = filter(x->endswith(x, ".csv") || endswith(x, ".h5"), readdir(dir))
    if isempty(files); println("Pusty folder."); return nothing; end
    println("\n--- Wybierz plik ---")
    for (i,f) in enumerate(files); println("[$i] $f"); end
    print("Wybór: ")
    try
        idx = parse(Int, readline())
        return joinpath(dir, files[idx])
    catch
        return nothing
    end
end

function prompt_axis(txt)
    println("\n$txt")
    for (i, (sym, desc)) in enumerate(PLOT_VARS); println("[$i] $desc"); end
    print("Wybór: ")
    try
        idx = parse(Int, readline())
        return PLOT_VARS[idx][1]
    catch
        return :T
    end
end

function save_simple_csv(simres, fname="ewolucja.csv")
    println("Zapisuje dane do $fname ...")
    data = DataFrame(Run=Int[], Tau=Float64[], T=Float64[], A=Float64[])

    for (i, sol) in enumerate(simres.solutions)
        if isempty(sol.t); continue; end
        # Surowe dane
        append!(data, DataFrame(Run=fill(i, length(sol.t)), Tau=sol.t, T=[u[1] for u in sol.u], A=[u[2] for u in sol.u]))
    end
    CSV.write(fname, data)
    println("Zapisano.")
end

function main()
    file = prompt_dataset()
    isnothing(file) && return

    # Ładujemy raz na początku, żeby było prosto
    sets = load_simulation_settings(file)
    sim = run_simulation(settings=sets, ic_file=file)

    while true
        println("\n=== MENU PROSTE ===")
        println("[1] PCA Custom")
        println("[2] Wykresy (Wybór osi)")
        println("[3] Animacja (Wybór osi)")
        println("[5] Ewolucja A, T + Chmura A vs T + CSV")
        println("[q] Wyjście")

        c = readline()

        if c == "1"
            println("Wybierz cechy (indeksy po przecinku):")
            for (i, (n, _)) in enumerate(FEATURES_MAP); println("[$i] $n"); end
            idxs = [parse(Int, x) for x in split(readline(), ",")]
            feats = [FEATURES_MAP[i][2] for i in idxs]

            println("Metoda (1:std, 2:center, 3:none): "); m_idx = parse(Int, readline())
            m = [:standardize, :center, :none][m_idx]

            println("Ile kroków czasu: "); n_steps = parse(Int, readline())
            res = modPCA.run_pca_over_time(sim, feats, n_steps, 2, Dict(:method=>m))

            save("pca_var.png", modPlots.plot_explained_variance_evolution(res))
            save("pca_grid.png", modPlots.visualize_pca_static_grid(res, sim, 6))

        elseif c == "2"
            x = prompt_axis("Oś X:")
            y = prompt_axis("Oś Y:")
            println("Czasy (np. 0.5 1.0): ")
            ts = [parse(Float64, t) for t in split(readline())]

            save("grid_$(x)_$(y).png", modPlots.plot_phase_space_grid(sim, ts, x, y))

        elseif c == "3"
            x = prompt_axis("Oś X:")
            y = prompt_axis("Oś Y:")
            modPlots.animate_phase_space_evolution(sim, x, y; output_filename="anim.gif")

        elseif c == "5"
            # println("--- 1. Wykresy Trajektorii ---")
            # fig = modPlots.plot_thermodynamics_evolution(sim)
            # save("trajektorie_surowe.png", fig)
            # println("Zapisano: trajektorie_surowe.png")

            println("--- 2. Zapis CSV ---")
            save_simple_csv(sim, "dane_surowe.csv")

            println("--- 3. Chmura punktów A vs T ---")
            println("(Oś X = T, Oś Y = A)")
            print("Podaj czasy snapshotów: ")
            ts = [parse(Float64, t) for t in split(readline())]

            fig_cloud = modPlots.plot_phase_space_grid(sim, ts, :T, :A)
            save("chmura_AvsT.png", fig_cloud)
            println("Zapisano: chmura_AvsT.png")

        elseif c == "q"
            break
        end
    end
end
