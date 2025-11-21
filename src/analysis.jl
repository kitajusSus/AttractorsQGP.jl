include("lib.jl")
include("pca.jl")
include("plt.jl")

using .modHydroSim
using .modPCA
using .modPlots
using GLMakie
using CSV
using DataFrames

const FEATS = [("T",:T), ("A",:A), ("dT",:dTdτ), ("dA",:dAdτ), ("τ₀T",:tau0_T), ("τ₀²dT",:tau0sq_dTdτ)]
const AXES = [(:T,"T"), (:A,"A"), (:dT,"dT"), (:dA,"dA"), (:tau0_T,"τ₀T"), (:tau0sq_dT,"τ₀²dT")]

function prompt_file(dir="datasets")
    !isdir(dir) && return nothing
    files = readdir(dir)
    for (i,f) in enumerate(files); println("[$i] $f"); end
    try
        return joinpath(dir, files[parse(Int64, readline())])
    catch
        return nothing
    end
end

function prompt_axis(lbl)
    println(lbl)
    for (i,v) in enumerate(AXES); println("[$i] $(v[2])"); end
    return AXES[parse(Int64, readline())][1]
end

function save_csv(simres, name)
    df = DataFrame(Run=Int[], Tau=Float64[], T=Float64[], A=Float64[])
    for (i,s) in enumerate(simres.solutions)
        isempty(s.t) && continue
        append!(df, DataFrame(Run=i, Tau=s.t, T=[u[1] for u in s.u], A=[u[2] for u in s.u]))
    end
    CSV.write(name, df)
    println("Saved $name")
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
            for (i,f) in enumerate(FEATS); println("[$i] $(f[1])"); end
            ids = [parse(Int64, x) for x in split(readline(), ",")]
            fs = [FEATS[i][2] for i in ids]

            println("Kroki: "); n = parse(Int64, readline())
            res = modPCA.run_pca_over_time(sim, fs, n, 2, Dict(:method=>:standardize))

            save("pca_var.png", modPlots.plot_explained_variance_evolution(res))
            save("pca_grid.png", modPlots.visualize_pca_static_grid(res, sim, 6))
            save("pca_load.png", modPlots.plot_loadings_evolution(res, [FEATS[i][1] for i in ids]))

        elseif c == "2"
            x, y = prompt_axis("X:"), prompt_axis("Y:")
            println("Czasy: "); ts = [parse(Float64,t) for t in split(readline())]
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

        elseif c == "q"
            break
        end
    end
end
