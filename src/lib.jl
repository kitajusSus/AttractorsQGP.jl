# Zaimplementowane teorie:
# 1. BRSSS (Baier, Romatschke, Son, Starinets, Stephanov)
#    - Cechy: Drugo-rzędowa hydrodynamika lepka, wprowadza czas relaksacji.
#    - Zmienne stanu: [T(τ), A(τ)] (Temperatura, Anizotropia ciśnień).
#
# 2. MIS (Müller, Israel, Stewart)
#    - Cechy: Uproszczona wersja BRSSS, często używana jako jej podstawa.
#      Formalnie, jest to BRSSS z współczynnikiem C_λ1 = 0.
#    - Zmienne stanu: [T(τ), A(τ)].
#
# 3. HJSW (Heller, Janik, Spaliński, Witaszczyk)
#    - Cechy: Teoria "poza-hydrodynamiczna", jawnie uwzględnia mody oscylacyjne.
#      Opisana równaniem różniczkowym wyższego rzędu.
#    - Zmienne stanu: [T(τ), A(τ), Z(τ)], gdzie Z jest zmienną pomocniczą
#      związaną z pochodną anizotropii po czasie bezwymiarowym w = τT.
#


using DifferentialEquations
using Random
using Distributions
using Plots
using CSV
using LaTeXStrings
using DataFrames
using HDF5
using Dates
using Printf

module modHydroSim
using Base: annotate!
using DifferentialEquations
using Random
using Distributions
using Plots
gr()
using LaTeXStrings
using CSV
using DataFrames
using HDF5
using Dates
using Printf

export AbstractHydroParams,
    BRSSSParams,
    HJSWParams,
    SimSettings,
    SimResult,
    PARAMS_SYM_BRSSS,
    PARAMS_MIS,
    PARAMS_SYM_HJSW,
    run_simulation,
    TA,
    generate_and_save_ics,
    repl_run_brsss,
    repl_run_mis,
    repl_run_hjsw,
    repl_demo_file_io,
    fm,
    MeV,
    run_all_theories,
    NAMESPACES,
    Namespaces,
    load_settings

const fm = 1.0
const MeV = 1 / (197.0 * fm)
const PLOTS_DIR = "plots"


struct Namespaces
    datasets::String
    plots_tex::String
    plots::String
    label::String
end
const NAMESPACES = Namespaces("datasets", "plots_tex", "plots", "Eksperyment_1")

abstract type AbstractHydroParams end

struct BRSSSParams <: AbstractHydroParams
    C_τπ::Float64
    C_η::Float64
    C_λ1::Float64
end

struct HJSWParams <: AbstractHydroParams
    C_η::Float64
    C_σ::Float64
    Ω_R::Float64
    Ω_I::Float64
end

const PARAMS_SYM_BRSSS = BRSSSParams((2 - log(2)) / (2 * π), 1 / (4 * π), 1 / (2 * π))
const PARAMS_MIS = BRSSSParams((2 - log(2)) / (2 * π), 1 / (4 * π), 0.0)
const PARAMS_SYM_HJSW = HJSWParams(1 / (4 * π), 2 * π, 9.800, 8.629)

struct SimSettings
    theory::Symbol
    params::AbstractHydroParams
    ode::Function
    tspan::Tuple{Float64,Float64}
    n_points::Int
    T_range::Tuple{Float64,Float64}
    A_range::Tuple{Float64,Float64}
    Z_range::Tuple{Float64,Float64}
    seed::Int
end

function SimSettings(;
    theory::Symbol=:BRSSS,
    n_points=500,
    tspan=(0.2, 1),
    T_range=(300.0 * MeV, 1500 * MeV),
    A_range=(-25, 25),
    Z_range=(-20.0, 20.0),
    seed=5,
)
    if theory == :BRSSS
        params, ode = PARAMS_SYM_BRSSS, ode_brsss!
    elseif theory == :MIS
        params, ode = PARAMS_MIS, ode_brsss!
    elseif theory == :HJSW
        params, ode = PARAMS_SYM_HJSW, ode_hjsw!
    else
        error("Unknown theory $theory. Available: :BRSSS, :MIS, :HJSW")
    end
    return SimSettings(
        theory,
        params,
        ode,
        tspan,
        n_points,
        T_range,
        A_range,
        Z_range,
        seed,
    )
end

struct SimResult
    solutions::Vector
    settings::SimSettings
end

function ode_brsss!(du, u, p::BRSSSParams, τ)
    T, A = u
    C_τπ, C_η, C_λ1 = p.C_τπ, p.C_η, p.C_λ1
    du[1] = (T / τ) * (-1 / 3 + A / 18)
    term_T = τ * T * (A + (C_λ1 / (12 * C_η)) * A^2)
    term_A2 = (2 / 9) * C_τπ * A^2
    du[2] = (1 / (C_τπ * τ)) * (8 * C_η - term_T - term_A2)
end

function ode_hjsw!(du, u, p::HJSWParams, τ)
    T, A, Z = u[1], u[2], u[3]
    C_η, C_σ, Ω_R, Ω_I = p.C_η, p.C_σ, p.Ω_R, p.Ω_I
    w = τ * T
    (T <= 1e-9 || w <= 1e-9 || !isfinite(T) || !isfinite(A) || !isfinite(Z)) &&
        (du .= 0.0;
        return)
    Ω² = Ω_R^2 + Ω_I^2
    α1, α2 = w^2 * (A + 12)^2, w^2 * (A + 12)
    α3, α4 = 12w * (A + 12) * (A + 3w * Ω_I), 48 * (3w * Ω_I - 1)
    α5, α6 = 108 * (-4C_η * C_σ + 3w^2 * Ω²), -864C_η * (-2C_σ + 3w * Ω²)
    du[1] = (T / τ) * (-1 / 3 + A / 18)
    dw_dτ = T + τ * du[1]

    du[2] = dw_dτ * Z
    A_double_prime_numerator = -(α2 * Z^2 + α3 * Z + α4 * A^2 + α5 * A + α6)
    A_double_prime = α1 ≈ 0 ?
    0.0 : A_double_prime_numerator / α1
    du[3] = dw_dτ * A_double_prime
end

function load_ics(filepath::String, theory::Symbol)
    println("Loading of initial_conditions : $filepath")

    if endswith(filepath, ".csv")
        df = CSV.read(filepath, DataFrame, comment="#")
    elseif endswith(filepath, ".h5")
        df = h5open(filepath, "r") do file
            g = file["initial_conditions"]

            cols = names(g)
            DataFrame([col => read(g[col]) for col in cols])
        end
    else
        error("Unsupported file type: $filepath. use .csv or .h5")
    end

    if theory == :HJSW
        if !("Z_0" in names(df))
            error("file dont have  'Z_0' required to  HJSW theory .")
        end
        return [[row.T_0, row.A_0, row.Z_0] for row in eachrow(df)]

    else
        return [[row.T_0, row.A_0] for row in eachrow(df)]
    end
end

function generate_random_ics(settings::SimSettings)
    rng = Xoshiro(settings.seed)
    Ts = rand(rng, Uniform(settings.T_range...), settings.n_points)
    As = rand(rng, Uniform(settings.A_range...), settings.n_points)

    if settings.theory == :HJSW
        Zs = rand(rng, Uniform(settings.Z_range...), settings.n_points)
        return [[Ts[j], As[j], Zs[j]] for j = 1:(settings.n_points)]
    else
        return [[Ts[j], As[j]] for j = 1:(settings.n_points)]
    end
end

function initial_conditions(settings, seed=5)
    rng = Xoshiro(seed)
    Tmin, Tmax = settings.T_range

    Amin, Amax = settings.A_range
    Ts = rand(rng, Uniform(Tmin, Tmax), settings.n_points)
    As = rand(rng, Uniform(Amin, Amax), settings.n_points)
    return [[Ts[j], As[j]] for j = 1:settings.n_points]
end

function evol(u0, p)
    prob = ODEProblem(p.ode, u0, p.tspan, p.params)
    return solve(prob, Tsit5())
end

function run_simulation(; settings::SimSettings, ic_file::Union{String,Nothing}=nothing)
    println("Starting simulation for theory: $(settings.theory)...")

    initial_states = if ic_file !== nothing
        load_ics(ic_file, settings.theory)
    else
        println(

           "There is no file $(ic_file), Generating    $(settings.n_points) random initial_conditions (seed: $(settings.seed)).",
        )
        generate_random_ics(settings)
    end

    solutions = ODESolution[]
    for u0 in initial_states
        sol = evol(u0, settings)
        push!(solutions, sol)
    end

    println("Simulation ended, generated $(length(solutions)) trajectories.")
    return SimResult(solutions, settings)
end

function load_settings_from_csv(filepath::String)
    config = Dict{Symbol,Any}()
    param_config = Dict{Symbol,Any}()


    for line in eachline(filepath)
        !startswith(line, "#") && break

        local parts
        if startswith(line, "#   ")
            line_content = strip(line[5:end])
            parts = split(line_content, " = ")
            length(parts) != 2 && continue

            key, value_str = parts[1], parts[2]
            param_config[Symbol(key)] = parse(Float64, value_str)
        elseif startswith(line, "# ")
            line_content = strip(line[3:end])
            parts = split(line_content, " = ")
            length(parts) != 2 && continue
            key, value_str = parts[1], parts[2]

            key_sym = Symbol(key)
            if key_sym == :theory
                config[key_sym] = Symbol(value_str)
            elseif key_sym in [:n_points, :seed]
                config[key_sym] = parse(Int, value_str)
            elseif key_sym in [:tspan, :T_range, :A_range, :Z_range]

                val = eval(Meta.parse(value_str))
                config[key_sym] = val
            end
        end
    end

    theory = config[:theory]
    local params, ode
    if theory == :BRSSS ||
        theory == :MIS
        params = BRSSSParams(param_config[:C_τπ], param_config[:C_η], param_config[:C_λ1])
        ode = ode_brsss!
    elseif theory == :HJSW
        params = HJSWParams(param_config[:C_η], param_config[:C_σ], param_config[:Ω_R], param_config[:Ω_I])
        ode = ode_hjsw!
    else
        error("Unknown theory $theory during CSV load.")
    end

    !haskey(config, :Z_range) && (config[:Z_range] = (-20.0, 20.0))

    return SimSettings(
        theory, params, ode,
        config[:tspan], config[:n_points], config[:T_range],
        config[:A_range], config[:Z_range], config[:seed]
    )
end

function load_settings_from_h5(filepath::String)
    config = Dict{Symbol,Any}()
    param_config = Dict{Symbol,Any}()

    h5open(filepath, "r") do file
        g = file["settings"]

        for key in keys(attrs(g))
            value = read(attrs(g)[key])
            if startswith(key, "param_")
                param_key = Symbol(replace(key, "param_" => ""))
                param_config[param_key] = value
            else

                key_sym = Symbol(key)
                if key_sym == :theory
                    config[key_sym] = Symbol(value)
                elseif isa(value, Vector) && length(value) == 2
                    config[key_sym] = tuple(value...)

                else
                    config[key_sym] = value
                end
            end
        end
    end

    theory = config[:theory]
    local params, ode
    if theory == :BRSSS ||
        theory == :MIS
        params = BRSSSParams(param_config[:C_τπ], param_config[:C_η], param_config[:C_λ1])
        ode = ode_brsss!
    elseif theory == :HJSW
        params = HJSWParams(param_config[:C_η], param_config[:C_σ], param_config[:Ω_R], param_config[:Ω_I])
        ode = ode_hjsw!
    else
        error("Unknown theory $theory during H5 load.")
    end

    !haskey(config, :Z_range) && (config[:Z_range] = (-20.0, 20.0))

    return SimSettings(
        theory, params, ode,
        config[:tspan], config[:n_points], config[:T_range],
        config[:A_range], config[:Z_range], config[:seed]
    )
end

function load_settings(filepath::String)
    println("Wczytywanie ustawień z pliku: $filepath")

    if endswith(filepath, ".csv")
        return load_settings_from_csv(filepath)

    elseif endswith(filepath, ".h5")
        return load_settings_from_h5(filepath)
    else
        error("Nieobsługiwany typ pliku: $filepath. Użyj .csv lub .h5")
    end
end

function TA(simres::SimResult, t::Float64)
    params, ode_func!, n_sols =
        simres.settings.params, simres.settings.ode, length(simres.solutions)

    valid_mask = falses(n_sols)

    if simres.settings.theory == :HJSW
        Ts, As, Zs = (fill(NaN, n_sols) for _ = 1:3)
        dTs,
        dAs, dZs = (fill(NaN, n_sols) for _ = 1:3)
        du = [0.0, 0.0, 0.0]
        for (i, sol) in enumerate(simres.solutions)
            u = sol(t)
            if any(!isfinite, u) ||
                u[1] <= 1e-9
                continue
            end
            ode_func!(du, u, params, t)
            Ts[i], As[i], Zs[i] = u
            dTs[i], dAs[i], dZs[i] = du
            valid_mask[i] = true
        end

        return (Ts, As, Zs, dTs, dAs, dZs), valid_mask
    else
        Ts, As = (fill(NaN, n_sols) for _ = 1:2)
        dTs, dAs = (fill(NaN, n_sols) for _ = 1:2)
        du = [0.0, 0.0]
        for (i, sol) in enumerate(simres.solutions)
            u = sol(t)

            if any(!isfinite, u) || u[1] <= 1e-9
                continue
            end
            ode_func!(du, u, params, t)
            Ts[i], As[i] = u
            dTs[i], dAs[i] = du
            valid_mask[i] = true

        end
        return (Ts, As, dTs, dAs), valid_mask
    end
end

function settings_to_header(settings::SimSettings)
    header = "# SIMULATION SETTINGS\n"
    header *= "# =====================\n"
    for field in
        fieldnames(typeof(settings))
        value = getfield(settings, field)
        if field == :params
            header *= "# Parameters for $(settings.theory):\n"
            for p_field in fieldnames(typeof(value))
                p_value = getfield(value, p_field)
                header *= "#   $(p_field) = $(p_value)\n"

            end
        elseif field != :ode
            header *= "# $(field) = $(value)\n"
        end
    end
    header *= "# =====================\n"
    return header
end


function generate_and_save_ics(;
    settings::SimSettings, output_filename_base="IC_")
    rng = Xoshiro(settings.seed)
    df = DataFrame(
        Run_ID=1:settings.n_points,
        T_0=rand(rng, Uniform(settings.T_range...), settings.n_points),
        A_0=rand(rng, Uniform(settings.A_range...), settings.n_points),
    )
    if settings.theory == :HJSW
        df.Z_0 = rand(rng, Uniform(settings.Z_range...), settings.n_points)
    end

    csv_filename = "$(output_filename_base)_$(settings.T_range)_$(settings.A_range)_$(settings.n_points)_t_$(settings.tspan).csv"
    header = settings_to_header(settings)
    open(csv_filename, "w") do f
        write(f, header)

        CSV.write(f, df, append=true, writeheader=true)
    end
    println("Zapisano warunki początkowe do: $csv_filename")

    h5_filename = "$(output_filename_base).h5"
    h5open(h5_filename, "w") do file
        g_data = create_group(file, "initial_conditions")
        for col in names(df)
            g_data[col] = df[!, col]
        end
        attrs(g_data)["description"] = "Zestaw losowych warunków początkowych."
        attrs(g_data)["timestamp"] = string(now())

        g_settings = create_group(file, "settings")
        attrs(g_settings)["description"] = "Ustawienia symulacji użyte do wygenerowania danych."
        for field in fieldnames(typeof(settings))
            value = getfield(settings, field)
            s_field = string(field)
            if field == :params
                for p_field in fieldnames(typeof(value))
                    attrs(g_settings)["param_$(p_field)"] = getfield(value, p_field)

                end
            elseif field != :ode
                attr_value = if isa(value, Tuple)
                    [value...]
                elseif isa(value, Symbol)
                    string(value)

                else
                    value
                end
                attrs(g_settings)[s_field] = attr_value
            end
        end
    end
    println("Zapisano warunki początkowe do: $h5_filename")
    return df
end

function repl_run_brsss()
    println("\n---Simulation for  BRSSS (random points) ---")
    settings = SimSettings(
        theory=:BRSSS,
        n_points=500,
        tspan=(0.2, 2.5),
        A_range=(-1.0, 5.0),
    )
    result = run_simulation(settings=settings)

    return result
end

function repl_run_mis()
    println("\n--- SIMULATION FOR THEORY MIS (random initial_conditions) ---")
    settings =

        SimSettings(theory=:MIS, n_points=500, tspan=(0.2, 1), A_range=(-50, 15))
    result = run_simulation(settings=settings)

    return result
end

function repl_run_hjsw()
    println("\n--- SIMULATION FOR HJSW THEORY  (random initial_conditions) ---")
    settings = SimSettings(
        theory=:HJSW,
        n_points=500,
        tspan=(0.2, 1.5),
        A_range=(-1.0, 5.0),
    )
    result = run_simulation(settings=settings)

    return result
end


function repl_demo_file_io()
    println("\n--- Demo of loading data---")
    filename = "demo_ic.csv"






    settings = SimSettings(theory=:BRSSS, tspan=(0.2, 2.5))
    result = run_simulation(settings=settings, ic_file=filename)

    return result
end

function run_all_theories(ic_file::String;
    tspan=(0.2, 1.2))
    println("="^60)
    println(" Starting benchmark for initial_conditions from : $ic_file")
    println("="^60)

    df = CSV.read(ic_file, DataFrame)
    T_min_max = (minimum(df.T_0) * MeV, maximum(df.T_0) * MeV)
    A_min_max = (minimum(df.A_0), maximum(df.A_0))
    Z_min_max = hasproperty(df, :Z_0) ?
    (minimum(df.Z_0), maximum(df.Z_0)) : (-20.0, 20.0)

    theories = [:MIS, :BRSSS, :HJSW]
    results = Dict{Symbol,SimResult}()
    for theory in theories
        current_tspan = theory == :HJSW ?
        (tspan[1], 2.5) : tspan

        settings = SimSettings(
            theory=theory,
            tspan=current_tspan,
            n_points=nrow(df),
            T_range=T_min_max,
            A_range=A_min_max,
            Z_range=Z_min_max,
        )


        result = run_simulation(settings=settings, ic_file=ic_file)
        results[theory] = result
        println("\n--- Generating plot for $theory ---")


        if theory != last(theories)
            println(
                "\n Plots generated for  $theory. Press Enter, to continue...",
            )

            readline()
        end
    end
    println("\nCykl porównawczy zakończony.")
    return results
end

end
