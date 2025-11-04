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
module modHydroSim

using DifferentialEquations
using Random
using Distributions
using CSV
using DataFrames
using HDF5
using Dates
using Printf

export AbstractHydroParams,
    BRSSSParams,
    SimSettings,
    SimResult,
    PARAMS_SYM_BRSSS,
    PARAMS_MIS,
    run_simulation,
    MeV,
    extract_phase_space_slice,
    generate_and_save_ics,
    load_initial_conditions,
    load_simulation_settings

const fm = 1.0
const MeV = 1 / (197.0 * fm)

abstract type AbstractHydroParams end

struct BRSSSParams <: AbstractHydroParams
    C_τπ::Float64
    C_η::Float64
    C_λ1::Float64
end

struct SimSettings
    theory::Symbol
    params::AbstractHydroParams
    ode::Function
    tspan::Tuple{Float64,Float64}
    n_points::Int
    T_range::Tuple{Float64,Float64}
    A_range::Tuple{Float64,Float64}
    seed::Int
end

const PARAMS_SYM_BRSSS = BRSSSParams((2 - log(2)) / (2 * π), 1 / (4 * π), 1 / (2 * π))
const PARAMS_MIS = BRSSSParams((2 - log(2)) / (2 * π), 1 / (4 * π), 0.0)

function SimSettings(;
    theory::Symbol=:BRSSS,
    n_points=500,
    tspan=(0.2, 1.0),
    T_range=(300.0 * MeV, 1500 * MeV),
    A_range=(-25.0, 25.0),
    seed=5,
)
    params, ode = if theory == :BRSSS
        PARAMS_SYM_BRSSS, ode_brsss!
    elseif theory == :MIS
        PARAMS_MIS, ode_brsss!
    else
        error("Unknown theory $theory. Available: :BRSSS, :MIS")
    end

    return SimSettings(
        theory,
        params,
        ode,
        tspan,
        n_points,
        T_range,
        A_range,
        seed,
    )
end

struct SimResult
    solutions::Vector{ODESolution}
    settings::SimSettings
end

function ode_brsss!(du, u, p::BRSSSParams, τ)
    T, A = u
    C_τπ, C_η, C_λ1 = p.C_τπ, p.C_η, p.C_λ1

    (T <= 1e-9 || !isfinite(T) || !isfinite(A)) && (du .= 0.0; return)

    du[1] = (T / τ) * (-1 / 3 + A / 18)
    term_T = τ * T * (A + (C_λ1 / (12 * C_η)) * A^2)
    term_A2 = (2 / 9) * C_τπ * A^2
    du[2] = (1 / (C_τπ * τ)) * (8 * C_η - term_T - term_A2)
end

function evol(u0, settings::SimSettings)
    prob = ODEProblem(settings.ode, u0, settings.tspan, settings.params)

    return solve(prob, Rodas5(), save_everystep=false, dense=true, abstol=1e-6, reltol=1e-6)
end

function run_simulation(; settings::SimSettings, ic_file::Union{String,Nothing}=nothing)
    println("Starting simulation for theory: $(settings.theory)...")

    initial_states = if ic_file !== nothing
        load_initial_conditions(ic_file)
    else
        println(
            "No file provided, generating $(settings.n_points) random initial conditions (seed: $(settings.seed)).",
        )
        generate_initial_conditions(settings)
    end

    solutions = ODESolution[]
    for u0 in initial_states
        sol = evol(u0, settings)
        push!(solutions, sol)
    end

    println("Simulation ended, generated $(length(solutions)) trajectories.")
    return SimResult(solutions, settings)
end

function load_initial_conditions(filepath::String)
    println("Loading initial conditions from: $filepath")

    df = if endswith(filepath, ".csv")
        CSV.read(filepath, DataFrame; comment="#")
    elseif endswith(filepath, ".h5")
        h5open(filepath, "r") do file
            g = file["initial_conditions"]
            cols = names(g)
            DataFrame([col => read(g[col]) for col in cols])
        end
    else
        error("Unsupported file type: $filepath. Use .csv or .h5")
    end

    return [[row.T_0, row.A_0] for row in eachrow(df)]
end

function generate_initial_conditions(settings::SimSettings)
    rng = Xoshiro(settings.seed)
    Ts = rand(rng, Uniform(settings.T_range...), settings.n_points)
    As = rand(rng, Uniform(settings.A_range...), settings.n_points)
    return [[Ts[j], As[j]] for j = 1:(settings.n_points)]
end

function settings_to_header(settings::SimSettings)
    header = "# SIMULATION SETTINGS\n"
    header *= "# =====================\n"
    for field in fieldnames(typeof(settings))
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
    settings::SimSettings,
    output_base_filename="initial_conditions",
)
    ics_list = generate_initial_conditions(settings)
    df = DataFrame(T_0=[ic[1] for ic in ics_list], A_0=[ic[2] for ic in ics_list])
    df.Run_ID = 1:settings.n_points

    csv_filename = "$(output_base_filename).csv"
    header = settings_to_header(settings)
    open(csv_filename, "w") do f
        write(f, header)
        CSV.write(f, df[!, [:Run_ID, :T_0, :A_0]], append=true, writeheader=true)
    end
    println("Saved initial conditions to: $csv_filename")

    h5_filename = "$(output_base_filename).h5"
    h5open(h5_filename, "w") do file
        g_data = create_group(file, "initial_conditions")
        for col in names(df)
            g_data[col] = df[!, col]
        end
        attrs(g_data)["description"] = "Random initial conditions."
        attrs(g_data)["timestamp"] = string(now())

        g_settings = create_group(file, "settings")
        attrs(g_settings)["description"] = "Simulation settings used to generate this data."
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
    println("Saved initial conditions to: $h5_filename")
    return df
end

function extract_phase_space_slice(simres::SimResult, t::Float64)
    params = simres.settings.params
    ode_func! = simres.settings.ode
    n_sols = length(simres.solutions)
    n_vars = length(simres.solutions[1].u[1])

    valid_mask = falses(n_sols)
    u_values = [fill(NaN, n_sols) for _ = 1:n_vars]
    du_values = [fill(NaN, n_sols) for _ = 1:n_vars]

    du_cache = zeros(Float64, n_vars)

    for (i, sol) in enumerate(simres.solutions)
        if t < sol.t[1] || t > sol.t[end]
            continue
        end

        u = sol(t)

        if any(!isfinite, u) || u[1] <= 1e-9
            continue
        end

        ode_func!(du_cache, u, params, t)

        for j = 1:n_vars
            u_values[j][i] = u[j]
            du_values[j][i] = du_cache[j]
        end

        valid_mask[i] = true
    end

    return (u_values, du_values, valid_mask)
end

function load_simulation_settings(filepath::String)
    println("Wczytywanie ustawień z pliku: $filepath")

    config = Dict{Symbol,Any}()
    param_config = Dict{Symbol,Any}()

    if endswith(filepath, ".csv")
        for line in eachline(filepath)
            !startswith(line, "#") && break

            local parts, line_content
            if startswith(line, "#   ")
                line_content = strip(line[5:end])
                parts = split(line_content, " = ")
                length(parts) != 2 && continue
                param_config[Symbol(parts[1])] = parse(Float64, parts[2])
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
        params, ode = if theory == :BRSSS || theory == :MIS
            BRSSSParams(param_config[:C_τπ], param_config[:C_η], param_config[:C_λ1]), ode_brsss!
        else
            error("Unknown or unsupported theory $theory during CSV load.")
        end

        return SimSettings(
            theory,
            params,
            ode,
            config[:tspan],
            config[:n_points],
            config[:T_range],
            config[:A_range],
            config[:seed]
        )

    elseif endswith(filepath, ".h5")
        error("Wczytywanie ustawień z H5 nie jest jeszcze zaimplementowane.")
    else
        error("Nieobsługiwany typ pliku: $filepath. Użyj .csv lub .h5")
    end
end

end
