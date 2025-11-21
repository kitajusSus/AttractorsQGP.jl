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

export AbstractHydroParams, BRSSSParams, SimSettings, SimResult,
       run_simulation, extract_phase_space_slice,
       generate_and_save_ics, load_initial_conditions, load_simulation_settings

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

struct SimResult
    solutions::Vector{ODESolution}
    settings::SimSettings
end

const PARAMS_SYM_BRSSS = BRSSSParams((2 - log(2)) / (2 * π), 1 / (4 * π), 1 / (2 * π))
const PARAMS_MIS = BRSSSParams((2 - log(2)) / (2 * π), 1 / (4 * π), 0.0)

function ode_brsss!(du, u, p::BRSSSParams, τ)
    T, A = u
    C_τπ, C_η, C_λ1 = p.C_τπ, p.C_η, p.C_λ1

    if T <= 1e-9 || !isfinite(T) || !isfinite(A)
        du .= 0.0
        return
    end

    du[1] = (T / τ) * (-1 / 3 + A / 18)
    term_T = τ * T * (A + (C_λ1 / (12 * C_η)) * A^2)
    term_A2 = (2 / 9) * C_τπ * A^2
    du[2] = (1 / (C_τπ * τ)) * (8 * C_η - term_T - term_A2)
end

function SimSettings(;
    theory::Symbol=:BRSSS,
    n_points=500,
    tspan=(0.2, 1.0),
    T_range=(1.5, 8.0),
    A_range=(-25.0, 25.0),
    seed=5,
)
    params, ode = if theory == :BRSSS
        PARAMS_SYM_BRSSS, ode_brsss!
    elseif theory == :MIS
        PARAMS_MIS, ode_brsss!
    else
        error("Unknown theory $theory")
    end

    return SimSettings(theory, params, ode, tspan, n_points, T_range, A_range, seed)
end

function evol(u0, settings::SimSettings)
    prob = ODEProblem(settings.ode, u0, settings.tspan, settings.params)
    return solve(prob, Rodas5(), save_everystep=false, dense=true, abstol=1e-8, reltol=1e-8)
end

function run_simulation(; settings::SimSettings, ic_file::Union{String,Nothing}=nothing)
    println("Starting simulation: $(settings.theory)")

    initial_states = if ic_file !== nothing
        load_initial_conditions(ic_file)
    else
        generate_initial_conditions(settings)
    end

    solutions = ODESolution[]
    for u0 in initial_states
        push!(solutions, evol(u0, settings))
    end

    return SimResult(solutions, settings)
end

function generate_initial_conditions(settings::SimSettings)
    rng = Xoshiro(settings.seed)
    Ts = rand(rng, Uniform(settings.T_range...), settings.n_points)
    As = rand(rng, Uniform(settings.A_range...), settings.n_points)
    return [[Ts[j], As[j]] for j = 1:(settings.n_points)]
end

function load_initial_conditions(filepath::String)
    df = if endswith(filepath, ".csv")
        CSV.read(filepath, DataFrame; comment="#")
    elseif endswith(filepath, ".h5")
        h5open(filepath, "r") do file
            g = file["initial_conditions"]
            DataFrame([col => read(g[col]) for col in names(g)])
        end
    else
        error("Unsupported file type")
    end
    return [[row.T_0, row.A_0] for row in eachrow(df)]
end

function generate_and_save_ics(; settings::SimSettings, output_base_filename="initial_conditions")
    ics_list = generate_initial_conditions(settings)
    df = DataFrame(T_0=[ic[1] for ic in ics_list], A_0=[ic[2] for ic in ics_list])
    df.Run_ID = 1:settings.n_points

    csv_filename = "$(output_base_filename).csv"
    open(csv_filename, "w") do f
        write(f, settings_to_header(settings))
        CSV.write(f, df[!, [:Run_ID, :T_0, :A_0]], append=true, writeheader=true)
    end

    h5_filename = "$(output_base_filename).h5"
    h5open(h5_filename, "w") do file
        g_data = create_group(file, "initial_conditions")
        for col in names(df); g_data[col] = df[!, col]; end
        attrs(g_data)["timestamp"] = string(now())

        g_settings = create_group(file, "settings")
        for field in fieldnames(typeof(settings))
            val = getfield(settings, field)
            if field == :params
                for p in fieldnames(typeof(val))
                    attrs(g_settings)["param_$p"] = getfield(val, p)
                end
            elseif field != :ode
                attrs(g_settings)[string(field)] = string(val)
            end
        end
    end
    return df
end

function settings_to_header(settings::SimSettings)
    header = "# SIMULATION SETTINGS\n"
    for field in fieldnames(typeof(settings))
        val = getfield(settings, field)
        if field == :params
            for p in fieldnames(typeof(val))
                header *= "#   $p = $(getfield(val, p))\n"
            end
        elseif field != :ode
            header *= "# $field = $val\n"
        end
    end
    return header
end

function load_simulation_settings(filepath::String)
    config = Dict{Symbol,Any}()
    param_config = Dict{Symbol,Any}()

    if endswith(filepath, ".csv")
        for line in eachline(filepath)
            !startswith(line, "#") && break
            content = ""
            is_param = false

            if startswith(line, "#   ")
                content = strip(replace(line, "#   " => "", count=1))
                is_param = true
            elseif startswith(line, "# ")
                content = strip(replace(line, "# " => "", count=1))
            else
                continue
            end

            parts = split(content, " = ")
            length(parts) != 2 && continue
            key, val_str = Symbol(parts[1]), parts[2]

            if is_param
                param_config[key] = parse(Float64, val_str)
            else
                if key == :theory
                    config[key] = Symbol(val_str)
                elseif key in [:n_points, :seed]
                    config[key] = parse(Int, val_str)
                elseif key in [:tspan, :T_range, :A_range]
                    config[key] = eval(Meta.parse(val_str))
                end
            end
        end

        theory = config[:theory]
        params, ode = if theory in [:BRSSS, :MIS]
            BRSSSParams(param_config[:C_τπ], param_config[:C_η], param_config[:C_λ1]), ode_brsss!
        else
            error("Unsupported theory in CSV")
        end

        return SimSettings(theory, params, ode, config[:tspan], config[:n_points], config[:T_range], config[:A_range], config[:seed])
    else
        error("Only CSV supported for settings load")
    end
end

function extract_phase_space_slice(simres::SimResult, t::Float64)
    params = simres.settings.params
    ode! = simres.settings.ode
    n_sols = length(simres.solutions)
    n_vars = 2

    u_vals = [fill(NaN, n_sols) for _ in 1:n_vars]
    du_vals = [fill(NaN, n_sols) for _ in 1:n_vars]
    valid = falses(n_sols)
    cache = zeros(n_vars)

    for (i, sol) in enumerate(simres.solutions)
        if t < sol.t[1] || t > sol.t[end]; continue; end
        u = sol(t)
        if any(!isfinite, u) || u[1] <= 1e-9; continue; end

        ode!(cache, u, params, t)
        for j in 1:n_vars
            u_vals[j][i] = u[j]      # Było u_values
            du_vals[j][i] = cache[j] # Było du_values
        end
        valid[i] = true
    end
    return (u_vals, du_vals, valid) # Zwracamy poprawne zmienne
end

end
