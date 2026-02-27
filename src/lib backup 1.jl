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

export AbstractHydroParams, BRSSSParams, MISParams, HJSWParams,
       SimSettings, SimResult,
       run_simulation, extract_phase_space_slice,
       generate_and_save_ics, load_initial_conditions, load_simulation_settings

abstract type AbstractHydroParams end

struct BRSSSParams <: AbstractHydroParams
    C_τπ::Float64
    C_η::Float64
    C_λ1::Float64
end


struct MISParams <: AbstractHydroParams
    C_τπ::Float64
    C_η::Float64
end

struct HJSWParams <: AbstractHydroParams
    C_τπ::Float64
    C_η::Float64
    ω0::Float64
end

struct SimSettings{P<:AbstractHydroParams, F}
    theory::Symbol
    params::P
    ode::F
    tspan::Tuple{Float64,Float64}
    n_points::Int
    T_range::Tuple{Float64,Float64}
    A_range::Tuple{Float64,Float64}
    seed::Int
end

struct SimResult{P<:AbstractHydroParams,F}
    solutions::Vector{ODESolution}
    settings::SimSettings{P,F}
end

const PARAMS_SYM_BRSSS = BRSSSParams((2 - log(2)) / (2 * π), 1 / (4 * π), 1 / (2 * π))
const PARAMS_MIS = MISParams((2 - log(2)) / (2 * π), 1 / (4 * π))
const PARAMS_HJSW = HJSWParams((2 - log(2)) / (2 * π), 1 / (4 * π), 2.0)

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

function ode_mis!(du, u, p::MISParams, τ)
    T, A = u
    C_τπ, C_η = p.C_τπ, p.C_η

    if T <= 1e-9 || !isfinite(T) || !isfinite(A)
        du .= 0.0
        return
    end

    du[1] = (T / τ) * (-1 / 3 + A / 18)
    du[2] = (1 / (C_τπ * τ)) * (8 * C_η - (τ*T*A) - (2/9)*C_τπ*A^2)
end

function ode_hjsw!(du, u, p::HJSWParams, τ)
    T, A, Z = u
    C_τπ, C_η, ω0 = p.C_τπ, p.C_η, p.ω0

    if T <= 1e-9 || !isfinite(T) || !isfinite(A) || !isfinite(Z)
        du .= 0.0
        return
    end

    du[1] = (T / τ) * (-1 / 3 + A / 18)
    du[2] = Z
    du[3] = -ω0^2 * A - (1/(C_τπ*τ)) * Z
end

function SimSettings(; theory::Symbol=:BRSSS, n_points=500,
                     tspan=(0.2,1.0), T_range=(1.5,8.0),
                     A_range=(-25.0,25.0), seed=5)
    if theory == :BRSSS
        params = PARAMS_SYM_BRSSS
        ode = ode_brsss!
    elseif theory == :MIS
        params = PARAMS_MIS
        ode = ode_mis!
    elseif theory == :HJSW
        params = PARAMS_HJSW
        ode = ode_hjsw!
    else
        error("Unknown theory $theory")
    end
    return SimSettings(theory, params, ode, tspan, n_points, T_range, A_range, seed)
end

function evol(u0, settings::SimSettings{P,F}) where {P,F}
    prob = ODEProblem(settings.ode, u0, settings.tspan, settings.params)
    return solve(prob, Rodas5(), abstol=1e-6, reltol=1e-6)
end

function run_simulation(; settings::SimSettings, ic_file::Union{Nothing,String}=nothing)
    println("Starting simulation: $(settings.theory)")
    initial_states = ic_file === nothing ? generate_initial_conditions(settings) : load_initial_conditions(ic_file)
    sols = ODESolution[]
    for u0 in initial_states
        push!(sols, evol(u0, settings))
    end
    return SimResult(sols, settings)
end

function generate_initial_conditions(settings::SimSettings)
    rng = Xoshiro(settings.seed)
    Ts = rand(rng, Uniform(settings.T_range...), settings.n_points)
    As = rand(rng, Uniform(settings.A_range...), settings.n_points)
    if settings.theory == :HJSW
        Zs = zeros(settings.n_points)
        return [[Ts[i], As[i], Zs[i]] for i in 1:settings.n_points]
    else
        return [[Ts[i], As[i]] for i in 1:settings.n_points]
    end
end

function load_initial_conditions(filepath::String)
    df = endswith(filepath, ".csv") ? CSV.read(filepath, DataFrame; comment="#") :
         endswith(filepath, ".h5") ? h5open(filepath,"r") do file
             g = file["initial_conditions"]
             DataFrame([col => read(g[col]) for col in names(g)])
         end : error("Unsupported file type")
    if :Z_0 in names(df)
        return [[row.T_0,row.A_0,row.Z_0] for row in eachrow(df)]
    else
        return [[row.T_0,row.A_0] for row in eachrow(df)]
    end
end

function generate_and_save_ics(; settings::SimSettings, output_base_filename="initial_conditions")
    ics_list = generate_initial_conditions(settings)
    df = DataFrame(T_0=[ic[1] for ic in ics_list], A_0=[ic[2] for ic in ics_list])
    if settings.theory == :HJSW
        df.Z_0 = [ic[3] for ic in ics_list]
    end
    df.Run_ID = 1:settings.n_points

    CSV.write("$(output_base_filename).csv", df)

    h5open("$(output_base_filename).h5", "w") do file
        g_data = create_group(file, "initial_conditions")
        for col in names(df)
            g_data[col] = df[!, col]
        end
        attrs(g_data)["timestamp"] = string(now())
    end
    return df
end

function extract_phase_space_slice(simres::SimResult, t::Float64)
    params = simres.settings.params
    ode! = simres.settings.ode
    n_sols = length(simres.solutions)
    n_vars = length(simres.solutions[1].u[1])
    u_vals = [fill(NaN, n_sols) for _ in 1:n_vars]
    du_vals = [fill(NaN, n_sols) for _ in 1:n_vars]
    valid = falses(n_sols)
    cache = zeros(n_vars)

    for (i, sol) in enumerate(simres.solutions)
        if t < sol.t[1] || t > sol.t[end]
            continue
        end
        u = sol(t)
        if any(!isfinite, u) || u[1] <= 1e-9
            continue
        end

        ode!(cache, u, params, t)
        for j in 1:n_vars
            u_vals[j][i] = u[j]
            du_vals[j][i] = cache[j]
        end
        valid[i] = true
    end
    return (u_vals, du_vals, valid)
end

function Base.show(io::IO, sim::SimResult)
    succ = count(s -> string(s.retcode) == "Success", sim.solutions)
    n = length(sim.solutions)
    print(io, "SimResult(theory=$(sim.settings.theory), $succ/$n sukcesów, zakres T=$(sim.settings.T_range))")
end

end # module
