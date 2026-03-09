using DifferentialEquations
# using Trixie
"""
Solve  hydro equasions for a list of initial conditions.
"""
function generate_trajectories(
    model::AbstractHydroModel,
    initial_conditions::AbstractVector{<:AbstractVector{<:Real}},
    tspan::Tuple{<:Real,<:Real};
    saveat=nothing,
    parallel::Symbol=:threads,
)
    @assert !isempty(initial_conditions) "At least one initial condition is required."

    first_solution = solve_hydro(model, initial_conditions[1], tspan; saveat=saveat)
    solT = typeof(first_solution)
    solutions = Vector{solT}(undef, length(initial_conditions))
    solutions[1] = first_solution

    if length(initial_conditions) == 1
        return solutions
    end

    base_problem = first_solution.prob
    function prob_func(prob, i, _)
        remake(prob, u0=initial_conditions[i])
    end

    ensemble_problem = EnsembleProblem(base_problem; prob_func=prob_func)
    ensemble_alg = parallel === :threads ? EnsembleThreads() : EnsembleSerial()
    solve_kwargs = (
        trajectories=length(initial_conditions),
        abstol=1e-8,
        reltol=1e-8,
    )
    ensemble_solution = if isnothing(saveat)
        solve(ensemble_problem, Tsit5(), ensemble_alg; solve_kwargs...)
    else
        solve(ensemble_problem, Tsit5(), ensemble_alg; solve_kwargs..., saveat=saveat)
    end
    @inbounds for i in eachindex(solutions)
        solutions[i] = ensemble_solution.u[i]
    end
    return solutions
end

"""
Build a dense matrix dataset from solution snapshots.
Rows are samples, columns are [tau, T, A].
Keyword `temperature_unit` controls T output unit (`:fm` or `:MeV`).
```julia

function build_dataset(solutions::AbstractVector)
    n_rows = 0

    data = Matrix{Float64}(undef, n_rows, 3)
    @inbounds for sol in solutions
        for i in eachindex(sol.t)
        ###
        end
    end
    return data
end
```
"""
function build_dataset(solutions::AbstractVector; temperature_unit::Symbol=:fm)
    rows = Vector{NTuple{3,Float64}}()
    sizehint!(rows, sum(length(sol.t) for sol in solutions))

    @inbounds for sol in solutions
        for i in eachindex(sol.t)
            state = sol.u[i]
            τ = Float64(sol.t[i])
            T = to_temperature_unit(state[1], temperature_unit)
            A = Float64(state[2])

            if isfinite(τ) && isfinite(T) && isfinite(A)
                push!(rows, (τ, T, A))
            end
        end
    end

    data = Matrix{Float64}(undef, length(rows), 3)
    @inbounds for i in eachindex(rows)
        data[i, 1], data[i, 2], data[i, 3] = rows[i]
    end

    return data
end
