using DifferentialEquations

"""
Solve trajectories for a list of initial conditions.
Use EnsembleProblem for efficient threaded execution.
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
    ensemble_solution = solve(
        ensemble_problem,
        Tsit5(),
        ensemble_alg;
        trajectories=length(initial_conditions),
        saveat=saveat,
        abstol=1e-8,
        reltol=1e-8,
    )

    @inbounds for i in eachindex(solutions)
        solutions[i] = ensemble_solution.u[i]
    end
    return solutions
end

"""
Build a dense matrix dataset from solution snapshots.
Rows are samples, columns are [tau, T, A].
"""
function build_dataset(solutions::AbstractVector)
    n_rows = 0
    @inbounds for sol in solutions
        n_rows += length(sol.t)
    end

    data = Matrix{Float64}(undef, n_rows, 3)
    row = 1
    @inbounds for sol in solutions
        for i in eachindex(sol.t)
            state = sol.u[i]
            data[row, 1] = Float64(sol.t[i])
            data[row, 2] = Float64(state[1])
            data[row, 3] = Float64(state[2])
            row += 1
        end
    end
    return data
end
