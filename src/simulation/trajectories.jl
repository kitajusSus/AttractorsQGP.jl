using DifferentialEquations

"""
    generate_trajectories(model, initial_conditions, tspan; saveat=nothing, parallel=:serial)

Generuje wektor rozwiązań równań hydrodynamiki dla zadanych warunków początkowych.
"""
function generate_trajectories(
        model::AbstractHydroModel,
        initial_conditions::AbstractVector{<:AbstractVector{<:Real}},
        tspan::Tuple{<:Real, <:Real};
        saveat = nothing,
        parallel = nothing,
        abstol::Real = 1.0e-8,
        reltol::Real = 1.0e-8,
        solver = Rodas5(),
    )
    first_initial_state = SVector{length(initial_conditions[1]), Float64}(initial_conditions[1]...)
    base_problem = ODEProblem(rhs, first_initial_state, tspan, model)

    function problem_generator_function(problem, second_argument, remaining_arguments...)
        trajectory_index = isempty(remaining_arguments) ? (second_argument isa Integer ? second_argument : second_argument.sim_id) : second_argument
        return remake(problem, u0 = initial_conditions[trajectory_index])
    end

    ensemble_problem = EnsembleProblem(base_problem; prob_func = problem_generator_function)

    extra_keyword_arguments = isnothing(saveat) ? (;) : (; saveat)
    ensemble_solution = solve(
        ensemble_problem,
        solver,
        EnsembleThreads();
        trajectories = length(initial_conditions),
        abstol = abstol,
        reltol = reltol,
        extra_keyword_arguments...
    )

    return ensemble_solution.u
end

function build_dataset(
        solutions::AbstractVector;
        temperature_unit::Symbol = :fm,
        has_temperature::Bool = true,
        include_tau::Bool = false,
        include_w::Bool = false,
        include_temperature::Bool = true,
    )
    state_dim = length(solutions[1].u[1])
    has_extra = has_temperature && (include_tau || include_w)

    n_features = if has_extra
        include_temperature ? (2 + state_dim) : (1 + state_dim)
    elseif has_temperature
        include_temperature ? (1 + state_dim) : state_dim
    else
        1 + state_dim
    end

    n_trajectories = length(solutions)
    n_time_steps = length(solutions[1].t)

    data = Array{Float64, 3}(undef, n_trajectories, n_time_steps, n_features)

    @inbounds for (traj_idx, sol) in enumerate(solutions)
        for time_idx in 1:n_time_steps
            if time_idx <= length(sol.t) && time_idx <= length(sol.u)
                t_var = sol.t[time_idx]

                if has_extra
                    # Determine tau and w
                    T_fm = sol.u[time_idx][1]
                    if include_tau
                        tau = t_var / T_fm
                        w = t_var
                    else
                        tau = t_var
                        w = tau * T_fm
                    end
                    T = temperature_unit === :MeV ? T_fm * MEV_PER_FM : T_fm

                    if include_temperature
                        data[traj_idx, time_idx, 1] = tau
                        data[traj_idx, time_idx, 2] = T
                        data[traj_idx, time_idx, 3] = w
                        for col in 4:n_features
                            data[traj_idx, time_idx, col] = sol.u[time_idx][col - 2]
                        end
                    else
                        data[traj_idx, time_idx, 1] = tau
                        data[traj_idx, time_idx, 2] = w
                        for col in 3:n_features
                            data[traj_idx, time_idx, col] = sol.u[time_idx][col - 1]
                        end
                    end
                elseif has_temperature
                    data[traj_idx, time_idx, 1] = t_var
                    T = sol.u[time_idx][1]
                    if temperature_unit === :fm
                        # standard fm units
                    elseif temperature_unit === :MeV
                        T = T * MEV_PER_FM
                    else
                        throw(ArgumentError("Unsupported temperature_unit: $temperature_unit. Expected :fm or :MeV."))
                    end

                    if include_temperature
                        data[traj_idx, time_idx, 2] = T
                        for col in 3:n_features
                            data[traj_idx, time_idx, col] = sol.u[time_idx][col - 1]
                        end
                    else
                        for col in 2:n_features
                            data[traj_idx, time_idx, col] = sol.u[time_idx][col]
                        end
                    end
                else
                    data[traj_idx, time_idx, 1] = t_var
                    for col in 2:n_features
                        data[traj_idx, time_idx, col] = sol.u[time_idx][col - 1]
                    end
                end
            else
                # Pad with NaNs if trajectory is shorter
                for col in 1:n_features
                    data[traj_idx, time_idx, col] = NaN
                end
            end
        end
    end

    return data
end
