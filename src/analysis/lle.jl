"""
Estimate largest Lyapunov exponent from two nearby trajectories.
"""
function run_LLE(
    model::AbstractHydroModel,
    u0::AbstractVector{<:Real},
    tspan::Tuple{<:Real,<:Real};
    perturbation::Real=1e-6,
    saveat::Real=1e-3,
)
    @assert perturbation > 0 "perturbation must be positive."

    u1 = collect(float.(u0))
    u2 = copy(u1)
    u2[1] += perturbation

    sol1 = solve_hydro(model, u1, tspan; saveat=saveat)
    sol2 = solve_hydro(model, u2, tspan; saveat=saveat)

    n = min(length(sol1.t), length(sol2.t))
    if n < 2
        throw(ArgumentError("Not enough samples to estimate LLE."))
    end

    local_sum = 0.0
    count = 0
    for i in 1:n
        d = hypot(sol2.u[i][1] - sol1.u[i][1], sol2.u[i][2] - sol1.u[i][2])
        if d > 0 && isfinite(d)
            local_sum += log(d / perturbation)
            count += 1
        end
    end
    @assert count > 0 "No valid separation points for LLE estimation."
    Δt = (sol1.t[n] - sol1.t[1]) / max(n - 1, 1)
    return (local_sum / count) / Δt
end
