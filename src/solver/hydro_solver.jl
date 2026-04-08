using DifferentialEquations
using StaticArrays

"""
Solve hydrodynamic evolution for one initial condition.
```julia

solve_hydro(model::AbstractHydroModel, u0::AbstractVector{<:Real}, tspan::Tuple{<:Real,<:Real};
    solver=Rodas5(),
    abstol::Real=1e-6,
    reltol::Real=1e-6,
    saveat=nothing,
)

```




"""
function solve_hydro(model::AbstractHydroModel, u0::AbstractVector{<:Real}, tspan::Tuple{<:Real,<:Real};
    solver=Rodas5(),
    abstol::Real=1e-6,
    reltol::Real=1e-6,
    saveat=nothing,
)
    @assert length(u0) == 2 "Initial state must contain [T, A]."
    if tspan[1] >= tspan[2]
        throw(DomainError(tspan, "tspan must satisfy t0 < t1."))
    end

    Tstate = promote_type(eltype(u0), typeof(tspan[1]), typeof(tspan[2]), Float64)
    state0 = SVector{2,Tstate}(u0[1], u0[2])
    problem = ODEProblem(rhs, state0, (Tstate(tspan[1]), Tstate(tspan[2])), model)

    if isnothing(saveat)
        return solve(problem, solver; abstol=abstol, reltol=reltol)
    end
    return solve(problem, solver; abstol=abstol, reltol=reltol, saveat=saveat)

end
