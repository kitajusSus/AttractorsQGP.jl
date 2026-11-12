using StaticArrays
using OrdinaryDiffEqRosenbrock

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
Zwraca 

```julia 
    if isnothing(saveat)
        return solve(problem, solver; abstol = abstol, reltol = reltol)
    end
    return solve(problem, solver; abstol = abstol, reltol = reltol, saveat = saveat)



```

"""
function solve_hydro(
        model::AbstractHydroModel, u0::AbstractVector{<:Real}, tspan::Tuple{<:Real, <:Real};
        solver = Rodas5(),
        abstol::Real = 1.0e-6,
        reltol::Real = 1.0e-6,
        saveat = nothing,
    )
    N = length(u0)
    Tstate = promote_type(eltype(u0), typeof(tspan[1]), typeof(tspan[2]), Float64)
    # T0_val = u0[1] > 50.0 ? u0[1] / 197.3269804 : u0[1]
    # state0 = setindex(SVector{N, Tstate}(u0...), Tstate(T0_val), 1)
    state0 = SVector{N, Tstate}(u0...)
    problem = ODEProblem(rhs, state0, (Tstate(tspan[1]), Tstate(tspan[2])), model)

    if isnothing(saveat)
        return solve(problem, solver; abstol = abstol, reltol = reltol)
    end
    return solve(problem, solver; abstol = abstol, reltol = reltol, saveat = saveat)
end
