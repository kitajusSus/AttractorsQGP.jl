using StaticArrays

"""
In-place RHS for Bjorken flow in BRSSS-like variables u = [T, A].
defined as:


```julia
function rhs(u::AbstractVector{<:Real}, model::AbstractHydroModel, τ::Real)
end
```

Compute the right-hand side of the Bjorken flow equations for
a given state `u`, model parameters, and proper time `τ`.
```julia
function rhs!(du, u, model, τ)
    # Compute dT and dA based on the equations of motion
    du[1] = dT
    du[2] = dA
end
```

returns:

```julia
    return SVector(dT, dA)
```
"""
function rhs(u::AbstractVector{<:Real}, model::AbstractHydroModel, τ::Real)
    @assert length(u) == 2 "State must contain [T, A]."
    if τ <= 0
        throw(DomainError(τ, "Proper time τ must be positive."))
    end

    T, A = u[1], u[2]
    if !isfinite(T) || !isfinite(A)
        throw(DomainError((T, A), "State values must be finite."))
    end

    p = model.params

    dT = (T / τ) * (-one(T) / 3 + A / 18)
    term_T = τ * T * (A + (p.lambda1 / (12 * p.eta_over_s)) * A^2)

    term_A2 = (2 / 9) * p.tau_pi * A^2
    dA = (1 / (p.tau_pi * τ)) * (8 * p.eta_over_s - term_T - term_A2)
    return SVector(dT, dA)
end
