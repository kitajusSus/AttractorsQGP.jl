using Random
using StaticArrays

"""
Generate random initial conditions [T, A] as stack-friendly SVectors.
Temperature is always stored in internal `fm` units.
```julia
function generate_initial_conditions(
    n::Integer;
    T_range::Tuple{<:Real,<:Real}=(400.0, 2500.0),
    A_range::Tuple{<:Real,<:Real}=(-8.0, 20.0),
    temperature_unit::Symbol=:MeV,
    rng::AbstractRNG=Random.default_rng(),
)
```
"""
function generate_initial_conditions(
    n::Integer;
    T_range::Tuple{<:Real,<:Real}=(400.0, 2500.0),
    A_range::Tuple{<:Real,<:Real}=(-8.0, 20.0),
    temperature_unit::Symbol=:MeV,
    rng::AbstractRNG=Random.default_rng(),
)
    @assert n > 0 "n must be positive."
    T_min, T_max = T_range
    A_min, A_max = A_range
    @assert T_min < T_max "Invalid T_range."
    @assert A_min < A_max "Invalid A_range."

    ics = Vector{SVector{2,Float64}}(undef, n)
    @inbounds for i in eachindex(ics)
        T0 = rand(rng) * (T_max - T_min) + T_min
        T0_fm = temperature_to_fm(T0, temperature_unit)
        A0 = rand(rng) * (A_max - A_min) + A_min
        ics[i] = SVector{2,Float64}(T0_fm, A0)
    end
    return ics
end
