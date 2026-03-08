
"""
BRSSS model configuration.
"""
struct BRSSSModel{T<:Real} <: AbstractHydroModel
    params::HydroParams{T}
end

"""
Create BRSSS model with optional transport parameters.
"""
function BRSSSModel(; eta_over_s::Real=1 / (4 * π), tau_pi::Real=(2 - log(2)) / (2 * π), lambda1::Real=1 / (2 * π))
    T = promote_type(typeof(eta_over_s), typeof(tau_pi), typeof(lambda1))
    params = HydroParams(T(eta_over_s), T(tau_pi), T(lambda1))
    return BRSSSModel(params)
end
