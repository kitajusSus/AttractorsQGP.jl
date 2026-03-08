"""
MIS model as BRSSS with lambda1 = 0.
"""
struct MISModel{T<:Real} <: AbstractHydroModel
    params::HydroParams{T}
end

"""
Create MIS model with optional transport parameters.
"""
function MISModel(; eta_over_s::Real=1 / (4 * π), tau_pi::Real=(2 - log(2)) / (2 * π))
    T = promote_type(typeof(eta_over_s), typeof(tau_pi))
    params = HydroParams(T(eta_over_s), T(tau_pi), zero(T))
    return MISModel(params)
end
