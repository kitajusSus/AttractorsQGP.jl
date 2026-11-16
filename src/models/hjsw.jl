"""
Konfiguracja modelu HJSW (Heller-Janik-Spaliński-Witaszczyk).
"""
struct HJSWModel{T <: Real} <: AbstractHydroModel
    params::HydroParams{T}
    omega_R::T
    omega_I::T
end

"""
    HJSWModel(; eta_over_s, tau_pi, lambda1, omega_R, omega_I)
"""
function HJSWModel(;
        # eta_over_s::Real = 1 / (4 * π),
        eta_over_s::Real = 0.08,
        # omega_R::Real = 9.8,
        # omega_I::Real = 8.629,
        omega_R::Real = 9.80005,
        omega_I::Real = 2.87631,
    )

    # tau_pi = 1 / (2 * π)
    tau_pi = 0.2
    lambda1 = 0.0
    T = promote_type(typeof(eta_over_s), typeof(tau_pi), typeof(lambda1), typeof(omega_R), typeof(omega_I))
    params = HydroParams(T(eta_over_s), T(tau_pi), T(lambda1))
    return HJSWModel(params, T(omega_R), T(omega_I))
end

function Base.getproperty(model::HJSWModel, sym::Symbol)
    if sym === :C_eta
        return getfield(model, :params).eta_over_s
    else
        return getfield(model, sym)
    end
end

"""
Konfiguracja modelu HJSW w czasie t dla zmiennych u(t) = [w(t), A(t), B(t)].
"""
struct HJSWwModel{T <: Real} <: AbstractHydroModel
    params::HydroParams{T}
    omega_R::T
    omega_I::T
end

"""
    HJSWwModel(; eta_over_s, omega_R, omega_I)
"""
function HJSWwModel(;
        eta_over_s::Real = 0.0795775,
        omega_R::Real = 9.80005,
        omega_I::Real = 2.87631,
    )
    tau_pi = 1 / (2 * π)
    lambda1 = 0.0
    T_type = promote_type(typeof(eta_over_s), typeof(tau_pi), typeof(lambda1), typeof(omega_R), typeof(omega_I))
    params = HydroParams(T_type(eta_over_s), T_type(tau_pi), T_type(lambda1))
    return HJSWwModel(params, T_type(omega_R), T_type(omega_I))
end

function Base.getproperty(model::HJSWwModel, sym::Symbol)
    if sym === :C_eta
        return getfield(model, :params).eta_over_s
    else
        return getfield(model, sym)
    end
end
