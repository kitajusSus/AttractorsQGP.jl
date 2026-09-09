"""
Batched PINN loss functions.

All collocation points in a mini-batch are stacked into a single 5×N input
matrix and processed in one neural-network call.  This replaces the original
point-by-point loop and gives a ~100× speed-up on CPU.

Physics residual architecture
──────────────────────────────
  1. Build X_batch  (5×N)  from the sampled collocation points
  2. Three batched forward passes:
       Y_c      = nn(X_c,     ps, st)   # predicted state
       Y_plus   = nn(X_plus,  ps, st)   # τ_norm perturbed +h
       Y_minus  = nn(X_minus, ps, st)   # τ_norm perturbed -h
  3. Central FD → dU/dτ  (vectorised, Zygote-safe)
  4. Vectorised Bjorken RHS (no struct creation, pure arithmetic)
  5. MSE between dU_NN and dU_physics
"""

using Statistics


# Scalar (re-)normalisation helpers — already defined in network.jl
# @inline _norm(v, lo, hi)   = 2*(v-lo)/(hi-lo) - 1
# @inline _denorm(v, lo, hi) = (v+1)/2*(hi-lo) + lo

"""
    _build_input_matrix(batch, config) -> Matrix{Float64}  (5 × N)

Stack normalised input vectors for all collocation points into a single
matrix suitable for a batched Lux forward pass.
"""
function _build_input_matrix(batch, config::PINNConfig)
    return hcat(
        [
            normalize_pinn_input(b.τ, b.T₀, b.A₀, b.η, b.τπ, config)
                for b in batch
        ]...
    )
end

function _build_ic_matrix(batch, config::PINNConfig)
    τ₀ = config.τ_range[1]
    return hcat(
        [
            normalize_pinn_input(τ₀, b.T₀, b.A₀, b.η, b.τπ, config)
                for b in batch
        ]...
    )
end

"""
    _bjorken_rhs_vec(T, A, τ, η, τπ, λ1)

Vectorised Bjorken-flow RHS over arrays of physical-unit values.

Returns `(dT, dA)` as vectors of the same length.
"""
function _bjorken_rhs_vec(T, A, τ, η, τπ, λ1::Real)
    dT = @. (T / τ) * (-1 / 3 + A / 18)
    term_T = @. τ * T * (A + (λ1 / (12 * η)) * A^2)
    term_A = @. (2 / 9) * τπ * A^2
    dA = @. (1 / (τπ * τ)) * (8 * η - term_T - term_A)
    return dT, dA
end


function _make_brsss_from(m::BRSSSModel, η::Real, τπ::Real)
    return BRSSSModel(; eta_over_s = η, tau_pi = τπ, lambda1 = m.params.lambda1)
end
function _make_brsss_from(m::MISModel, η::Real, τπ::Real)
    return MISModel(; eta_over_s = η, tau_pi = τπ)
end
function ic_residual(nn, ps, st, T₀, A₀, η, τπ, config::PINNConfig)
    τ₀ = config.τ_range[1]
    u_pred = pinn_predict(nn, ps, st, τ₀, T₀, A₀, η, τπ, config)
    return sum(abs2, u_pred .- [T₀, A₀])
end
"""
    pinn_loss(nn, ps, st, batch_colloc, batch_ic, model, config; λ_physics, λ_ic, h)
    -> (loss::Float64, st)

Batched PINN loss.

All N collocation points are processed with **three** batched forward passes
(one for the state, two for the central FD derivative).  The Bjorken RHS is
evaluated via vectorised arithmetic — no per-point loops, no struct creation
inside the gradient tape.

`batch_colloc` : Vector of NamedTuples  (τ, T₀, A₀, η, τπ)
`batch_ic`     : Vector of NamedTuples  (T₀, A₀, η, τπ)
"""
function pinn_loss(
        nn, ps, st,
        batch_colloc::AbstractVector,
        batch_ic::AbstractVector,
        model::AbstractHydroModel,
        config::PINNConfig;
        λ_physics::Real = 1.0,
        λ_ic::Real = 10.0,
        h::Real = 1.0e-4,
    )
    λ1 = model.params.lambda1
    dτn = 2.0 / (config.τ_range[2] - config.τ_range[1])   # dτ_norm/dτ
    Tlo, Thi = config.T_range
    Alo, Ahi = config.A_range
    Nc = length(batch_colloc)

    # ── Build batched input 5×Nc ──────────────────────────────────────────
    X_c = _build_input_matrix(batch_colloc, config)

    # Central FD: perturb only row 1 (τ_norm)
    X_plus = vcat(X_c[1:1, :] .+ h, X_c[2:end, :])
    X_minus = vcat(X_c[1:1, :] .- h, X_c[2:end, :])

    # Three batched forward passes (vs. Nc × 3 individual calls before)
    Y_c, _ = nn(X_c, ps, st)   # 2×Nc  predicted state
    Y_plus, _ = nn(X_plus, ps, st)   # 2×Nc  +h state
    Y_minus, _ = nn(X_minus, ps, st)   # 2×Nc  -h state

    # Denormalize to physical units
    T_pred = @. _denorm(Y_c[1, :], Tlo, Thi)
    A_pred = @. _denorm(Y_c[2, :], Alo, Ahi)
    T_plus = @. _denorm(Y_plus[1, :], Tlo, Thi)
    A_plus = @. _denorm(Y_plus[2, :], Alo, Ahi)
    T_minus = @. _denorm(Y_minus[1, :], Tlo, Thi)
    A_minus = @. _denorm(Y_minus[2, :], Alo, Ahi)

    # du/dτ via central FD + chain rule
    dT_nn = @. (T_plus - T_minus) / (2h) * dτn
    dA_nn = @. (A_plus - A_minus) / (2h) * dτn

    # Vectorised physics RHS
    τ_c = [b.τ  for b in batch_colloc]
    η_c = [b.η  for b in batch_colloc]
    τπ_c = [b.τπ for b in batch_colloc]

    dT_rhs, dA_rhs = _bjorken_rhs_vec(T_pred, A_pred, τ_c, η_c, τπ_c, λ1)

    L_phys = (sum(abs2, dT_nn .- dT_rhs) + sum(abs2, dA_nn .- dA_rhs)) / Nc

    # ── IC loss ───────────────────────────────────────────────────────────
    Ni = length(batch_ic)
    X_ic = _build_ic_matrix(batch_ic, config)
    Y_ic, _ = nn(X_ic, ps, st)   # 2×Ni

    T_ic_pred = @. _denorm(Y_ic[1, :], Tlo, Thi)
    A_ic_pred = @. _denorm(Y_ic[2, :], Alo, Ahi)
    T_ic_true = [b.T₀ for b in batch_ic]
    A_ic_true = [b.A₀ for b in batch_ic]

    L_ic = (sum(abs2, T_ic_pred .- T_ic_true) + sum(abs2, A_ic_pred .- A_ic_true)) / Ni

    return λ_physics * L_phys + λ_ic * L_ic, st
end
