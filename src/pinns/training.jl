using Lux
using Optimisers
using Zygote
using Random
using ProgressMeter
using ComponentArrays

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

"""
    sample_collocation_batch(rng, n, model, config) -> Vector{NamedTuple}

Sample `n` collocation points for the physics loss.

Each point is a `NamedTuple` with fields:
- `τ`  : proper time  (fm/c)
- `T₀` : temperature at τ₀  (fm⁻¹)
- `A₀` : anisotropy  at τ₀  (dimensionless)
- `η`  : η/s (transport coefficient)
- `τπ` : τ_π (relaxation time, fm/c)

`η` and `τπ` are sampled uniformly from `config.η_range` / `config.τπ_range`
so that the PINN learns a single surrogate valid for the whole model family.
`lambda1` is inherited from `model.params.lambda1` via `_make_brsss_from`.
"""
function sample_collocation_batch(
        rng::AbstractRNG,
        n::Int,
        config::PINNConfig,
    )
    _rnd(lo, hi) = lo + rand(rng) * (hi - lo)
    return [
        (
                τ = _rnd(config.τ_range...),
                T₀ = _rnd(config.T_range...),
                A₀ = _rnd(config.A_range...),
                η = _rnd(config.η_range...),
                τπ = _rnd(config.τπ_range...),
            )
            for _ in 1:n
    ]
end

"""
    sample_ic_batch(rng, n, config) -> Vector{NamedTuple}

Sample `n` initial-condition points (all evaluated at τ = τ_range[1]).
"""
function sample_ic_batch(
        rng::AbstractRNG,
        n::Int,
        config::PINNConfig,
    )
    _rnd(lo, hi) = lo + rand(rng) * (hi - lo)
    return [
        (
                T₀ = _rnd(config.T_range...),
                A₀ = _rnd(config.A_range...),
                η = _rnd(config.η_range...),
                τπ = _rnd(config.τπ_range...),
            )
            for _ in 1:n
    ]
end

# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

"""
    PINNResult

Holds the trained network and all metadata needed for inference.

# Fields
- `network`:       Lux model (immutable architecture)
- `ps`:            trained parameters as `ComponentArray`
- `st`:            Lux model state (typically empty for MLP)
- `config`:        normalization config used during training
- `loss_history`:  scalar loss at each epoch
- `reference_model`: model used during training (determines `lambda1`)
"""
struct PINNResult{M, P, S, C, R}
    network::M
    ps::P
    st::S
    config::C
    loss_history::Vector{Float64}
    reference_model::R
end

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

"""
    train_pinn(model, config; kwargs...) -> PINNResult

Train a PINN surrogate for Bjorken flow.

The network learns the mapping:
    (τ, T₀, A₀, η/s, τ_π)  →  (T(τ), A(τ))

satisfying the BRSSS/MIS ODE residual and initial conditions as soft constraints.

# Keyword arguments
| Argument              | Default  | Description                                    |
|-----------------------|----------|------------------------------------------------|
| `n_epochs`            | 5000     | number of gradient steps                       |
| `batch_size_colloc`   | 256      | collocation points per step (physics loss)     |
| `batch_size_ic`       | 64       | IC points per step                             |
| `learning_rate`       | 1e-3     | Adam step size                                 |
| `λ_physics`           | 1.0      | weight for physics residual loss               |
| `λ_ic`                | 10.0     | weight for IC loss (higher → stricter IC)      |
| `h`                   | 1e-4     | FD step for dU/dτ in normalized τ              |
| `seed`                | 42       | RNG seed for reproducibility                   |
| `verbose`             | true     | show progress bar                              |
| `log_every`           | 100      | print loss every N epochs                      |

# Example
```julia
model = BRSSSModel()
config = PINNConfig()
result = train_pinn(model, config; n_epochs=3000)
```
"""
function train_pinn(
        model::AbstractHydroModel,
        config::PINNConfig = PINNConfig();
        n_epochs::Integer = 5000,
        batch_size_colloc::Integer = 256,
        batch_size_ic::Integer = 64,
        learning_rate::Real = 1.0e-3,
        λ_physics::Real = 1.0,
        λ_ic::Real = 10.0,
        h::Real = 1.0e-4,
        seed::Integer = 42,
        verbose::Bool = true,
        log_every::Integer = 100,
    )
    rng = MersenneTwister(seed)

    # ── Build & initialise network ──────────────────────────────────────────
    nn = build_pinn_network(config)
    ps_raw, st = Lux.setup(rng, nn)
    ps = ComponentArray{Float64}(ps_raw)   # force Float64 (Lux defaults to Float32)   # flat array ← Optimisers wants this

    # ── Optimizer ───────────────────────────────────────────────────────────
    opt = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(opt, ps)

    loss_history = Float64[]
    sizehint!(loss_history, n_epochs)

    prog = verbose ? Progress(n_epochs; desc = "Training PINN: ", showspeed = true) : nothing

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in 1:n_epochs
        batch_c = sample_collocation_batch(rng, batch_size_colloc, config)
        batch_i = sample_ic_batch(rng, batch_size_ic, config)

        # gradient w.r.t. ps  (Zygote reverse mode)
        loss_val, (∇ps,) = Zygote.withgradient(ps) do p
            l, _ = pinn_loss(
                nn, p, st, batch_c, batch_i, model, config;
                λ_physics = λ_physics, λ_ic = λ_ic, h = h
            )
            l
        end

        opt_state, ps = Optimisers.update(opt_state, ps, ∇ps)
        push!(loss_history, loss_val)

        if verbose && !isnothing(prog) && epoch % log_every == 0
            next!(
                prog; showvalues = [
                    (:epoch, epoch),
                    (:loss, round(loss_val, sigdigits = 4)),
                ]
            )
        end
    end

    return PINNResult(nn, ps, st, config, loss_history, model)
end
