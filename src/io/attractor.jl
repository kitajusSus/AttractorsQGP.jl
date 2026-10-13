using Interpolations
using CSV
using DataFrames
using HDF5
using Serialization
using StaticArrays
using DifferentialEquations

function attractor_data(
        model::AbstractHydroModel;
        tau_max = 5.0,
        krok = 0.01,
        temperature_unit::Symbol = :fm
    )
    tspan_attr = (0.1, tau_max)
    ic_attr = [4, 20]
    sol_attr = solve_hydro(model, ic_attr, tspan_attr; saveat = krok)
    return build_dataset([sol_attr]; temperature_unit = temperature_unit)
end

function attractor_data(
        model::MISModel;
        tau_max = 5.0,
        krok = 0.01,
        n_interp = 5000,
        w_min::Real = 1.0e-15,
        kwargs...
    )
    w_min_f = Float64(w_min)
    w_max = Float64(tau_max)
    p = model.params

    A0 = Float64(6 * sqrt(p.eta_over_s / p.tau_pi))

    # Prawa strona równania ODE
    rhs_A(u, _, w) = begin
        A_val = u[1]
        dA = 18 * (8 * p.eta_over_s - w * A_val - (2 / 9) * p.tau_pi * A_val^2) /
            (p.tau_pi * w * (A_val + 12))
        return SVector(dA)
    end

    problem = ODEProblem(
        rhs_A,
        SVector(A0),
        (w_min_f, w_max),
        nothing
    )

    sol = solve(problem, Rodas5(); abstol = 1.0e-9, reltol = 1.0e-9)

    ws = range(w_min_f, w_max, length = n_interp)
    A_vals = first.(sol.(ws))

    return [ws ones(length(ws)) A_vals]
end

"""
    attractor_data(; kwargs...)

Returns the theoretical universal attractor data (by default for MISModel).
"""
attractor_data(; kwargs...) = attractor_data(MISModel(); kwargs...)

"""
    _resolve_attractor_data(attr_input, dataset=nothing)

Resolves attractor data from the provided `attractor` or `Attractor` parameter:
- `false` / `nothing`: returns `nothing` (no attractor line)
- `true`: returns default theoretical attractor (`attractor_data(MISModel())`)
- `model::AbstractHydroModel`: computes attractor for given model
- `matrix::AbstractMatrix`: returns the matrix as-is
"""
function _resolve_attractor_data(attr_input, dataset = nothing)
    if isnothing(attr_input) || attr_input === false
        return nothing
    elseif attr_input === true
        return attractor_data(MISModel())
    elseif attr_input isa AbstractHydroModel
        return attractor_data(attr_input)
    elseif attr_input isa AbstractMatrix{<:Real}
        return attr_input
    else
        throw(ArgumentError("Nieobsługiwany typ parametru attractor: $(typeof(attr_input)). Oczekiwano Bool (true/false), AbstractMatrix lub AbstractHydroModel."))
    end
end

"""
    build_attractor_interpolant(attractor)

Constructs a robust, non-extrapolating (Flat) interpolation A(ω) from attractor data.
"""
function build_attractor_interpolant(attractor::AbstractMatrix{<:Real})
    Tfm = attractor[:, 2]
    ω = attractor[:, 1] .* Tfm
    A = attractor[:, 3]

    p = sortperm(ω)
    ωsorted = ω[p]
    Asorted = A[p]

    # return linear_interpolation(
    #     ωsorted,
    #     Asorted;
    #     extrapolation_bc = Line()
    # )

    return linear_interpolation(
        ωsorted,
        Asorted;
        extrapolation_bc = Flat()
    )
end
"""
    temperature_grid(Tmin, Tmax, n)

Construct a uniform temperature grid of length `n` with physical positivity safety checks.
"""
function temperature_grid(T_values::AbstractVector{<:Real}, n::Integer)
    # Tmin = minimum(T_values) * 0.95
    # Tmax = maximum(T_values) * 1.05

    Tmin = max(minimum(T_values) * 0.95, 1.0e-6)
    Tmax = maximum(T_values) * 1.05
    return temperature_grid(Tmin, Tmax, n)
end

function temperature_grid(Tmin::Real, Tmax::Real, n::Integer)
    @assert n > 0 "Temperature grid must contain at least one point."

    # if n == 1
    #     return [(Tmin + Tmax) / 2]
    # end
    # return range(Tmin, Tmax; length = n)

    Tmin_safe = max(Tmin, 1.0e-6)
    if n == 1
        return [(Tmin_safe + Tmax) / 2]
    end

    return range(Tmin_safe, Tmax; length = n)
end

"""
    get_attractor_line(...)

Project attractor onto arbitrary observables.
"""
function get_attractor_line(
        dataset::AbstractMatrix{<:Real},
        attractor::AbstractMatrix{<:Real},
        τ::Real,
        xdef,
        ydef;
        n_points::Int = 200
    )
    _, xmap = resolve_def(xdef)
    _, ymap = resolve_def(ydef)

    _, slice = get_tau_slice(
        dataset,
        τ;
        feature_cols = 1:size(dataset, 2)
    )

    # Tmin = minimum(slice[:, 2])
    # Tmax = maximum(slice[:, 2])
    # Tgrid = range(Tmin * 0.0001, Tmax * 10; length = n_points)

    Tgrid = temperature_grid(slice[:, 2], n_points)
    Aω = build_attractor_interpolant(attractor)

    x = Vector{Float64}(undef, n_points)
    y = Vector{Float64}(undef, n_points)

    ncols = size(dataset, 2)
    state = zeros(Float64, ncols)
    state[1] = τ

    for (i, T) in enumerate(Tgrid)
        ω = τ * T
        state[2] = T
        state[3] = Aω(ω)

        x[i] = xmap(state, slice)
        y[i] = ymap(state, slice)
    end

    return (; x, y)
end

function interpolate_attractor_state(
        dataset::AbstractMatrix{S},
        attractor::AbstractMatrix{<:Real},
        τ::Real,
        T::Real
    ) where {S}
    # omega = attractor[:, 1] .* attractor[:, 2]
    # order = sortperm(omega)
    # omega_sorted = omega[order]
    # attr_sorted = attractor[order, :]
    # target = τ * T
    # ...

    Aω = build_attractor_interpolant(attractor)
    target = τ * T

    state = Vector{S}(undef, size(dataset, 2))
    state[1] = τ
    state[2] = T
    state[3] = Aω(target)

    if size(dataset, 2) > 3
        state[4:end] .= zero(S)
    end
    return state
end

"""
    get_attractor_line_for_frame(dataset::AbstractMatrix{<:Real}, attractor_universe::AbstractMatrix{<:Real}, t::Real, xdef, ydef; limits=nothing, n_points=150)

Oblicza dokładne współrzędne linii teoretycznego atraktora dla wybranej klatki czasowej `t`.
"""
function get_attractor_line_for_frame(
        dataset::AbstractMatrix{<:Real},
        attractor_universe::AbstractMatrix{<:Real},
        t::Real,
        xdef,
        ydef;
        limits = nothing,
        n_points::Int = 150
    )
    _, xfn = resolve_def(xdef)
    _, yfn = resolve_def(ydef)

    _, cloud_slice = get_tau_slice(
        dataset,
        t;
        atol = 1.0e-8,
        feature_cols = 1:size(dataset, 2),
        nearest = true,
    )
    τ = cloud_slice[1, 1]

    get_T_from_val = (val, def) -> begin
        if def === :T
            return val
        elseif def === :tauT
            return val / τ
        elseif def === :T_norm
            return val * maximum(cloud_slice[:, 2])
        elseif def === :tauT_heller
            return val / 0.2
        else
            return val
        end
    end

    Tmin, Tmax = if !isnothing(limits) && length(limits) == 4
        xmin, xmax, ymin, ymax = limits
        if xdef in (:T, :tauT, :T_norm, :tauT_heller)
            get_T_from_val(xmin, xdef), get_T_from_val(xmax, xdef)
        elseif ydef in (:T, :tauT, :T_norm, :tauT_heller)
            get_T_from_val(ymin, ydef), get_T_from_val(ymax, ydef)
        else
            minimum(dataset[:, 2]) * 0.95, maximum(dataset[:, 2]) * 1.05
        end
    else
        minimum(dataset[:, 2]) * 0.95, maximum(dataset[:, 2]) * 1.05
    end

    if Tmin > Tmax
        Tmin, Tmax = Tmax, Tmin
    end

    T_grid = temperature_grid(Tmin, Tmax, n_points)

    # for (i, T) in enumerate(T_grid)
    #     state = interpolate_attractor_state(dataset, attractor_universe, τ, T)
    #     x[i] = xfn(state, cloud_slice)
    #     y[i] = yfn(state, cloud_slice)
    # end

    Aω = build_attractor_interpolant(attractor_universe)

    x = Vector{Float64}(undef, n_points)
    y = Vector{Float64}(undef, n_points)

    ncols = size(dataset, 2)
    state = zeros(Float64, ncols)
    state[1] = τ

    for (i, T) in enumerate(T_grid)
        state[2] = T
        state[3] = Aω(τ * T)

        x[i] = xfn(state, cloud_slice)
        y[i] = yfn(state, cloud_slice)
    end

    return (; x, y)
end
