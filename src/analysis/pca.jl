using LinearAlgebra
using Statistics
using Random
using MultivariateStats

"""
    explained_variance_ratio_from_svd(S::AbstractVector{<:Real}, n_samples::Integer)

Compute explained variance ratios from singular values `S` for a data matrix
with `n_samples` rows.

Returns a vector of ratios summing to 1.0 when total variance is positive.
"""
function explained_variance_ratio_from_svd(S::AbstractVector{<:Real}, n_samples::Integer)
    @assert n_samples > 1 "Need at least two samples."
    λ = (Float64.(S) .^ 2) ./ (n_samples - 1)
    total = sum(λ)
    return total <= eps(Float64) ? zeros(Float64, length(λ)) : λ ./ total
end

"""
    normalize_minmax(X::AbstractMatrix{<:Real})

Apply column-wise min-max normalization to `X`.

For constant columns, denominator is set to 1.0 to avoid division by zero.

Returns `(Xn, mn, mx)` where:
- `Xn` is normalized data
- `mn` is per-column minimum
- `mx` is per-column maximum
"""
function normalize_minmax(X::AbstractMatrix{<:Real})
    Xf = Matrix{Float64}(X)
    mn = minimum(Xf, dims=1)
    mx = maximum(Xf, dims=1)
    r = mx .- mn
    r[r .== 0.0] .= 1.0
    Xn = (Xf .- mn) ./ r
    return Xn, mn, mx
end

"""
    run_pca(data::AbstractMatrix{<:Real}; n_components::Integer=2)

Run linear PCA using SVD after column-wise min-max normalization.

Returns a named tuple with fields:
- `transformed`
- `components`
- `explained_variance_ratio`
- `explained_variance_ratio_full`
- `minvals`
- `maxvals`
"""
function run_pca(data::AbstractMatrix{<:Real}; n_components::Integer=2)
    @assert size(data, 1) > 1 "Need at least two samples."
    @assert size(data, 2) > 0 "Need at least one feature."

    Xn, mn, mx = normalize_minmax(data)
    Xc = Xn .- mean(Xn; dims=1)
    _, S, V = svd(Xc)

    k = min(n_components, size(V, 2))
    components = V[:, 1:k]
    transformed = Xc * components

    ratio_full = explained_variance_ratio_from_svd(S, size(Xc, 1))

    return (
        transformed=transformed,
        components=components,
        explained_variance_ratio=ratio_full[1:k],
        explained_variance_ratio_full=ratio_full,
        minvals=vec(mn),
        maxvals=vec(mx),
    )
end

"""
    run_pca_kernel(data::AbstractMatrix{<:Real}; n_components::Integer=2, gamma::Float64=1.0)

Run RBF kernel PCA for `data`.

Returns a named tuple with fields:
- `transformed`
- `components`
- `explained_variance_ratio`
- `explained_variance_ratio_full`


For large `n_samples`, full kernel PCA may require O(n_samples^2) memory.
`max_kernel_gb` limits estimated Gram-matrix size.
"""
function run_pca_kernel(
    data::AbstractMatrix{<:Real};
    n_components::Integer=2,
    gamma::Float64=1.0,
    max_kernel_gb::Float64=8.0,
)
    @assert size(data, 1) > 1 "Need at least two samples."
    @assert size(data, 2) > 0 "Need at least one feature."
    @assert n_components ≥ 1 "n_components must be >= 1."
    @assert gamma > 0 "gamma must be > 0."

    X = Matrix{Float64}(data)
    n, p = size(X)
    k = min(Int(n_components), n, p)

    # Memory guard for dense Gram matrix K (n x n, Float64)
    estimated_bytes = n * n * sizeof(Float64)
    limit_bytes = max_kernel_gb * 1024^3
    if estimated_bytes > limit_bytes
        est_gb = estimated_bytes / 1024^3
        throw(ArgumentError(
            "Kernel PCA requires dense Gram matrix of size $(n)x$(n) (~$(round(est_gb, digits=2)) GB). " *
            "This exceeds max_kernel_gb=$(max_kernel_gb). " *
            "Use fewer samples (subsampling), lower max_kernel_gb only if RAM allows, or use linear PCA."
        ))
    end

    kernel = (x, y) -> exp(-gamma * sum(abs2, x .- y))
    model = fit(KernelPCA, X'; kernel=kernel, maxoutdim=k)
    transformed = MultivariateStats.transform(model, X')'

    ev = Float64.(eigvals(model))
    total = sum(ev)
    ratio_full = total <= eps(Float64) ? zeros(Float64, length(ev)) : ev ./ total
    ratio = ratio_full[1:min(k, length(ratio_full))]

    components = try
        projection(model)
    catch
        Matrix{Float64}(undef, p, 0)
    end

    return (
        transformed=transformed,
        components=components,
        explained_variance_ratio=ratio,
        explained_variance_ratio_full=ratio_full,
    )
end

"""
    get_tau_slice(taus::AbstractVector{<:Real}, tau::Real; atol::Real=1e-8)

Return indices of elements in `taus` matching `tau` within `atol`.

This helper is useful when you already extracted the first column of a dataset.
"""
function get_tau_slice(taus::AbstractVector{<:Real}, tau::Real; atol::Real=1e-8)
    return findall(isapprox.(taus, tau; atol=atol))
end

"""
    get_tau_slice(dataset::AbstractMatrix{<:Real}, tau::Real; atol::Real=1e-8)

Extract rows for a given time value `tau` from a dataset structured as
`[tau, features...]`.

Returns `(idx, Xtau)` where:
- `idx` are row indices matching `tau`
- `Xtau` is the feature matrix for that time

Use `feature_cols` to choose which feature columns (from `2:size(dataset,2)`) are included.
"""
function get_tau_slice(dataset::AbstractMatrix{<:Real}, tau::Real; atol::Real=1e-8, feature_cols::AbstractVector{<:Integer}=collect(2:size(dataset, 2)))
    @assert !isempty(feature_cols) "feature_cols must contain at least one column index."
    @assert all(2 <= c <= size(dataset, 2) for c in feature_cols) "feature_cols must point to feature columns (2:size(dataset, 2))."

    idx = get_tau_slice(view(dataset, :, 1), tau; atol=atol)
    Xtau = Matrix{Float64}(dataset[idx, feature_cols])
    return idx, Xtau
end

"""
    run_pca_per_time(
        dataset::AbstractMatrix{<:Real};
        n_components::Integer=2,
        method::Symbol=:minmax,
        gamma::Float64=1.0,
        atol::Real=1e-8,
    )

Run PCA independently for each unique `tau` in a dataset with layout
`[tau, features...]`.

Each time slice is treated as a separate coordinate system with its own PCA fit.

Returns a named tuple:
- `taus`
- `pca_results`
- `explained_variance_ratio`

Use `feature_cols` to select which variables are used in every time-slice PCA.
"""
function run_pca_per_time(
    dataset::AbstractMatrix{<:Real};
    n_components::Integer=2,
    method::Symbol=:minmax,
    gamma::Float64=1.0,
    atol::Real=1e-8,
    feature_cols::AbstractVector{<:Integer}=collect(2:size(dataset, 2)),
)
    @assert size(dataset, 2) >= 3 "Dataset must contain at least [tau, features...]."

    taus = sort(unique(Float64.(dataset[:, 1])))
    pca_results = Dict{Float64, NamedTuple}()
    evr = fill(NaN, length(taus), n_components)

    for (i, tau) in pairs(taus)
        _, Xtau = get_tau_slice(dataset, tau; atol=atol, feature_cols=feature_cols)

        if size(Xtau, 1) > 1
            res = if method === :minmax
                run_pca(Xtau; n_components=n_components)
            elseif method === :kernel
                run_pca_kernel(Xtau; n_components=n_components, gamma=gamma)
            else
                error("Unknown PCA method. Choose :minmax or :kernel.")
            end

            pca_results[tau] = res
            n_local = min(n_components, length(res.explained_variance_ratio))
            evr[i, 1:n_local] .= res.explained_variance_ratio[1:n_local]
        end
    end

    return (taus=taus, pca_results=pca_results, explained_variance_ratio=evr)
end

function run_pca_per_time(
    dataset::AbstractMatrix{<:Real};
    n_components::Integer=2,
    method::Symbol=:minmax,
    gamma::Float64=1.0,
    atol::Real=1e-8,
    feature_cols::AbstractVector{<:Integer}=collect(2:size(dataset, 2)),
)
    @assert size(dataset, 2) >= 3 "Dataset must contain at least [tau, features...]."

    taus = sort(unique(Float64.(dataset[:, 1])))
    pca_results = Dict{Float64, NamedTuple}()
    evr = fill(NaN, length(taus), n_components)

    for (i, tau) in pairs(taus)
        _, Xtau = get_tau_slice(dataset, tau; atol=atol, feature_cols=feature_cols)

        if size(Xtau, 1) > 1
            res = if method === :minmax
                run_pca(Xtau; n_components=n_components)
            elseif method === :kernel
                run_pca_kernel(Xtau; n_components=n_components, gamma=gamma)
            else
                error("Unknown PCA method. Choose :minmax or :kernel.")
            end

            pca_results[tau] = res
            n_local = min(n_components, length(res.explained_variance_ratio))
            evr[i, 1:n_local] .= res.explained_variance_ratio[1:n_local]
        end
    end

    return (taus=taus, pca_results=pca_results, explained_variance_ratio=evr)
end

function run_pca_per_time(
    dataset::AbstractMatrix{<:Real};
    n_components::Integer=2,
    method::Symbol=:minmax,
    gamma::Float64=1.0,
    atol::Real=1e-8,
    feature_cols::AbstractVector{<:Integer}=collect(2:size(dataset, 2)),
)
    @assert size(dataset, 2) >= 3 "Dataset must contain at least [tau, features...]."

    taus = sort(unique(Float64.(dataset[:, 1])))
    pca_results = Dict{Float64, NamedTuple}()
    evr = fill(NaN, length(taus), n_components)

    for (i, tau) in pairs(taus)
        _, Xtau = get_tau_slice(dataset, tau; atol=atol, feature_cols=feature_cols)

        if size(Xtau, 1) > 1
            res = if method === :minmax
                run_pca(Xtau; n_components=n_components)
            elseif method === :kernel
                run_pca_kernel(Xtau; n_components=n_components, gamma=gamma)
            else
                error("Unknown PCA method. Choose :minmax or :kernel.")
            end

            pca_results[tau] = res
            n_local = min(n_components, length(res.explained_variance_ratio))
            evr[i, 1:n_local] .= res.explained_variance_ratio[1:n_local]
        end
    end

    return (taus=taus, pca_results=pca_results, explained_variance_ratio=evr)
end


"""
    run_pca_for_tau(
        dataset::AbstractMatrix{<:Real},
        tau::Real;
        n_components::Integer=2,
        method::Symbol=:minmax,
        gamma::Float64=1.0,
        atol::Real=1e-8,
        feature_cols::AbstractVector{<:Integer}=collect(2:size(dataset, 2)),
    )

Run PCA for one selected `tau` value using all points from that time slice.
"""
function run_pca_for_tau(
    dataset::AbstractMatrix{<:Real},
    tau::Real;
    n_components::Integer=2,
    method::Symbol=:minmax,
    gamma::Float64=1.0,
    atol::Real=1e-8,
    feature_cols::AbstractVector{<:Integer}=collect(2:size(dataset, 2)),
)
    _, Xtau = get_tau_slice(dataset, tau; atol=atol, feature_cols=feature_cols)
    @assert size(Xtau, 1) > 1 "Need at least two points in the selected tau slice."

    res = if method === :minmax
        run_pca(Xtau; n_components=n_components)
    elseif method === :kernel
        run_pca_kernel(Xtau; n_components=n_components, gamma=gamma)
    else
        error("Unknown PCA method. Choose :minmax or :kernel.")
    end

    return (tau=tau, n_points=size(Xtau, 1), pca_result=res)
end


"""
    run_evolution_pca_workflow(
        model::AbstractHydroModel;
        n_points::Integer=1000,
        tspan::Tuple{<:Real,<:Real}=(0.22, 1.2),
        T_range::Tuple{<:Real,<:Real}=(400.0, 2500.0),
        A_range::Tuple{<:Real,<:Real}=(-13.0, 20.0),
        saveat::Union{Real, AbstractVector{<:Real}, Nothing}=0.01,
        temperature_unit::Symbol=:fm,
        n_components::Integer=2,
        method::Symbol=:minmax,
        gamma::Float64=1.0,
        feature_cols::AbstractVector{<:Integer}=[2, 3],
        atol::Real=1e-8,
        parallel::Symbol=:threads,
        seed::Integer=5,
        rng::Union{AbstractRNG,Nothing}=nothing,
    )

Run the full workflow in three explicit steps:
1) sample initial conditions
2) solve ODE trajectories for all points
3) run PCA independently for each `tau` slice and collect EVR over time

Returns a named tuple with:
- `initial_conditions`
- `solutions`
- `dataset`
- `pca_over_time`
"""
function run_evolution_pca_workflow(
    model::AbstractHydroModel;
    n_points::Integer=1000,
    tspan::Tuple{<:Real,<:Real}=(0.22, 1.2),
    T_range::Tuple{<:Real,<:Real}=(400.0, 2500.0),
    A_range::Tuple{<:Real,<:Real}=(-13.0, 20.0),
    saveat::Union{Real, AbstractVector{<:Real}, Nothing}=0.01,
    temperature_unit::Symbol=:fm,
    n_components::Integer=2,
    method::Symbol=:minmax,
    gamma::Float64=1.0,
    feature_cols::AbstractVector{<:Integer}=[2, 3],
    atol::Real=1e-8,
    parallel::Symbol=:threads,
    seed::Integer=5,
    rng::Union{AbstractRNG,Nothing}=nothing,
)
    initial_conditions = generate_initial_conditions(
        n_points;
        T_range=T_range,
        A_range=A_range,
        seed=seed,
        rng=rng,
    )

    solutions = generate_trajectories(
        model,
        initial_conditions,
        tspan;
        saveat=saveat,
        parallel=parallel,
    )

    dataset = build_dataset(solutions; temperature_unit=temperature_unit)

    pca_over_time = run_pca_per_time(
        dataset;
        n_components=n_components,
        method=method,
        gamma=gamma,
        feature_cols=feature_cols,
        atol=atol,
    )

    return (
        initial_conditions=initial_conditions,
        solutions=solutions,
        dataset=dataset,
        pca_over_time=pca_over_time,
    )
end
