using LinearAlgebra
using NearestNeighbors
using Random
using Statistics


@views function dims(X; k = 12, tol = 1.0e-8)
    n, d = size(X)
    k = min(k, n)

    tree = KDTree(X')
    out = zeros(Float64, n)

    Y_buf = zeros(Float64, k, d)
    C_buf = zeros(Float64, d, d)

    knn_idxs, _ = knn(tree, X', k, true)

    for i in 1:n
        idxs = knn_idxs[i]

        Y = Y_buf[1:k, :]
        for r in 1:k
            for c in 1:d
                Y[r, c] = X[idxs[r], c]
            end
        end

        μ = mean(Y, dims = 1)
        Y .-= μ

        C = C_buf[1:d, 1:d]
        mul!(C, Y', Y)
        C ./= k

        λ = eigvals!(Symmetric(C))
        sum_λ = sum(λ)
        if sum_λ > 0.0
            cnt = 0
            for val in λ
                if (val / sum_λ) > tol
                    cnt += 1
                end
            end
            out[i] = cnt
        else
            out[i] = 1.0
        end
    end
    return out
end

function normalize_max(X)
    max_per_column = maximum(abs, X, dims = 1)
    max_per_column[max_per_column .== 0.0] .= 1.0
    return X ./ max_per_column
end

"""
    apply_normalization(X::AbstractMatrix{<:Real}, method::Union{Symbol, Function} = :max)

Applies data normalization / regularization to matrix `X`.

Supported methods:
- `:max`: Divides each column by max absolute value (rescales to [-1, 1]).
- `:minmax`: Min-Max column normalization (rescales to [0, 1]).
- `:zscore` / `:standard`: Zero mean, unit variance.
- `:none` / `:raw`: Unscaled raw matrix.
- Custom `Function`: User-supplied scaling function `f(X) -> X_scaled`.
"""
function apply_normalization(X::AbstractMatrix{<:Real}, method::Union{Symbol, Function, AbstractVector{<:Real}} = :standard)
    if method isa AbstractVector{<:Real}
        return apply_frozen_scales(X, method)
    elseif method isa Function
        return method(X)
    elseif method === :max
        return normalize_max(X)
    elseif method === :minmax
        Xn, _, _ = normalize_minmax(X)
        return Xn
    elseif method === :zscore || method === :standard
        Xf = Matrix{Float64}(X)
        μ = mean(Xf, dims = 1)
        σ = std(Xf, dims = 1)
        σ[σ .== 0.0] .= 1.0
        return (Xf .- μ) ./ σ
    elseif method === :none || method === :raw
        return Matrix{Float64}(X)
    else
        throw(ArgumentError("Unknown normalization method: $method. Supported: :max, :minmax, :zscore, :none, Vector of scales, or a Function."))
    end
end

function swiss_roll(n; noise = 0.0)
    t = (3π / 2) .* (1 .+ 2 .* rand(n))
    h = 10 .* rand(n)
    x = t .* cos.(t)
    y = h
    z = t .* sin.(t)
    X = hcat(x, y, z)
    if noise > 0
        X .+= noise .* randn(size(X))
    end
    return X, t
end

@views function compute_lpca(
        dataset,
        k::Int,
        n_slices::Int;
        feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize::Union{Symbol, Function} = :standard
    )
    taus = sort(unique(dataset[:, 1]))
    idxs = round.(Int, range(1, length(taus), length = n_slices))
    selected_taus = taus[idxs]

    tol = 0.01
    n_selected = length(selected_taus)

    tau_values = zeros(Float64, n_selected)
    mean_dims = zeros(Float64, n_selected)
    std_dims = zeros(Float64, n_selected)

    for (idx, τ) in enumerate(selected_taus)
        _, X_tau = get_tau_slice(dataset, τ; feature_cols = feature_cols)
        X_norm = apply_normalization(X_tau, normalize)

        local_dims = dims(X_norm; k = k, tol = tol)

        tau_values[idx] = τ
        mean_dims[idx] = mean(local_dims)
        std_dims[idx] = std(local_dims)
    end

    return tau_values, mean_dims, std_dims
end


function to_2d_local_lpca(dataset::AbstractArray{<:Real, 3})
    dataset_2d = reshape(permutedims(dataset, (2, 1, 3)), :, size(dataset, 3))
    valid_rows = .!isnan.(dataset_2d[:, 1])
    return dataset_2d[valid_rows, :]
end

# ==============================================================================
# Coordinate Transformations & Scalings for Attractor & LPCA Studies
# ==============================================================================

"""
    transform_to_dimensionless(dataset; time_index = 1, temperature_index = 2)

Converts physical temperature T to dimensionless scaling variable w = tau * T.
Works on 2D matrices [rows, features] and 3D arrays [trajectories, time, features].
"""
function transform_to_dimensionless(
        dataset::AbstractMatrix{<:Real};
        time_index::Integer = 1,
        temperature_index::Integer = 2
    )
    transformed = copy(dataset)
    for row in axes(transformed, 1)
        proper_time = transformed[row, time_index]
        temperature = transformed[row, temperature_index]
        transformed[row, temperature_index] = proper_time * temperature
    end
    return transformed
end

function transform_to_dimensionless(
        dataset::AbstractArray{<:Real, 3};
        time_index::Integer = 1,
        temperature_index::Integer = 2
    )
    transformed = copy(dataset)
    for traj in axes(transformed, 1)
        for t in axes(transformed, 2)
            transformed[traj, t, temperature_index] = transformed[traj, t, time_index] * transformed[traj, t, temperature_index]
        end
    end
    return transformed
end

"""
    rescale_temperature(dataset, scale_factor; temperature_index = 2)

Rescales the temperature column by `scale_factor` (e.g. 10 * T) to test scale invariance.
Works on 2D matrices and 3D arrays.
"""
function rescale_temperature(
        dataset::AbstractMatrix{<:Real},
        scale_factor::Real;
        temperature_index::Integer = 2
    )
    transformed = copy(dataset)
    transformed[:, temperature_index] .*= scale_factor
    return transformed
end

function rescale_temperature(
        dataset::AbstractArray{<:Real, 3},
        scale_factor::Real;
        temperature_index::Integer = 2
    )
    transformed = copy(dataset)
    transformed[:, :, temperature_index] .*= scale_factor
    return transformed
end

"""
    create_mixed_scaled(dimensionless_data; w_scale = 10.0, a_scale = 2.0)

Creates mixed rescaled coordinates (e.g. 10w, 2A, B) or (10w, 2A).
Works on 2D matrices and 3D arrays.
"""
function create_mixed_scaled(
        dimensionless_data::AbstractMatrix{<:Real};
        w_scale::Real = 10.0,
        a_scale::Real = 2.0
    )
    res = Matrix{Float64}(copy(dimensionless_data))
    res[:, 2] .*= w_scale
    res[:, 3] .*= a_scale
    return res
end

function create_mixed_scaled(
        dimensionless_data::AbstractArray{<:Real, 3};
        w_scale::Real = 10.0,
        a_scale::Real = 2.0
    )
    res = Array{Float64, 3}(copy(dimensionless_data))
    res[:, :, 2] .*= w_scale
    res[:, :, 3] .*= a_scale
    return res
end

"""
    prepare_dataset_variants(dataset, model_name::Symbol) -> NamedTuple

Generates coordinate variants for invariance studies:
- `:physical`: [tau, T, A, ...]
- `:dimensionless`: [tau, w = tau*T, A, ...]
- `:scaled_10x`: [tau, 10*T, A, ...]
- `:mixed_scaled`: [tau, 10*w, 2*A, ...]
"""
function prepare_dataset_variants(dataset::AbstractArray{<:Real}, model_name::Symbol)
    physical_data = copy(dataset)
    dimensionless_data = transform_to_dimensionless(physical_data)
    scaled_data = rescale_temperature(physical_data, 10)
    mixed_data = create_mixed_scaled(dimensionless_data; w_scale = 10.0, a_scale = 2.0)

    return (
        model = model_name,
        physical = physical_data,
        dimensionless = dimensionless_data,
        scaled_10x = scaled_data,
        mixed_scaled = mixed_data,
    )
end

load_hydro_dataset(path::AbstractString) = load_dataset(path)

# ==============================================================================
# LPCA Systematic Evaluation: K-Dependency, Normalizations, Invariance & Distributions
# ==============================================================================

"""
    evaluate_k_pair(dataset, k_base, k_expanded, tau_values; feature_indices, normalize_method, tolerance)
"""
function evaluate_k_pair(
        dataset::AbstractMatrix{<:Real},
        k_base::Integer,
        k_expanded::Integer,
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize_method::Symbol = :max,
        tolerance::Real = 0.01
    )
    slice_count = length(tau_values)
    mean_dimension_k1 = zeros(Float64, slice_count)
    std_dimension_k1 = zeros(Float64, slice_count)
    mean_dimension_k2 = zeros(Float64, slice_count)
    std_dimension_k2 = zeros(Float64, slice_count)
    relative_difference = zeros(Float64, slice_count)

    for (index, tau) in enumerate(tau_values)
        _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
        normalized_slice = apply_normalization(raw_slice, normalize_method)

        dimension_samples_k1 = dims(normalized_slice; k = k_base, tol = tolerance)
        dimension_samples_k2 = dims(normalized_slice; k = k_expanded, tol = tolerance)

        mean_k1 = mean(dimension_samples_k1)
        mean_k2 = mean(dimension_samples_k2)

        mean_dimension_k1[index] = mean_k1
        std_dimension_k1[index] = std(dimension_samples_k1)
        mean_dimension_k2[index] = mean_k2
        std_dimension_k2[index] = std(dimension_samples_k2)

        if mean_k1 > 0
            relative_difference[index] = ((mean_k2 - mean_k1) / mean_k1) * 100
        else
            relative_difference[index] = 0.0
        end
    end

    return (
        k_base = k_base,
        k_expanded = k_expanded,
        tau_values = copy(tau_values),
        mean_dimension_k1 = mean_dimension_k1,
        std_dimension_k1 = std_dimension_k1,
        mean_dimension_k2 = mean_dimension_k2,
        std_dimension_k2 = std_dimension_k2,
        relative_difference = relative_difference,
    )
end
evaluate_k_pair(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = evaluate_k_pair(to_2d_local_lpca(dataset), args...; kwargs...)

"""
    evaluate_k_dependency(dataset, k_pairs, tau_values; feature_indices, normalize_method, tolerance)
"""
function evaluate_k_dependency(
        dataset::AbstractMatrix{<:Real},
        k_pairs::AbstractVector{<:Tuple{<:Integer, <:Integer}},
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize_method::Symbol = :max,
        tolerance::Real = 0.01
    )
    results = NamedTuple[]
    for (k_base, k_expanded) in k_pairs
        pair_result = evaluate_k_pair(
            dataset,
            k_base,
            k_expanded,
            tau_values;
            feature_indices = feature_indices,
            normalize_method = normalize_method,
            tolerance = tolerance
        )
        push!(results, pair_result)
    end
    return results
end
evaluate_k_dependency(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = evaluate_k_dependency(to_2d_local_lpca(dataset), args...; kwargs...)

"""
    evaluate_k_pair_pr(dataset, k_base, k_expanded, tau_values; feature_indices, normalize_method, use_density_weights)
"""
function evaluate_k_pair_pr(
        dataset::AbstractMatrix{<:Real},
        k_base::Integer,
        k_expanded::Integer,
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize_method::Symbol = :max,
        use_density_weights::Bool = false
    )
    slice_count = length(tau_values)
    mean_dimension_k1 = zeros(Float64, slice_count)
    std_dimension_k1 = zeros(Float64, slice_count)
    mean_dimension_k2 = zeros(Float64, slice_count)
    std_dimension_k2 = zeros(Float64, slice_count)
    relative_difference = zeros(Float64, slice_count)

    for (index, tau) in enumerate(tau_values)
        _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
        normalized_slice = apply_normalization(raw_slice, normalize_method)

        res_k1 = compute_local_pr_dimension(
            normalized_slice;
            k = k_base,
            use_density_weights = use_density_weights
        )
        res_k2 = compute_local_pr_dimension(
            normalized_slice;
            k = k_expanded,
            use_density_weights = use_density_weights
        )

        mean_k1 = res_k1.mean
        mean_k2 = res_k2.mean

        mean_dimension_k1[index] = mean_k1
        std_dimension_k1[index] = res_k1.std
        mean_dimension_k2[index] = mean_k2
        std_dimension_k2[index] = res_k2.std

        if mean_k1 > 0
            relative_difference[index] = ((mean_k2 - mean_k1) / mean_k1) * 100
        else
            relative_difference[index] = 0.0
        end
    end

    return (
        k_base = k_base,
        k_expanded = k_expanded,
        tau_values = copy(tau_values),
        mean_dimension_k1 = mean_dimension_k1,
        std_dimension_k1 = std_dimension_k1,
        mean_dimension_k2 = mean_dimension_k2,
        std_dimension_k2 = std_dimension_k2,
        relative_difference = relative_difference,
    )
end
evaluate_k_pair_pr(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = evaluate_k_pair_pr(to_2d_local_lpca(dataset), args...; kwargs...)

"""
    evaluate_k_dependency_pr(dataset, k_pairs, tau_values; feature_indices, normalize_method, use_density_weights)
"""
function evaluate_k_dependency_pr(
        dataset::AbstractMatrix{<:Real},
        k_pairs::AbstractVector{<:Tuple{<:Integer, <:Integer}},
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize_method::Symbol = :max,
        use_density_weights::Bool = false
    )
    results = NamedTuple[]
    for (k_base, k_expanded) in k_pairs
        pair_result = evaluate_k_pair_pr(
            dataset,
            k_base,
            k_expanded,
            tau_values;
            feature_indices = feature_indices,
            normalize_method = normalize_method,
            use_density_weights = use_density_weights
        )
        push!(results, pair_result)
    end
    return results
end
evaluate_k_dependency_pr(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = evaluate_k_dependency_pr(to_2d_local_lpca(dataset), args...; kwargs...)

"""
    compare_normalization_methods_pr(dataset, normalization_methods, k_pairs, tau_values; feature_indices, use_density_weights)
"""
function compare_normalization_methods_pr(
        dataset::AbstractMatrix{<:Real},
        normalization_methods::AbstractVector{Symbol},
        k_pairs::AbstractVector{<:Tuple{<:Integer, <:Integer}},
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        use_density_weights::Bool = false
    )
    method_results = Dict{Symbol, Vector{NamedTuple}}()
    for method in normalization_methods
        method_results[method] = evaluate_k_dependency_pr(
            dataset,
            k_pairs,
            tau_values;
            feature_indices = feature_indices,
            normalize_method = method,
            use_density_weights = use_density_weights
        )
    end
    return method_results
end
compare_normalization_methods_pr(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = compare_normalization_methods_pr(to_2d_local_lpca(dataset), args...; kwargs...)


"""
    compare_normalization_methods(dataset, normalization_methods, k_pairs, tau_values; feature_indices, tolerance)
"""
function compare_normalization_methods(
        dataset::AbstractMatrix{<:Real},
        normalization_methods::AbstractVector{Symbol},
        k_pairs::AbstractVector{<:Tuple{<:Integer, <:Integer}},
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        tolerance::Real = 0.01
    )
    method_results = Dict{Symbol, Vector{NamedTuple}}()
    for method in normalization_methods
        method_results[method] = evaluate_k_dependency(
            dataset,
            k_pairs,
            tau_values;
            feature_indices = feature_indices,
            normalize_method = method,
            tolerance = tolerance
        )
    end
    return method_results
end
compare_normalization_methods(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = compare_normalization_methods(to_2d_local_lpca(dataset), args...; kwargs...)

"""
    test_coordinate_invariance(variant_datasets, k_neighbor, tau_values; normalize_method, tolerance)
"""
function test_coordinate_invariance(
        variant_datasets::NamedTuple,
        k_neighbor::Integer,
        tau_values::AbstractVector{<:Real};
        normalize_method::Symbol = :max,
        tolerance::Real = 0.01
    )
    variant_keys = [:physical, :dimensionless, :scaled_10x]
    dimension_curves = Dict{Symbol, Vector{Float64}}()

    for key in variant_keys
        raw_data = getfield(variant_datasets, key)
        dataset = raw_data isa AbstractArray{<:Real, 3} ? to_2d_local_lpca(raw_data) : raw_data
        feature_cols = collect(2:size(dataset, 2))
        means = Float64[]

        for tau in tau_values
            _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_cols)
            normalized_slice = apply_normalization(raw_slice, normalize_method)
            dims_i = dims(normalized_slice; k = k_neighbor, tol = tolerance)
            push!(means, mean(dims_i))
        end
        dimension_curves[key] = means
    end

    max_difference_scaled = maximum(abs.(dimension_curves[:scaled_10x] .- dimension_curves[:physical]))
    max_difference_dimensionless = maximum(abs.(dimension_curves[:dimensionless] .- dimension_curves[:physical]))

    return (
        model = variant_datasets.model,
        normalize_method = normalize_method,
        k_neighbor = k_neighbor,
        tau_values = copy(tau_values),
        curves = dimension_curves,
        max_difference_scaled = max_difference_scaled,
        max_difference_dimensionless = max_difference_dimensionless,
    )
end

"""
    analyze_dimension_distribution(dataset, k_neighbor, tau_values; feature_indices, normalize_method, tolerance)
"""
function analyze_dimension_distribution(
        dataset::AbstractMatrix{<:Real},
        k_neighbor::Integer,
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize_method::Symbol = :max,
        tolerance::Real = 0.01
    )
    embedding_dimension = length(feature_indices)
    slice_count = length(tau_values)

    mean_dimension = zeros(Float64, slice_count)
    std_dimension = zeros(Float64, slice_count)
    median_dimension = zeros(Float64, slice_count)
    q25_dimension = zeros(Float64, slice_count)
    q75_dimension = zeros(Float64, slice_count)

    dimension_fractions = zeros(Float64, slice_count, embedding_dimension)

    for (index, tau) in enumerate(tau_values)
        _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
        normalized_slice = apply_normalization(raw_slice, normalize_method)
        point_dimensions = dims(normalized_slice; k = k_neighbor, tol = tolerance)

        sample_count = length(point_dimensions)
        mean_dimension[index] = mean(point_dimensions)
        std_dimension[index] = std(point_dimensions)
        median_dimension[index] = median(point_dimensions)
        q25_dimension[index] = quantile(point_dimensions, 0.25)
        q75_dimension[index] = quantile(point_dimensions, 0.75)

        for dim_val in 1:embedding_dimension
            dimension_fractions[index, dim_val] = count(==(dim_val), point_dimensions) / sample_count
        end
    end

    return (
        k_neighbor = k_neighbor,
        embedding_dimension = embedding_dimension,
        tau_values = copy(tau_values),
        mean_dimension = mean_dimension,
        std_dimension = std_dimension,
        median_dimension = median_dimension,
        q25_dimension = q25_dimension,
        q75_dimension = q75_dimension,
        dimension_fractions = dimension_fractions,
    )
end
analyze_dimension_distribution(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = analyze_dimension_distribution(to_2d_local_lpca(dataset), args...; kwargs...)

"""
    sweep_k_dimensions_over_tau(dataset, k_range, tau_values; feature_indices, normalize_method, tolerance)

Computes the dependency of both discrete dimension dims() and continuous Participation Ratio pr()
as a function of neighborhood size K in k_range, across multiple proper time slices tau_values.
Returns a NamedTuple containing Dict{Float64, NamedTuple} mapping each tau to:
(k_values, mean_dims, std_dims, mean_pr, std_pr).
"""
function sweep_k_dimensions_over_tau(
        dataset::AbstractMatrix{<:Real},
        k_range::AbstractVector{<:Integer},
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize_method::Symbol = :zscore,
        tolerance::Real = 0.01,
        max_points::Union{Nothing, Integer} = 10_000,
        seed::Integer = 42
    )
    max_k = maximum(k_range)
    D = length(feature_indices)
    k_vec = collect(k_range)
    n_k = length(k_vec)

    results = Dict{Float64, NamedTuple}()

    # Preallocate buffers for speed
    pts_buf = zeros(Float64, max_k + 1, D)
    cov_buf = zeros(Float64, D, D)

    for (t_idx, tau) in enumerate(tau_values)
        _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
        if !isnothing(max_points) && size(raw_slice, 1) > max_points
            rng = MersenneTwister(seed + t_idx)
            rows = randperm(rng, size(raw_slice, 1))[1:max_points]
            raw_slice = raw_slice[rows, :]
        end
        normalized_slice = apply_normalization(raw_slice, normalize_method)
        N = size(normalized_slice, 1)

        tree = KDTree(transpose(normalized_slice))
        all_idxs, _ = knn(tree, transpose(normalized_slice), max_k + 1, true)

        mean_dims = zeros(Float64, n_k)
        std_dims = zeros(Float64, n_k)
        mean_pr = zeros(Float64, n_k)
        std_pr = zeros(Float64, n_k)

        d_vals = zeros(Float64, N)
        pr_vals = zeros(Float64, N)

        for (k_idx, k) in enumerate(k_vec)
            for i in 1:N
                nbrs = @view all_idxs[i][1:(k + 1)]
                for c in 1:D
                    m_c = 0.0
                    for r in 1:(k + 1)
                        pts_buf[r, c] = normalized_slice[nbrs[r], c]
                        m_c += pts_buf[r, c]
                    end
                    m_c /= (k + 1)
                    for r in 1:(k + 1)
                        pts_buf[r, c] -= m_c
                    end
                end

                local_pts = @view pts_buf[1:(k + 1), :]
                mul!(cov_buf, transpose(local_pts), local_pts)
                cov_buf ./= k

                evals = eigvals!(Symmetric(cov_buf))
                for j in 1:D
                    evals[j] = max(evals[j], 0.0)
                end
                sum_e = sum(evals)
                sum_e2 = sum(evals .^ 2)

                # Discrete dims
                if sum_e > 0.0
                    cnt = 0
                    for j in 1:D
                        if (evals[j] / sum_e) > tolerance
                            cnt += 1
                        end
                    end
                    d_vals[i] = cnt
                else
                    d_vals[i] = 1.0
                end

                # Continuous PR
                if sum_e2 > 0.0
                    pr_vals[i] = (sum_e^2) / sum_e2
                else
                    pr_vals[i] = 1.0
                end
            end

            mean_dims[k_idx] = mean(d_vals)
            std_dims[k_idx] = std(d_vals)
            mean_pr[k_idx] = mean(pr_vals)
            std_pr[k_idx] = std(pr_vals)
        end

        results[Float64(tau)] = (
            k_values = copy(k_vec),
            mean_dims = mean_dims,
            std_dims = std_dims,
            mean_pr = mean_pr,
            std_pr = std_pr,
        )
    end

    return (
        k_values = copy(k_vec),
        tau_values = copy(tau_values),
        results = results,
        normalize_method = normalize_method,
        tolerance = tolerance,
    )
end
sweep_k_dimensions_over_tau(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) =
    sweep_k_dimensions_over_tau(to_2d_local_lpca(dataset), args...; kwargs...)

"""
    analyze_tolerance_sensitivity(dataset, tolerances, k_neighbor, tau_values; feature_indices, normalize_method)
"""
function analyze_tolerance_sensitivity(
        dataset::AbstractMatrix{<:Real},
        tolerances::AbstractVector{<:Real},
        k_neighbor::Integer,
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize_method::Symbol = :max
    )
    results = Dict{Float64, Vector{Float64}}()
    for tol_val in tolerances
        means = Float64[]
        for tau in tau_values
            _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
            normalized_slice = apply_normalization(raw_slice, normalize_method)
            dims_i = dims(normalized_slice; k = k_neighbor, tol = tol_val)
            push!(means, mean(dims_i))
        end
        results[Float64(tol_val)] = means
    end

    return (
        k_neighbor = k_neighbor,
        tolerances = copy(tolerances),
        tau_values = copy(tau_values),
        results = results,
    )
end
analyze_tolerance_sensitivity(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = analyze_tolerance_sensitivity(to_2d_local_lpca(dataset), args...; kwargs...)

"""
    compute_parameterization_k_sweep(variant_datasets, k_values, tau_values; normalize_method, tolerance)
"""
function compute_parameterization_k_sweep(
        variant_datasets::NamedTuple,
        k_values::AbstractVector{<:Integer},
        tau_values::AbstractVector{<:Real};
        normalize_method::Symbol = :max,
        tolerance::Real = 0.01
    )
    variant_candidates = (:physical, :dimensionless, :scaled_10x, :mixed_scaled)
    variant_keys = Tuple(k for k in variant_candidates if hasfield(typeof(variant_datasets), k))

    curves = Dict{Symbol, Dict{Int, Vector{Float64}}}()
    envelopes = Dict{Symbol, NamedTuple{(:min, :max), Tuple{Vector{Float64}, Vector{Float64}}}}()

    for key in variant_keys
        raw_data = getfield(variant_datasets, key)
        dataset = raw_data isa AbstractArray{<:Real, 3} ? to_2d_local_lpca(raw_data) : raw_data
        feature_cols = collect(2:size(dataset, 2))
        k_dict = Dict{Int, Vector{Float64}}()

        for k in k_values
            k_dict[k] = zeros(Float64, length(tau_values))
        end

        for (t_idx, tau) in enumerate(tau_values)
            _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_cols)
            normalized_slice = apply_normalization(raw_slice, normalize_method)

            for k in k_values
                dims_i = dims(normalized_slice; k = k, tol = tolerance)
                k_dict[k][t_idx] = mean(dims_i)
            end
        end

        curves[key] = k_dict

        min_curve = zeros(Float64, length(tau_values))
        max_curve = zeros(Float64, length(tau_values))
        for t_idx in eachindex(tau_values)
            d_vals = [k_dict[k][t_idx] for k in k_values]
            min_curve[t_idx] = minimum(d_vals)
            max_curve[t_idx] = maximum(d_vals)
        end
        envelopes[key] = (min = min_curve, max = max_curve)
    end

    return (
        model = variant_datasets.model,
        normalize_method = normalize_method,
        k_values = copy(k_values),
        tau_values = copy(tau_values),
        curves = curves,
        envelopes = envelopes,
    )
end

"""
    PointwiseDimensionSlice

Stores coordinates and corresponding local dimensions at a specific proper time.
"""
struct PointwiseDimensionSlice{T <: Real}
    tau::T
    coordinates::Matrix{T}
    dimensions::Vector{T}
    feature_names::Vector{Symbol}
end

"""
    compute_pointwise_dimensions(dataset, tau; feature_indices, feature_names, k_neighbor, tolerance, normalize_method, return_normalized)
"""
function compute_pointwise_dimensions(
        dataset::AbstractMatrix{<:Real},
        tau::Real;
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        feature_names::AbstractVector{Symbol} = Symbol[],
        k_neighbor::Integer = 24,
        tolerance::Real = 0.01,
        normalize_method::Symbol = :zscore,
        return_normalized::Bool = false,
        max_points::Union{Nothing, Integer} = nothing,
        seed::Integer = 42
    )
    tau_col = dataset[:, 1]
    rows = findall(isapprox.(tau_col, tau; atol = 1.0e-5))
    if isempty(rows)
        nearest_idx = argmin(abs.(tau_col .- tau))
        nearest_tau = tau_col[nearest_idx]
        rows = findall(isapprox.(tau_col, nearest_tau; atol = 1.0e-5))
    end

    if !isnothing(max_points) && length(rows) > max_points
        rng = MersenneTwister(seed)
        rows = rows[randperm(rng, length(rows))[1:max_points]]
    end

    raw_features = Matrix{Float64}(dataset[rows, feature_indices])
    normalized_features = apply_normalization(raw_features, normalize_method)
    local_dims = dims(normalized_features; k = k_neighbor, tol = tolerance)

    coords_to_return = return_normalized ? normalized_features : raw_features

    return PointwiseDimensionSlice(
        Float64(tau),
        coords_to_return,
        local_dims,
        isempty(feature_names) ? [Symbol("x$i") for i in 1:length(feature_indices)] : feature_names
    )
end
compute_pointwise_dimensions(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = compute_pointwise_dimensions(to_2d_local_lpca(dataset), args...; kwargs...)

"""
    compute_pointwise_pr(dataset, tau; feature_indices, feature_names, k, normalize_method, return_normalized, use_density_weights, max_points, seed)
"""
function compute_pointwise_pr(
        dataset::AbstractMatrix{<:Real},
        tau::Real;
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        feature_names::AbstractVector{Symbol} = Symbol[],
        k::Integer = 24,
        normalize_method::Symbol = :zscore,
        return_normalized::Bool = false,
        use_density_weights::Bool = true,
        max_points::Union{Nothing, Integer} = nothing,
        seed::Integer = 42
    )
    tau_col = dataset[:, 1]
    rows = findall(isapprox.(tau_col, tau; atol = 1.0e-5))
    if isempty(rows)
        nearest_idx = argmin(abs.(tau_col .- tau))
        nearest_tau = tau_col[nearest_idx]
        rows = findall(isapprox.(tau_col, nearest_tau; atol = 1.0e-5))
    end

    if !isnothing(max_points) && length(rows) > max_points
        rng = MersenneTwister(seed)
        rows = rows[randperm(rng, length(rows))[1:max_points]]
    end

    raw_features = Matrix{Float64}(dataset[rows, feature_indices])
    normalized_features = apply_normalization(raw_features, normalize_method)
    res = compute_local_pr_dimension(normalized_features; k = k, use_density_weights = use_density_weights)

    coords_to_return = return_normalized ? normalized_features : raw_features

    return (
        tau = Float64(tau),
        coordinates = coords_to_return,
        pr_values = res.d_pr,
        mean_pr = res.mean,
        std_pr = res.std,
        feature_names = isempty(feature_names) ? [Symbol("x$i") for i in 1:length(feature_indices)] : feature_names,
    )
end
compute_pointwise_pr(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = compute_pointwise_pr(to_2d_local_lpca(dataset), args...; kwargs...)

# ==============================================================================
# Soft-Weighted Local PCA (Sigmoidal Spectral Cutoff & Sampling Reliability)
# ==============================================================================

"""
    compute_soft_weighted_dimension(
        points::AbstractMatrix{<:Real};
        k::Integer = 16,
        tol::Real = 0.02,
        delta::Real = 0.005,
        tau_val::Union{Real, Nothing} = nothing,
        temp_col_idx::Integer = 1,
        w_focus::Union{Real, Nothing} = nothing,
        sigma_w::Union{Real, Nothing} = nothing,
        use_density_weights::Bool = true
    ) -> NamedTuple

Calculates a continuous (soft) local dimension using a sigmoidal spectral cutoff function
and weighted statistics taking into account local sampling density and optional clock variable (w = tau * T).
"""
function compute_soft_weighted_dimension(
        points::AbstractMatrix{<:Real};
        k::Integer = 16,
        tol::Real = 0.02,
        delta::Real = 0.005,
        tau_val::Union{Real, Nothing} = nothing,
        temp_col_idx::Integer = 1,
        w_focus::Union{Real, Nothing} = nothing,
        sigma_w::Union{Real, Nothing} = nothing,
        use_density_weights::Bool = true
    )
    N, D = size(points)
    eff_k = min(k, N - 1)
    @assert eff_k >= 2 "At least 3 points are needed to compute local PCA dimension."

    kdtree = KDTree(points')
    idxs, dists = knn(kdtree, points', eff_k + 1, true)

    d_soft = zeros(Float64, N)
    weights = ones(Float64, N)

    # Scale from distance to k-th neighbor
    k_dists = [d[end] for d in dists]
    sigma_R = median(k_dists)
    if sigma_R <= 0.0
        sigma_R = 1.0
    end

    sigmoid(z) = 1.0 / (1.0 + exp(-clamp(z, -30.0, 30.0)))

    centered_pts = zeros(Float64, eff_k + 1, D)
    cov_buf = zeros(Float64, D, D)

    for i in 1:N
        neighbor_indices = idxs[i]
        local_pts = @view points[neighbor_indices, :]

        # Centering
        for c in 1:D
            mean_c = 0.0
            for r in 1:(eff_k + 1)
                mean_c += local_pts[r, c]
            end
            mean_c /= (eff_k + 1)
            for r in 1:(eff_k + 1)
                centered_pts[r, c] = local_pts[r, c] - mean_c
            end
        end

        mul!(cov_buf, centered_pts', centered_pts)
        cov_buf ./= eff_k

        evals = eigvals!(Symmetric(cov_buf))
        sort!(evals, rev = true)

        lambda_1 = max(evals[1], 1.0e-14)

        dim_val = 1.0
        for j in 2:D
            r_j = evals[j] / lambda_1
            dim_val += sigmoid((r_j - tol) / delta)
        end
        d_soft[i] = dim_val

        w_i = 1.0
        if use_density_weights
            w_dens = exp(-(k_dists[i]^2) / (2.0 * sigma_R^2))
            w_i *= w_dens
        end

        if w_focus !== nothing && tau_val !== nothing && temp_col_idx in 1:D
            T_i = points[i, temp_col_idx]
            w_pt = tau_val * T_i
            sw = sigma_w !== nothing ? sigma_w : 0.5 * w_focus
            w_clock = exp(-((w_pt - w_focus)^2) / (2.0 * sw^2))
            w_i *= w_clock
        end

        weights[i] = w_i
    end

    total_w = sum(weights)
    if total_w <= 0.0
        weights .= 1.0
        total_w = Float64(N)
    end

    mean_dim = sum(weights .* d_soft) / total_w
    var_dim = sum(weights .* ((d_soft .- mean_dim) .^ 2)) / total_w

    return (
        mean = mean_dim,
        std = sqrt(max(0.0, var_dim)),
        d_soft = d_soft,
        weights = weights,
    )
end

"""
    scan_soft_weighted_dimension(dataset, tau_values; feature_indices, normalize_method, k, tol, delta, kwargs...) -> NamedTuple
"""
function scan_soft_weighted_dimension(
        dataset::AbstractMatrix{<:Real},
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize_method::Symbol = :max,
        k::Integer = 16,
        tol::Real = 0.02,
        delta::Real = 0.005,
        temp_col_in_features::Integer = 1,
        w_focus::Union{Real, Nothing} = nothing,
        sigma_w::Union{Real, Nothing} = nothing,
        use_density_weights::Bool = true
    )
    n_slices = length(tau_values)
    mean_dims = zeros(Float64, n_slices)
    std_dims = zeros(Float64, n_slices)
    unweighted_hard_means = zeros(Float64, n_slices)

    for (idx, tau) in enumerate(tau_values)
        _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
        norm_slice = apply_normalization(raw_slice, normalize_method)

        res = compute_soft_weighted_dimension(
            norm_slice;
            k = k,
            tol = tol,
            delta = delta,
            tau_val = tau,
            temp_col_idx = temp_col_in_features,
            w_focus = w_focus,
            sigma_w = sigma_w,
            use_density_weights = use_density_weights
        )

        mean_dims[idx] = res.mean
        std_dims[idx] = res.std

        hard_dims = dims(norm_slice; k = k, tol = tol)
        unweighted_hard_means[idx] = mean(hard_dims)
    end

    return (
        tau_values = copy(tau_values),
        mean_dims = mean_dims,
        std_dims = std_dims,
        unweighted_hard_means = unweighted_hard_means,
        k = k,
        tol = tol,
        delta = delta,
    )
end
scan_soft_weighted_dimension(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) =
    scan_soft_weighted_dimension(to_2d_local_lpca(dataset), args...; kwargs...)

"""
    compute_local_pr_dimension(
        points::AbstractMatrix{<:Real};
        k::Integer = 16,
        use_density_weights::Bool = true
    ) -> NamedTuple

Calculates the local Participation Ratio (PR) dimension:
    d_PR(x_i) = (Tr C_i)^2 / Tr(C_i^2) = (sum_j lambda_j)^2 / sum_j lambda_j^2
where C_i is the local covariance matrix evaluated on the k nearest neighbors.
Does not require any spectral cutoff threshold (tol) or sigmoid parameter (delta).
"""
function compute_local_pr_dimension(
        points::AbstractMatrix{<:Real};
        k::Union{Symbol, Integer} = :auto,
        k_neighbor::Union{Nothing, Symbol, Integer} = nothing,
        use_density_weights::Bool = true
    )
    N, D = size(points)
    raw_k = !isnothing(k_neighbor) ? k_neighbor : k
    eff_k = if raw_k === :auto
        suggest_optimal_k(N, D)
    else
        clamp(Int(raw_k), 2, max(2, N - 1))
    end
    @assert eff_k >= 2 "At least 3 points are needed to compute local PR dimension."

    kdtree = KDTree(points')
    idxs, dists = knn(kdtree, points', eff_k + 1, true)

    d_pr = zeros(Float64, N)
    weights = ones(Float64, N)

    k_dists = [d[end] for d in dists]
    sigma_R = median(k_dists)
    if sigma_R <= 0.0
        sigma_R = 1.0
    end

    centered_pts = zeros(Float64, eff_k + 1, D)
    cov_buf = zeros(Float64, D, D)

    for i in 1:N
        neighbor_indices = idxs[i]
        local_pts = @view points[neighbor_indices, :]

        for c in 1:D
            mean_c = 0.0
            for r in 1:(eff_k + 1)
                mean_c += local_pts[r, c]
            end
            mean_c /= (eff_k + 1)
            for r in 1:(eff_k + 1)
                centered_pts[r, c] = local_pts[r, c] - mean_c
            end
        end

        mul!(cov_buf, centered_pts', centered_pts)
        cov_buf ./= eff_k

        evals = eigvals!(Symmetric(cov_buf))
        for j in 1:D
            evals[j] = max(evals[j], 0.0)
        end

        sum_evals = sum(evals)
        sum_evals_sq = sum(evals .^ 2)

        if sum_evals_sq > 0.0
            d_pr[i] = (sum_evals^2) / sum_evals_sq
        else
            d_pr[i] = 1.0
        end

        if use_density_weights
            weights[i] = exp(-(k_dists[i]^2) / (2.0 * sigma_R^2))
        else
            weights[i] = 1.0
        end
    end

    total_w = sum(weights)
    if total_w <= 0.0
        weights .= 1.0
        total_w = Float64(N)
    end

    mean_dim = sum(weights .* d_pr) / total_w
    var_dim = sum(weights .* ((d_pr .- mean_dim) .^ 2)) / total_w

    return (
        mean = mean_dim,
        std = sqrt(max(0.0, var_dim)),
        d_pr = d_pr,
        weights = weights,
    )
end

"""
    scan_local_pr_dimension(dataset, tau_values; feature_indices, normalize_method, k, use_density_weights) -> NamedTuple

Scans local Participation Ratio (PR) dimension across proper time tau slices.
"""
function scan_local_pr_dimension(
        dataset::AbstractMatrix{<:Real},
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize_method::Symbol = :max,
        k::Integer = 16,
        use_density_weights::Bool = true
    )
    n_slices = length(tau_values)
    mean_dims = zeros(Float64, n_slices)
    std_dims = zeros(Float64, n_slices)
    unweighted_means = zeros(Float64, n_slices)

    for (idx, tau) in enumerate(tau_values)
        _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
        norm_slice = apply_normalization(raw_slice, normalize_method)

        res = compute_local_pr_dimension(
            norm_slice;
            k = k,
            use_density_weights = use_density_weights
        )

        mean_dims[idx] = res.mean
        std_dims[idx] = res.std
        unweighted_means[idx] = mean(res.d_pr)
    end

    return (
        tau_values = copy(tau_values),
        mean_dims = mean_dims,
        std_dims = std_dims,
        unweighted_means = unweighted_means,
        k = k,
    )
end
scan_local_pr_dimension(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) =
    scan_local_pr_dimension(to_2d_local_lpca(dataset), args...; kwargs...)

# ==============================================================================
# Optimal Neighborhood Size Heuristics (K Selection for Local PCA & PR Dimension)
# ==============================================================================

"""
    suggest_optimal_k(N::Integer, D::Integer; scaling_factor::Real = 1.0) -> Int
    suggest_optimal_k(points::AbstractMatrix{<:Real}; scaling_factor::Real = 1.0) -> Int

Heurystyka doboru optymalnej liczby najbliższych sąsiadów \$k\$ dla lokalnego PCA / wymiaru partycypacji \$d_{\\mathrm{PR}}\$.

### Uzasadnienie teoretyczne i geometryczne:
1. **Dolne ograniczenie (Random Matrix Theory / Marchenko-Pastur):**
   Aby fluktuacje próbkowania macierzy kowariancji nie powodowały pozornego zawyżenia małych wartości własnych
   (tzw. eigenvalue repulsion, która sztucznie zawyża \$d_{\\mathrm{PR}}\$ w stronę wymiaru tła \$D\$),
   liczba sąsiadów musi spełniać \$k \\ge 12 \\cdot D\$.
2. **Skalowanie z liczbą punktów (Manifold Density Scaling):**
   W estymacji geometrii rozmaitości o wymiarze wewnętrznym \$d\$ zanurzonej w \$D\$ wymiarach,
   optymalny kompromis MSE między obciążeniem krzywizną (\$\\mathrm{bias} \\propto R^2\$) a wariancją
   szumu próbkowania (\$\\mathrm{var} \\propto 1/k\$) prowadzi do relacji \$k \\propto N^{4/(d+4)}\$.
   Dla \$d \\in [1, 2]\$ daje to skalowanie \$k \\approx N^{0.45} \\cdot \\sqrt{D}\$.
3. **Górne ograniczenie (Curvature & Localization):**
   Dla dużych zbiorów \$k\$ jest ograniczane, aby promień kuli sąsiedztwa nie obejmował globalnych
   zagięć ani odgałęzień rozmaitości.

### Przykłady:
- \$N = 2\\,000, D = 2 \\implies k \\approx 43\$
- \$N = 2\\,000, D = 3 \\implies k \\approx 53\$
- \$N = 10\\,000, D = 2 \\implies k \\approx 89\$
- \$N = 10\\,000, D = 3 \\implies k \\approx 109\$
- \$N = 20\\,000, D = 2 \\implies k \\approx 122\$
"""
function suggest_optimal_k(N::Integer, D::Integer; scaling_factor::Real = 1.0)
    @assert N >= 3 "Co najmniej 3 punkty są wymagane do estymacji sąsiedztwa (podano N=$N)."
    min_k = min(12 * D, max(2, div(N, 2)))
    scaled_k = round(Int, scaling_factor * (Float64(N)^0.45) * sqrt(Float64(D)))
    max_k = max(min_k, min(div(N, 10), N - 1))
    return clamp(scaled_k, min_k, max(min_k, max_k))
end

suggest_optimal_k(points::AbstractMatrix{<:Real}; scaling_factor::Real = 1.0) =
    suggest_optimal_k(size(points, 1), size(points, 2); scaling_factor = scaling_factor)

"""
    find_optimal_k_spectral_gap(
        points::AbstractMatrix{<:Real};
        k_candidates::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        n_eval::Integer = 2000,
        normalize::Bool = true,
        gap_threshold::Real = 5.0
    ) -> NamedTuple

W pełni nienadzorowane (unsupervised) wyznaczanie optymalnej liczby najbliższych sąsiadów \$K^*\$
oraz wymiaru wewnętrznego \$d^*\$ w przestrzeni \$D\$-wymiarowej (np. \$D=3\$ dla modelu HJSW,
\$D=2\$ dla MIS) **bez wcześniejszej znajomości wymiaru ani wiedzy, czy wymiar ulega redukcji**.

### Zasada działania (Spectral Gap Maximization w D wymiarach):
1. **Lokalne widmo wartości własnych:**
   Dla każdego kandydata \$K \\in \\mathcal{K}\$ wyznaczane są lokalne wartości własne macierzy kowariancji:
   \$\\lambda_1(K) \\ge \\lambda_2(K) \\ge \\dots \\ge \\lambda_D(K) \\ge 0\$.
2. **Hierarchia przerw spektralnych:**
   W przestrzeni \$D\$-wymiarowej istnieje \$D - 1\$ potencjalnych rozszczepień spektralnych:
   \$g_j(K) = \\frac{\\langle \\lambda_j \\rangle}{\\max(\\langle \\lambda_{j+1} \\rangle, \\epsilon)}, \\quad j \\in \\{1, \\dots, D-1\\}\$.
   - \$j = 1\$ mierzy zapadnięcie do krzywej 1D (\$g_1 = \\lambda_1 / \\lambda_2\$),
   - \$j = 2\$ mierzy zapadnięcie do płata 2D (\$g_2 = \\lambda_2 / \\lambda_3\$), itd.
3. **Maksymalny rozstęp spektralny:**
   \$\\Delta_{\\max}(K) = \\max_{j \\in \\{1, \\dots, D-1\\}} g_j(K)\$.
4. **Warunek Random Matrix Theory (RMT / Marchenko-Pastur):**
   Aby zapobiec artefaktom pozornego rozpychania wartości własnych (eigenvalue repulsion) w małych próbach,
   poszukiwanie istotnych przerw spektralnych ograniczone jest do \$K \\ge 12D\$.
5. **Diagnoza wymiaru i redukcji:**
   - Jeśli \$\\max_{K \\ge 12D} \\Delta_{\\max}(K) \\ge \\text{gap\\_threshold}\$, układ wykazuje rzeczywiste
     zapadnięcie na podrozmaitość \$d^*\$-wymiarową (\$d^* < D\$). Wtedy:
     \$K^* = \\arg\\max_{K \\ge 12D} \\Delta_{\\max}(K)\$, a zdiagnozowany wymiar to \$d^* = \\arg\\max_j g_j(K^*)\$.
   - W przeciwnym razie układ zachowuje pełną objętość \$D\$-wymiarową (\$d^* = D\$). Optymalne \$K^*\$
     wybierane jest wówczas na podstawie minimalizacji czułości skali \$S(K) = |\\partial \\langle d_{\\mathrm{PR}} \\rangle / \\partial \\ln K| \\to \\min\$.

### Zwracane pola:
- `optimal_k::Int`: Optymalna liczba najbliższych sąsiadów \$K^*\$.
- `diagnosed_dimension::Int`: Zdiagnozowany wymiar całkowity \$d^* \\in \\{1, \\dots, D\\}\$.
- `effective_dimension::Float64`: Średni wymiar Participation Ratio \$\\langle d_{\\mathrm{PR}}(K^*) \\rangle\$.
- `has_submanifold_collapse::Bool`: Czy wykryto rzeczywistą redukcję wymiaru (\$d^* < D\$).
- `k_candidates::Vector{Int}`: Wektor przebadanych sąsiedztw \$K\$.
- `spectral_gaps::Matrix{Float64}`: Macierz luk spektralnych o rozmiarze `length(K) × (D-1)`.
- `max_gaps::Vector{Float64}`: Wektor \$\\Delta_{\\max}(K)\$ dla każdego \$K\$.
- `mean_eigenvalues::Matrix{Float64}`: Średnie wartości własne \$\\langle \\lambda_j \\rangle\$ (`length(K) × D`).
- `mean_dpr::Vector{Float64}`: Średni wymiar Participation Ratio dla każdego \$K\$.
- `std_dpr::Vector{Float64}`: Odchylenie standardowe \$\\sigma(d_{\\mathrm{PR}})\$ dla każdego \$K\$.
- `scale_sensitivity::Vector{Float64}`: Wskaźnik wrażliwości na skalę \$S(K) = |\\partial \\langle d_{\\mathrm{PR}} \\rangle / \\partial \\ln K|\$.
"""
function find_optimal_k_spectral_gap(
        points::AbstractMatrix{<:Real};
        k_candidates::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        n_eval::Integer = 2000,
        normalize::Bool = true,
        gap_threshold::Real = 5.0
    )
    N, D = size(points)
    @assert N >= 3 "Co najmniej 3 punkty są wymagane do estymacji sąsiedztwa (podano N=$N)."

    if D == 1
        k_opt = clamp(round(Int, N^0.45), 3, max(3, div(N, 10)))
        return (
            optimal_k = k_opt,
            diagnosed_dimension = 1,
            effective_dimension = 1.0,
            has_submanifold_collapse = false,
            k_candidates = [k_opt],
            spectral_gaps = zeros(Float64, 1, 1),
            max_gaps = [1.0],
            mean_eigenvalues = ones(Float64, 1, 1),
            mean_dpr = [1.0],
            std_dpr = [0.0],
            scale_sensitivity = [0.0],
        )
    end

    # 1. Normalizacja (Z-score standardization)
    X = if normalize
        Xf = Matrix{Float64}(points)
        μ = mean(Xf, dims = 1)
        σ = std(Xf, dims = 1)
        σ[σ .== 0.0] .= 1.0
        (Xf .- μ) ./ σ
    else
        Matrix{Float64}(points)
    end

    # 2. Siatka kandydatów K
    candidates = if k_candidates === nothing
        base_grid = [4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024]
        k_anchor = round(Int, (N^0.45) * sqrt(D))
        grid = sort(unique(vcat(base_grid, [k_anchor])))
        filter(k -> 2 <= k < N, grid)
    else
        sort(unique(filter(k -> 2 <= k < N, collect(k_candidates))))
    end

    @assert !isempty(candidates) "Brak poprawnych kandydatów k dla N=$N."

    # 3. Podpróbkowanie centrów ewaluacji dla dużych zbiorów
    eval_idxs = N <= n_eval ? collect(1:N) : round.(Int, range(1, N, length = n_eval))
    n_sample = length(eval_idxs)

    # 4. Jednokrotne zapytanie k-NN dla maksymalnego K
    max_k = maximum(candidates)
    tree = KDTree(transpose(X))
    knn_idxs_all, _ = knn(tree, transpose(X[eval_idxs, :]), max_k, true)

    n_k = length(candidates)
    mean_evals = zeros(Float64, n_k, D)
    mean_dpr = zeros(Float64, n_k)
    std_dpr = zeros(Float64, n_k)
    spectral_gaps = zeros(Float64, n_k, D - 1)

    local_evals = zeros(Float64, D)

    for (k_idx, k) in enumerate(candidates)
        eval_accum = zeros(Float64, D)
        dpr_vals = zeros(Float64, n_sample)

        for i in 1:n_sample
            idxs = knn_idxs_all[i][1:k]
            Y = X[idxs, :]
            Y_c = Y .- mean(Y, dims = 1)
            C = (transpose(Y_c) * Y_c) ./ (k - 1)

            ev = sort(eigvals(Symmetric(C)), rev = true)
            for j in 1:D
                val = max(ev[j], 1.0e-35)
                local_evals[j] = val
                eval_accum[j] += val
            end

            s_l = sum(local_evals)
            s_l2 = sum(local_evals .^ 2)
            dpr_vals[i] = s_l2 > 0.0 ? (s_l^2) / s_l2 : 1.0
        end

        eval_accum ./= n_sample
        mean_evals[k_idx, :] .= eval_accum
        mean_dpr[k_idx] = mean(dpr_vals)
        std_dpr[k_idx] = std(dpr_vals)

        # Luki spektralne: g_j = λ_j / λ_{j+1}
        for j in 1:(D - 1)
            spectral_gaps[k_idx, j] = eval_accum[j] / max(eval_accum[j + 1], 1.0e-35)
        end
    end

    # 5. Czułość skali S(K) = |d <d_PR> / d ln K|
    scale_sens = zeros(Float64, n_k)
    for i in 2:(n_k - 1)
        d_dpr = mean_dpr[i + 1] - mean_dpr[i - 1]
        d_lnk = log(candidates[i + 1]) - log(candidates[i - 1])
        scale_sens[i] = abs(d_dpr / d_lnk)
    end
    if n_k >= 2
        scale_sens[1] = abs((mean_dpr[2] - mean_dpr[1]) / (log(candidates[2]) - log(candidates[1])))
        scale_sens[end] = abs((mean_dpr[end] - mean_dpr[end - 1]) / (log(candidates[end]) - log(candidates[end - 1])))
    end

    # 6. Optymalizacja i diagnoza z uwzględnieniem progu RMT K >= 12D
    min_k_rmt = min(12 * D, div(N, 2))
    valid_idxs = findall(candidates .>= min_k_rmt)
    search_idxs = isempty(valid_idxs) ? collect(1:n_k) : valid_idxs

    # Skala plato o najmniejszej wrażliwości S(K)
    stab_idx = search_idxs[argmin(scale_sens[search_idxs])]
    base_eff_dim = mean_dpr[stab_idx]
    pr_dim = clamp(round(Int, base_eff_dim), 1, D)

    max_gaps = [maximum(spectral_gaps[k_idx, :]) for k_idx in 1:n_k]
    gap_dims = [argmax(spectral_gaps[k_idx, :]) for k_idx in 1:n_k]

    global_max_gap, rel_idx = findmax(max_gaps[search_idxs])
    best_k_idx = search_idxs[rel_idx]
    detected_gap_dim = gap_dims[best_k_idx]

    has_collapse = global_max_gap >= gap_threshold || pr_dim < D

    # Jeśli występuje redukcja wymiaru, wybieramy K* maksymalizujące separację modów
    optimal_k, diagnosed_dim = if pr_dim < D
        # Zapadnięcie na podrozmaitość pr_dim: maksymalizujemy lukę rozdzielającą podrozmaitość
        split_gaps = spectral_gaps[search_idxs, pr_dim]
        best_split_k = search_idxs[argmax(split_gaps)]
        (candidates[best_split_k], pr_dim)
    elseif has_collapse && detected_gap_dim < D
        (candidates[best_k_idx], detected_gap_dim)
    else
        # Pełna objętość D-wymiarowa (brak redukcji wymiaru)
        (candidates[stab_idx], D)
    end

    opt_k_idx = findfirst(==(optimal_k), candidates)
    eff_dim = mean_dpr[opt_k_idx]

    return (
        optimal_k = optimal_k,
        diagnosed_dimension = diagnosed_dim,
        effective_dimension = eff_dim,
        has_submanifold_collapse = has_collapse,
        k_candidates = candidates,
        spectral_gaps = spectral_gaps,
        max_gaps = max_gaps,
        mean_eigenvalues = mean_evals,
        mean_dpr = mean_dpr,
        std_dpr = std_dpr,
        scale_sensitivity = scale_sens,
    )
end

"""
    find_optimal_k(
        points::AbstractMatrix{<:Real};
        k_candidates::AbstractVector{<:Integer} = [16, 24, 32, 48, 64, 80, 100, 128, 160],
        criterion::Symbol = :bimodality_or_stability
    ) -> NamedTuple

Adaptacyjne (data-driven) wyznaczanie optymalnego \$k^*\$ na podstawie spektrum i rozkładu \$d_{\\mathrm{PR}}\$.
Obsługiwane kryteria:
- `:spectral_gap`: w pełni nienadzorowana maksymalizacja rozstępu spektralnego w \$D\$ wymiarach (nie wymaga znajomości wymiaru).
- `:bimodality`: maksymalizacja wskaźnika bimodalności Sarle'a (dla stadiów przejściowych relaksacji).
- `:variance`: maksymalizacja wariancji lokalnego wymiaru \$\\mathrm{Var}(d_{\\mathrm{PR}})\$.
- `:stability`: minimalizacja wrażliwości na skalę \$|\\Delta d_{\\mathrm{PR}} / \\Delta \\log k|\$ (plato skali).
- `:bimodality_or_stability` (domyślne): jeśli występuje silna dwumodalność (\$b > 0.6\$), wybiera maksimum
  bimodalności; w przeciwnym razie wybiera punkt największej stabilności spektralnej.
"""
function find_optimal_k(
        points::AbstractMatrix{<:Real};
        k_candidates::AbstractVector{<:Integer} = [16, 24, 32, 48, 64, 80, 100, 128, 160],
        criterion::Symbol = :bimodality_or_stability
    )
    if criterion === :spectral_gap
        return find_optimal_k_spectral_gap(points; k_candidates = k_candidates)
    end
    N, D = size(points)
    eff_candidates = filter(k -> 2 <= k < N, k_candidates)
    if isempty(eff_candidates)
        return (optimal_k = suggest_optimal_k(N, D), candidates = Int[], bimodality_scores = Float64[], variances = Float64[], means = Float64[])
    end

    b_scores = Float64[]
    vars = Float64[]
    means = Float64[]

    for k in eff_candidates
        res = compute_local_pr_dimension(points; k = k, use_density_weights = false)
        d = res.d_pr
        m = mean(d)
        s = std(d)
        v = s^2
        push!(means, m)
        push!(vars, v)
        if s > 1.0e-6
            m3 = mean((d .- m) .^ 3)
            m4 = mean((d .- m) .^ 4)
            skew = m3 / (s^3)
            kurt = m4 / (s^4)
            b = kurt > 1.0e-6 ? (skew^2 + 1.0) / kurt : 0.0
            push!(b_scores, b)
        else
            push!(b_scores, 0.0)
        end
    end

    best_idx = if criterion === :bimodality
        argmax(b_scores)
    elseif criterion === :variance
        argmax(vars)
    elseif criterion === :stability
        if length(eff_candidates) > 1
            stabs = [abs(means[i + 1] - means[i]) / (log(eff_candidates[i + 1]) - log(eff_candidates[i])) for i in 1:(length(eff_candidates) - 1)]
            push!(stabs, stabs[end])
            argmin(stabs)
        else
            1
        end
    else # :bimodality_or_stability
        max_b = isempty(b_scores) ? 0.0 : maximum(b_scores)
        if max_b > 0.6
            argmax(b_scores)
        elseif length(eff_candidates) > 1
            stabs = [abs(means[i + 1] - means[i]) / (log(eff_candidates[i + 1]) - log(eff_candidates[i])) for i in 1:(length(eff_candidates) - 1)]
            push!(stabs, stabs[end])
            argmin(stabs)
        else
            1
        end
    end

    optimal_k = eff_candidates[best_idx]
    return (
        optimal_k = optimal_k,
        candidates = eff_candidates,
        bimodality_scores = b_scores,
        variances = vars,
        means = means,
    )
end

# ==============================================================================
# Frozen Initial-Scale Normalization and LPCA Analysis
# ==============================================================================

"""
    compute_initial_scales(dataset::AbstractMatrix{<:Real}; tau_col::Integer = 1, feature_cols = 2:size(dataset, 2)) -> Vector{Float64}

Computes scaling factors x_j = std(X_j(t_0)) from the initial time slice t_0 = min(tau).
Dividing features by these scales ensures Var(X_j(t_0) / x_j) = 1.
"""
function compute_initial_scales(
        dataset::AbstractMatrix{<:Real};
        tau_col::Integer = 1,
        feature_cols = 2:size(dataset, 2)
    )
    taus = view(dataset, :, tau_col)
    t0 = minimum(taus)
    _, X0 = get_tau_slice(dataset, t0; feature_cols = feature_cols)
    scales = [std(X0[:, j]) for j in 1:size(X0, 2)]
    for j in eachindex(scales)
        if scales[j] == 0.0
            scales[j] = 1.0
        end
    end
    return scales
end

"""
    apply_frozen_scales(X::AbstractMatrix{<:Real}, scales::AbstractVector{<:Real}) -> Matrix{Float64}

Scales feature matrix X by dividing each column by fixed scales: X[:, j] ./ scales[j].
"""
function apply_frozen_scales(X::AbstractMatrix{<:Real}, scales::AbstractVector{<:Real})
    Z = Matrix{Float64}(undef, size(X))
    for j in 1:size(X, 2)
        Z[:, j] .= X[:, j] ./ scales[j]
    end
    return Z
end

"""
    lpca_frozen_slice(
        dataset::AbstractMatrix{<:Real},
        tau::Real,
        scales::AbstractVector{<:Real};
        k::Integer = 16,
        feature_cols = 2:size(dataset, 2)
    ) -> NamedTuple

Computes local PCA Participation Ratio dimension d_PR for a given tau slice
using frozen initial scales to define distances and local PCA coordinates.
"""
function lpca_frozen_slice(
        dataset::AbstractMatrix{<:Real},
        tau::Real,
        scales::AbstractVector{<:Real};
        k::Integer = 16,
        feature_cols = 2:size(dataset, 2)
    )
    _, Xt = get_tau_slice(dataset, tau; feature_cols = feature_cols)
    Z = apply_frozen_scales(Xt, scales)
    N, D = size(Z)
    eff_k = clamp(k, 2, N - 1)

    tree = KDTree(transpose(Z))
    knn_idxs, _ = knn(tree, transpose(Z), eff_k, true)

    d_pr = zeros(Float64, N)
    evals_mean = zeros(Float64, D)
    local_evals = zeros(Float64, D)

    for i in 1:N
        idxs = knn_idxs[i]
        Y = Z[idxs, :]
        Y .-= mean(Y, dims = 1)
        C = (transpose(Y) * Y) ./ (eff_k - 1)
        ev = sort(eigvals(Symmetric(C)), rev = true)
        for j in 1:D
            local_evals[j] = max(ev[j], 0.0)
            evals_mean[j] += local_evals[j]
        end
        s1 = sum(local_evals)
        s2 = sum(local_evals .^ 2)
        d_pr[i] = s2 > 0.0 ? (s1^2) / s2 : 1.0
    end
    evals_mean ./= N

    return (
        mean_dpr = mean(d_pr),
        std_dpr = std(d_pr),
        d_pr = d_pr,
        mean_evals = evals_mean,
        spectral_ratio = evals_mean[end] / max(evals_mean[1], 1e-35),
        k = eff_k,
        tau = tau
    )
end

"""
    k_sweep_frozen(
        dataset::AbstractMatrix{<:Real},
        tau::Real,
        scales::AbstractVector{<:Real};
        k_range = [4, 6, 8, 12, 16, 24, 32, 48, 64],
        feature_cols = 2:size(dataset, 2)
    ) -> NamedTuple

Performs a k-sweep for a specific tau slice using frozen initial scales,
returning the mean and std of d_PR across k values to detect the stability plateau.
"""
function k_sweep_frozen(
        dataset::AbstractMatrix{<:Real},
        tau::Real,
        scales::AbstractVector{<:Real};
        k_range = [4, 6, 8, 12, 16, 24, 32, 48, 64],
        feature_cols = 2:size(dataset, 2)
    )
    _, Xt = get_tau_slice(dataset, tau; feature_cols = feature_cols)
    Z = apply_frozen_scales(Xt, scales)
    N, D = size(Z)

    max_k = maximum(k_range)
    tree = KDTree(transpose(Z))
    knn_idxs_all, _ = knn(tree, transpose(Z), max_k, true)

    n_k = length(k_range)
    mean_dpr = zeros(Float64, n_k)
    std_dpr = zeros(Float64, n_k)
    mean_ratios = zeros(Float64, n_k)

    for (kid, k) in enumerate(k_range)
        eff_k = clamp(k, 2, N - 1)
        d_pr = zeros(Float64, N)
        ev_accum = zeros(Float64, D)

        for i in 1:N
            idxs = knn_idxs_all[i][1:eff_k]
            Y = Z[idxs, :]
            Y .-= mean(Y, dims = 1)
            C = (transpose(Y) * Y) ./ (eff_k - 1)
            ev = sort(eigvals(Symmetric(C)), rev = true)
            ev = max.(ev, 0.0)
            ev_accum .+= ev
            s1 = sum(ev)
            s2 = sum(ev .^ 2)
            d_pr[i] = s2 > 0.0 ? (s1^2) / s2 : 1.0
        end
        mean_dpr[kid] = mean(d_pr)
        std_dpr[kid] = std(d_pr)
        mean_ratios[kid] = ev_accum[end] / max(ev_accum[1], 1e-35)
    end

    return (
        k_values = collect(k_range),
        mean_dpr = mean_dpr,
        std_dpr = std_dpr,
        mean_ratios = mean_ratios,
        tau = tau
    )
end

"""
    scan_lpca_frozen(
        dataset::AbstractMatrix{<:Real},
        scales::AbstractVector{<:Real};
        tau_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
        k::Integer = 16,
        feature_cols = 2:size(dataset, 2)
    ) -> NamedTuple

Scans local PCA Participation Ratio dimension d_PR across tau slices
using frozen initial scales.
"""
function scan_lpca_frozen(
        dataset::AbstractMatrix{<:Real},
        scales::AbstractVector{<:Real};
        tau_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
        k::Integer = 16,
        feature_cols = 2:size(dataset, 2)
    )
    taus = tau_values === nothing ? sort(unique(dataset[:, 1])) : collect(tau_values)
    n_slices = length(taus)
    mean_dims = zeros(Float64, n_slices)
    std_dims = zeros(Float64, n_slices)
    spectral_ratios = zeros(Float64, n_slices)

    for (idx, τ) in enumerate(taus)
        res = lpca_frozen_slice(dataset, τ, scales; k = k, feature_cols = feature_cols)
        mean_dims[idx] = res.mean_dpr
        std_dims[idx] = res.std_dpr
        spectral_ratios[idx] = res.spectral_ratio
    end

    return (
        tau_values = taus,
        mean_dims = mean_dims,
        std_dims = std_dims,
        spectral_ratios = spectral_ratios,
        k = k
    )
end

"""
    plot_frozen_phase_space_slice(
        dataset::AbstractMatrix{<:Real},
        tau::Real,
        scales::AbstractVector{<:Real};
        k::Integer = 16,
        feature_indices = 2:size(dataset, 2),
        scaled_coords::Bool = false,
        x_label = nothing,
        y_label = nothing,
        z_label = nothing,
        colormap = lpca_style,
        figure_size = (800, 600),
        markersize = 6
    ) -> Figure

Plots a phase-space slice at proper time `tau` colored by the local Participation Ratio dimension `d_PR`
calculated using frozen initial scales.
"""
function plot_frozen_phase_space_slice(
        dataset::AbstractMatrix{<:Real},
        tau::Real,
        scales::AbstractVector{<:Real};
        k::Integer = 16,
        feature_indices = 2:size(dataset, 2),
        scaled_coords::Bool = false,
        x_label = nothing,
        y_label = nothing,
        z_label = nothing,
        colormap = lpca_style,
        figure_size = (800, 600),
        markersize = 6
    )
    CairoMakie.activate!()
    set_publication_theme(; cmap = lpca_style)
    res = lpca_frozen_slice(dataset, tau, scales; k = k, feature_cols = feature_indices)
    _, Xt = get_tau_slice(dataset, tau; feature_cols = feature_indices)
    pts = scaled_coords ? apply_frozen_scales(Xt, scales) : Xt
    D = size(pts, 2)

    lbl_x = x_label !== nothing ? x_label : (scaled_coords ? L"T / x_T" : L"T\ [\text{fm}^{-1}]")
    lbl_y = y_label !== nothing ? y_label : (scaled_coords ? L"\mathcal{A} / x_{\mathcal{A}}" : L"\mathcal{A}")
    lbl_z = z_label !== nothing ? z_label : (scaled_coords ? L"\mathcal{B} / x_{\mathcal{B}}" : L"\mathcal{B}")

    fig = Figure(size = figure_size)
    c_min, c_max = 1.0, Float64(D)

    if D == 2
        ax = Axis(fig[1, 1],
            title = L"\tau = %$(round(tau, digits=2))\,\text{fm}/c \quad (\langle d_{\text{PR}} \rangle = %$(round(res.mean_dpr, digits=3)))",
            xlabel = lbl_x,
            ylabel = lbl_y
        )
        sc = scatter!(ax, pts[:, 1], pts[:, 2]; color = res.d_pr, colormap = colormap, colorrange = (c_min, c_max), markersize = markersize, strokewidth = 0.2, strokecolor = (:black, 0.2))
        Colorbar(fig[1, 2], sc, label = L"d_{\text{PR}}", width = 16)
    elseif D >= 3
        ax = Axis3(fig[1, 1],
            title = L"\tau = %$(round(tau, digits=2))\,\text{fm}/c \quad (\langle d_{\text{PR}} \rangle = %$(round(res.mean_dpr, digits=3)))",
            xlabel = lbl_x,
            ylabel = lbl_y,
            zlabel = lbl_z
        )
        sc = scatter!(ax, pts[:, 1], pts[:, 2], pts[:, 3]; color = res.d_pr, colormap = colormap, colorrange = (c_min, c_max), markersize = markersize, strokewidth = 0.2, strokecolor = (:black, 0.2))
        Colorbar(fig[1, 2], sc, label = L"d_{\text{PR}}", width = 16)
    end
    return fig
end

"""
    plot_frozen_phase_space_grid(
        dataset::AbstractMatrix{<:Real},
        tau_grid::AbstractVector{<:Real},
        scales::AbstractVector{<:Real};
        k::Integer = 16,
        feature_indices = 2:size(dataset, 2),
        scaled_coords::Bool = false,
        colormap = lpca_style,
        figure_size = (1100, 850),
        markersize = 5
    ) -> Figure

Generates a multi-panel 2D grid of phase space slices across time tau,
showing the manifold collapse colored by local Participation Ratio dimension d_PR.
"""
function plot_frozen_phase_space_grid(
        dataset::AbstractMatrix{<:Real},
        tau_grid::AbstractVector{<:Real},
        scales::AbstractVector{<:Real};
        k::Integer = 16,
        feature_indices = 2:size(dataset, 2),
        scaled_coords::Bool = false,
        x_label = nothing,
        y_label = nothing,
        colormap = lpca_style,
        figure_size = (1100, 850),
        markersize = 5
    )
    CairoMakie.activate!()
    set_publication_theme(; cmap = lpca_style)
    n_panels = length(tau_grid)
    n_cols = n_panels <= 4 ? 2 : 3
    n_rows = ceil(Int, n_panels / n_cols)

    _, X_test = get_tau_slice(dataset, tau_grid[1]; feature_cols = feature_indices)
    D = size(X_test, 2)
    c_min, c_max = 1.0, Float64(D)

    lbl_x = x_label !== nothing ? x_label : (scaled_coords ? L"T / x_T" : L"T\ [\text{fm}^{-1}]")
    lbl_y = y_label !== nothing ? y_label : (scaled_coords ? L"\mathcal{A} / x_{\mathcal{A}}" : L"\mathcal{A}")

    fig = Figure(size = figure_size)
    local last_sc = nothing

    for (idx, τ) in enumerate(tau_grid)
        r = div(idx - 1, n_cols) + 1
        c = mod(idx - 1, n_cols) + 1

        res = lpca_frozen_slice(dataset, τ, scales; k = k, feature_cols = feature_indices)
        _, Xt = get_tau_slice(dataset, τ; feature_cols = feature_indices)
        pts = scaled_coords ? apply_frozen_scales(Xt, scales) : Xt

        ax = Axis(fig[r, c],
            title = L"\tau = %$(round(τ, digits=2))\,\text{fm}/c \quad (\langle d_{\text{PR}} \rangle = %$(round(res.mean_dpr, digits=2)))",
            xlabel = (r == n_rows ? lbl_x : ""),
            ylabel = (c == 1 ? lbl_y : "")
        )
        last_sc = scatter!(ax, pts[:, 1], pts[:, 2]; color = res.d_pr, colormap = colormap, colorrange = (c_min, c_max), markersize = markersize, strokewidth = 0.1, strokecolor = (:black, 0.15))
    end

    if last_sc !== nothing
        Colorbar(fig[:, n_cols + 1], last_sc, label = L"\text{Lokalny wymiar } d_{\text{PR}}", width = 18)
    end
    return fig
end

"""
    plot_frozen_k_sweep(
        dataset::AbstractMatrix{<:Real},
        tau_values::AbstractVector{<:Real},
        scales::AbstractVector{<:Real};
        k_range = [4, 6, 8, 12, 16, 20, 24, 32, 48, 64],
        feature_indices = 2:size(dataset, 2),
        figure_size = (900, 500)
    ) -> Figure

Plots d_PR vs k for selected tau slices to visualize the stability plateau.
"""
function plot_frozen_k_sweep(
        dataset::AbstractMatrix{<:Real},
        tau_values::AbstractVector{<:Real},
        scales::AbstractVector{<:Real};
        k_range = [4, 6, 8, 12, 16, 20, 24, 32, 48, 64],
        feature_indices = 2:size(dataset, 2),
        figure_size = (900, 500)
    )
    CairoMakie.activate!()
    set_publication_theme(; cmap = lpca_style)
    fig = Figure(size = figure_size)
    ax = Axis(fig[1, 1],
        title = L"\text{Zależność wymiaru } d_{\text{PR}} \text{ od } k \text{ (plateau stabilności)}",
        xlabel = L"\text{Liczba sąsiadów } k",
        ylabel = L"\langle d_{\text{PR}} \rangle",
        xscale = log10
    )

    colors = LPCA_COLORS
    for (i, τ) in enumerate(tau_values)
        sw = k_sweep_frozen(dataset, τ, scales; k_range = k_range, feature_cols = feature_indices)
        col = colors[mod1(i, length(colors))]
        lines!(ax, sw.k_values, sw.mean_dpr, label = L"\tau = %$(round(τ, digits=2))\,\text{fm}/c", color = col, linewidth = 2.5)
        scatter!(ax, sw.k_values, sw.mean_dpr, color = col, markersize = 8)
    end
    axislegend(ax, position = :rt)
    return fig
end

"""
    frozen(
        MODEL_NAME::Symbol = :mis;
        dataset_path::Union{Nothing, AbstractString} = nothing,
        K_NEIGHBOR::Integer = 16,
        K_SWEEP_RANGE = [4, 6, 8, 12, 16, 20, 24, 32, 48, 64],
        TAU_SLICES_PLOT = [0.20, 0.35, 0.55, 0.85, 1.50, 2.50],
        GENERATE_PLOTS::Bool = true,
        SAVE_FORMAT::Symbol = :png,
        OUTPUT_PLOT_DIR::String = "plots/frozen_lpca"
    ) -> NamedTuple

Kompleksowy workflow Local PCA z zamrożonymi skalami początkowymi:
1. Oblicza skale x_j = std(X_j(t0)) w chwili początkowej t0 = min(tau).
2. Skaluje cechy przez x_j dla wszystkich t >= t0.
3. Bada plateau w funkcji k (k-sweep).
4. Skanuje ewolucję wymiaru d_PR(tau).
5. Generuje wykresy przestrzeni fazowej w lpca_style i stylu publikacyjnym.
Zwraca krotkę z wynikami obliczeń i figurami Makie.
"""
function frozen(
        MODEL_NAME::Symbol = :mis;
        dataset_path::Union{Nothing, AbstractString} = nothing,
        K_NEIGHBOR::Integer = 16,
        K_SWEEP_RANGE = [4, 6, 8, 12, 16, 20, 24, 32, 48, 64],
        TAU_SLICES_PLOT = [0.20, 0.35, 0.55, 0.85, 1.50, 2.50],
        GENERATE_PLOTS::Bool = true,
        SAVE_FORMAT::Symbol = :png,
        OUTPUT_PLOT_DIR::String = "plots/frozen_lpca"
    )
    CairoMakie.activate!()
    set_publication_style(; cmap = lpca_style, fontsize = 18)

    fname = MODEL_NAME === :mis ? "mis_matched_seed5.h5" : "hjsw_matched_seed5.h5"
    DATASET_PATH = if dataset_path !== nothing
        dataset_path
    else
        possible_paths = [
            joinpath("datasets", "hjsw_lpca", fname),
            joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", fname),
            joinpath(@__DIR__, "datasets", "hjsw_lpca", fname)
        ]
        data_idx = findfirst(isfile, possible_paths)
        if isnothing(data_idx)
            error("Nie znaleziono pliku $fname w folderze datasets/hjsw_lpca/")
        end
        possible_paths[data_idx]
    end

    println("="^75)
    println(" ANALIZA LOCAL PCA Z ZAMROŻONYMI SKALAMI POCZĄTKOWYMI (FROZEN INITIAL SCALES)")
    println("="^75)
    println("Model:            ", MODEL_NAME)
    println("Ścieżka do danych: ", DATASET_PATH)

    println("\nWczytywanie zbioru danych...")
    dataset = load_dataset(DATASET_PATH)
    n_rows, n_cols = size(dataset)
    all_taus = sort(unique(dataset[:, 1]))
    t0 = all_taus[1]
    t_max = all_taus[end]
    feature_indices = collect(2:n_cols)
    D = length(feature_indices)

    println("Wczytano: $(n_rows) wierszy, $(n_cols) cech (D = $D), $(length(all_taus)) przekrojów tau od $(round(t0, digits=2)) do $(round(t_max, digits=2)) fm/c.")

    # KROK 1: Obliczenie zamrożonych skal
    println("\n" * "-"^75)
    println("KROK 1: Obliczanie skal początkowych x_j = std(X_j(t0)) w chwili t0 = $t0")
    println("-"^75)

    scales = compute_initial_scales(dataset; tau_col = 1, feature_cols = feature_indices)
    _, X0 = get_tau_slice(dataset, t0; feature_cols = feature_indices)
    Z0 = apply_frozen_scales(X0, scales)

    col_names = D == 2 ? ["T [fm^-1]", "A"] : ["T [MeV]", "A", "B"]
    for j in 1:D
        name = j <= length(col_names) ? col_names[j] : "X_$j"
        println("  Cecha $(rpad(name, 10)): średnia = $(round(mean(X0[:, j]), digits=4)),  skala x_$j = std = $(round(scales[j], digits=4))  ==>  Var po skalowaniu = $(round(var(Z0[:, j]), digits=6))")
    end

    # KROK 2: Analiza w t0 i t_max
    println("\n" * "-"^75)
    println("KROK 2: Weryfikacja wymiaru d_PR dla t0 = $t0 oraz chwili późnej t = $t_max (k = $K_NEIGHBOR)")
    println("-"^75)

    res_t0 = lpca_frozen_slice(dataset, t0, scales; k = K_NEIGHBOR, feature_cols = feature_indices)
    println("Chwila początkowa tau = $(round(t0, digits=2)) fm/c:")
    println("  Średni wymiar d_PR = $(round(res_t0.mean_dpr, digits=4)) ± $(round(res_t0.std_dpr, digits=4)) (teoria: D = $D)")
    println("  Stosunek spektralny lambda_min / lambda_max = $(res_t0.spectral_ratio)")

    res_late = lpca_frozen_slice(dataset, t_max, scales; k = K_NEIGHBOR, feature_cols = feature_indices)
    println("\nChwila asymptotyczna tau = $(round(t_max, digits=2)) fm/c:")
    println("  Średni wymiar d_PR = $(round(res_late.mean_dpr, digits=4)) ± $(round(res_late.std_dpr, digits=4)) (teoria: 1.0 dla atraktora)")
    println("  Stosunek spektralny lambda_min / lambda_max = $(res_late.spectral_ratio) (zanik poprzeczny)")

    # KROK 3: Badanie plateau w funkcji k
    println("\n" * "-"^75)
    println("KROK 3: Poszukiwanie plateau w funkcji liczby sąsiadów k (k-sweep)")
    println("-"^75)

    sweep_t0 = k_sweep_frozen(dataset, t0, scales; k_range = K_SWEEP_RANGE, feature_cols = feature_indices)
    sweep_late = k_sweep_frozen(dataset, t_max, scales; k_range = K_SWEEP_RANGE, feature_cols = feature_indices)

    println("   k  |  d_PR(t0 = $t0)  |  d_PR(t = $t_max)  | Stosunek spektralny (t_max)")
    println("------+-------------------+--------------------+----------------------------")
    for (idx, k_val) in enumerate(K_SWEEP_RANGE)
        println("  $(rpad(k_val, 3)) |  $(rpad(round(sweep_t0.mean_dpr[idx], digits=4), 6)) ± $(round(sweep_t0.std_dpr[idx], digits=4))  |  $(rpad(round(sweep_late.mean_dpr[idx], digits=4), 6)) ± $(round(sweep_late.std_dpr[idx], digits=4))  |  $(sweep_late.mean_ratios[idx])")
    end

    # KROK 4: Skan ewolucji w czasie
    println("\n" * "-"^75)
    println("KROK 4: Ewolucja wymiaru d_PR(tau) wzdłuż relaksacji hydrodynamicznej")
    println("-"^75)

    step_stride = max(1, div(length(all_taus), 15))
    selected_scan_taus = all_taus[1:step_stride:end]
    if selected_scan_taus[end] != all_taus[end]
        push!(selected_scan_taus, all_taus[end])
    end

    scan_res = scan_lpca_frozen(dataset, scales; tau_values = selected_scan_taus, k = K_NEIGHBOR, feature_cols = feature_indices)

    println("  tau [fm/c] |  Średni d_PR  |   std(d_PR)  | lambda_min / lambda_max")
    println("-------------+---------------+--------------+------------------------")
    for (i, τ) in enumerate(scan_res.tau_values)
        println("    $(rpad(round(τ, digits=2), 6))   |    $(rpad(round(scan_res.mean_dims[i], digits=4), 7))    |    $(rpad(round(scan_res.std_dims[i], digits=4), 7))   |       $(scan_res.spectral_ratios[i])")
    end

    # KROK 5: Wykresy
    figs = nothing
    if GENERATE_PLOTS
        println("\n" * "-"^75)
        println("KROK 5: Generowanie wykresów przestrzeni fazowej z motywem publikacyjnym i lpca_style...")
        println("-"^75)
        mkpath(OUTPUT_PLOT_DIR)
        ext = string(SAVE_FORMAT)

        valid_plot_taus = filter(τ -> any(isapprox.(all_taus, τ; atol = 0.05)), TAU_SLICES_PLOT)
        if isempty(valid_plot_taus)
            valid_plot_taus = all_taus[round.(Int, range(1, length(all_taus), length = 6))]
        end

        fig_grid_phys = plot_frozen_phase_space_grid(
            dataset, valid_plot_taus, scales;
            k = K_NEIGHBOR, feature_indices = feature_indices, scaled_coords = false,
            colormap = lpca_style
        )
        save(joinpath(OUTPUT_PLOT_DIR, "$(MODEL_NAME)_phase_space_grid_physical.$ext"), fig_grid_phys)

        fig_grid_scaled = plot_frozen_phase_space_grid(
            dataset, valid_plot_taus, scales;
            k = K_NEIGHBOR, feature_indices = feature_indices, scaled_coords = true,
            colormap = lpca_style
        )
        save(joinpath(OUTPUT_PLOT_DIR, "$(MODEL_NAME)_phase_space_grid_scaled.$ext"), fig_grid_scaled)

        taus_for_k_plot = [valid_plot_taus[1], valid_plot_taus[max(1, div(length(valid_plot_taus), 2))], valid_plot_taus[end]]
        fig_k_sweep = plot_frozen_k_sweep(
            dataset, taus_for_k_plot, scales;
            k_range = K_SWEEP_RANGE, feature_indices = feature_indices
        )
        save(joinpath(OUTPUT_PLOT_DIR, "$(MODEL_NAME)_k_plateau_analysis.$ext"), fig_k_sweep)

        fig_evo = Figure(size = (850, 500))
        ax_evo = Axis(fig_evo[1, 1],
            title = L"\text{Ewolucja lokalnego wymiaru } d_{\text{PR}}(\tau) \text{ w czasie (%$MODEL_NAME)}",
            xlabel = L"\text{Czas własny } \tau\,[\text{fm}/c]",
            ylabel = L"\langle d_{\text{PR}} \rangle"
        )
        main_color = AttractorsQGP.LPCA_COLORS[4]
        lines!(ax_evo, scan_res.tau_values, scan_res.mean_dims, color = main_color, linewidth = 2.8, label = L"\text{Lokalne PCA } (k = %$(K_NEIGHBOR))")
        scatter!(ax_evo, scan_res.tau_values, scan_res.mean_dims, color = main_color, markersize = 9)
        hlines!(ax_evo, [1.0], color = :gray50, linestyle = :dash, linewidth = 1.5, label = L"\text{Atraktor 1D } (d = 1)")
        hlines!(ax_evo, [Float64(D)], color = :black, linestyle = :dot, linewidth = 1.5, label = L"\text{Pełna przestrzeń } (d = %$D)")
        axislegend(ax_evo, position = :rt)
        save(joinpath(OUTPUT_PLOT_DIR, "$(MODEL_NAME)_dimension_evolution_tau.$ext"), fig_evo)

        println("  [OK] Zapisano komplet wykresów w: $OUTPUT_PLOT_DIR")
        figs = (grid_phys = fig_grid_phys, grid_scaled = fig_grid_scaled, k_sweep = fig_k_sweep, evolution = fig_evo)
    end

    println("\n" * "="^75)
    println(" ZAKOŃCZONO SUKCESEM!")
    println("="^75)

    return (
        model = MODEL_NAME,
        scales = scales,
        res_t0 = res_t0,
        res_late = res_late,
        sweep_t0 = sweep_t0,
        sweep_late = sweep_late,
        scan = scan_res,
        figures = figs
    )
end



