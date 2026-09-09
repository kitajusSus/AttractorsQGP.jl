using LinearAlgebra
using NearestNeighbors
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
function apply_normalization(X::AbstractMatrix{<:Real}, method::Union{Symbol, Function} = :standard)
    if method isa Function
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
        throw(ArgumentError("Unknown normalization method: $method. Supported: :max, :minmax, :zscore, :none, or a Function."))
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


# nizej jest testowe nie ważnbe
@views function dynamic_lpca_analysis(
        dataset::AbstractMatrix{<:Real};
        K::Int = 15,
        eta::Real = 0.95,
        delta::Real = 0.05,
        feature_cols::Union{AbstractVector{<:Integer}, Nothing} = nothing
    )
    taus = sort(unique(Float64.(dataset[:, 1])))
    d_bar = zeros(length(taus))
    cols = isnothing(feature_cols) ? (2:size(dataset, 2)) : feature_cols
    d = length(cols)

    for (nτ, τ) in enumerate(taus)
        mask = dataset[:, 1] .== τ
        X_tau = dataset[mask, cols]
        n_points = size(X_tau, 1)
        k = min(K, n_points - 1)

        if k < 2
            d_bar[nτ] = 1.0
            continue
        end

        X_norm = normalize_max(X_tau)
        tree = KDTree(X_norm')

        idxs, _ = knn(tree, X_norm', k + 1, true)
        d_sum = 0

        X_local_buf = zeros(Float64, k, d)

        for i in 1:n_points
            neighbors = idxs[i][2:end]

            X_local = X_local_buf[1:k, :]
            for r in 1:k
                for c in 1:d
                    X_local[r, c] = X_norm[neighbors[r], c]
                end
            end

            X_local .-= mean(X_local, dims = 1)
            S = svdvals!(X_local)

            λ_sum = 0.0
            for s in S
                λ_sum += s^2
            end

            if λ_sum ≈ 0.0
                d_sum += 1
                continue
            end

            evr = 0.0
            d_val = 1
            for j in eachindex(S)
                evr += (S[j]^2) / λ_sum
                if evr >= eta
                    d_val = j
                    break
                end
            end
            d_sum += d_val
        end
        d_bar[nτ] = d_sum / n_points
    end

    tau_LPCA = NaN
    threshold = 1.0 + delta
    for i in eachindex(taus)
        if all(d_bar[i:end] .<= threshold)
            tau_LPCA = taus[i]
            break
        end
    end

    return (; taus, d_bar, tau_LPCA)
end


@views function compute_lpca_entropy(
        dataset::AbstractMatrix{<:Real},
        k::Int;
        n_slices::Union{Nothing, Int} = nothing,
        feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        tol::Real = 1.0e-8
    )
    taus = sort(unique(dataset[:, 1]))
    if n_slices === nothing
        selected_taus = taus
    else
        idxs = round.(Int, range(1, length(taus), length = n_slices))
        selected_taus = taus[idxs]
    end

    n_selected = length(selected_taus)
    tau_values = zeros(Float64, n_selected)
    mean_entropy = zeros(Float64, n_selected)
    std_entropy = zeros(Float64, n_selected)

    d = length(feature_cols)

    for (idx, τ) in enumerate(selected_taus)
        _, X_tau = get_tau_slice(dataset, τ; feature_cols = feature_cols)
        X_norm = normalize_max(X_tau)
        n_points = size(X_norm, 1)
        k_local = min(k, n_points)

        if k_local < 2
            tau_values[idx] = τ
            mean_entropy[idx] = 0.0
            std_entropy[idx] = 0.0
            continue
        end

        tree = KDTree(X_norm')
        idxs_knn, _ = knn(tree, X_norm', k_local, true)
        entropies = zeros(n_points)

        Y_buf = zeros(Float64, k_local, d)
        C_buf = zeros(Float64, d, d)

        for i in 1:n_points
            neighbors_idx = idxs_knn[i]

            Y = Y_buf[1:k_local, :]
            for r in 1:k_local
                for c in 1:d
                    Y[r, c] = X_norm[neighbors_idx[r], c]
                end
            end

            μ = mean(Y, dims = 1)
            Y .-= μ

            C = C_buf[1:d, 1:d]
            mul!(C, Y', Y)
            C ./= k_local

            λ = eigvals!(Symmetric(C))
            λ .= max.(λ, 0.0)
            λ_sum = sum(λ)

            if λ_sum ≈ 0.0
                continue
            end

            S = 0.0
            r_count = 0
            for val in λ
                λ_norm_val = val / λ_sum
                if λ_norm_val > tol
                    S -= λ_norm_val * log(λ_norm_val)
                    r_count += 1
                end
            end

            if r_count <= 1
                continue
            end

            entropies[i] = S / log(r_count)
        end

        tau_values[idx] = τ
        mean_entropy[idx] = mean(entropies)
        std_entropy[idx] = std(entropies)
    end
    return tau_values, mean_entropy, std_entropy
end

function compute_stable_lpca_collapse(
        dataset::AbstractMatrix{<:Real};
        zakres_K::AbstractVector{<:Integer} = [10, 20, 30, 40],
        Scrit::Real = 0.2,
        delta_k::Real = 0.05,
        n_slices::Union{Nothing, Int} = nothing,
        feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        tol::Real = 1.0e-8,
        norm_type::Symbol = :embedding
    )
    taus = sort(unique(dataset[:, 1]))
    if n_slices !== nothing
        idxs = round.(Int, range(1, length(taus), length = n_slices))
        wybrane_tau = taus[idxs]
    else
        wybrane_tau = taus
    end

    n_taus = length(wybrane_tau)
    n_k = length(zakres_K)
    k_max = maximum(zakres_K)
    d = length(feature_cols)

    S_PCA_matrix = zeros(Float64, n_taus, n_k)

    for (tau_idx, tau) in enumerate(wybrane_tau)
        _, X_tau = get_tau_slice(dataset, tau; feature_cols = feature_cols)
        X_norm = normalize_max(X_tau)

        n_points, d_size = size(X_norm)
        k_max_actual = min(k_max, n_points)
        if k_max_actual < 2
            continue
        end

        tree = KDTree(X_norm')
        knn_idxs, _ = knn(tree, X_norm', k_max_actual, true)

        Y_buf = zeros(Float64, k_max_actual, d)
        C_buf = zeros(Float64, d, d)

        for (k_idx, k_val) in enumerate(zakres_K)
            k_actual = min(k_val, n_points)
            if k_actual < 2
                S_PCA_matrix[tau_idx, k_idx] = 0.0
                continue
            end

            entropies = zeros(Float64, n_points)
            for i in 1:n_points
                @views neighbor_idxs = knn_idxs[i][1:k_actual]

                Y = Y_buf[1:k_actual, :]
                for r in 1:k_actual
                    for c in 1:d
                        Y[r, c] = X_norm[neighbor_idxs[r], c]
                    end
                end

                μ = mean(Y, dims = 1)
                Y .-= μ

                C = C_buf[1:d, 1:d]
                mul!(C, Y', Y)
                C ./= k_actual

                λ = eigvals!(Symmetric(C))
                λ .= max.(λ, 0.0)
                sum_λ = sum(λ)

                if sum_λ ≈ 0.0
                    entropies[i] = 0.0
                    continue
                end

                S_i = 0.0
                r_count = 0
                for val in λ
                    λ_norm = val / sum_λ
                    if λ_norm > tol
                        S_i -= λ_norm * log(λ_norm)
                        r_count += 1
                    end
                end

                if norm_type == :embedding
                    entropies[i] = S_i / log(d)
                elseif norm_type == :active
                    entropies[i] = r_count <= 1 ? 0.0 : S_i / log(r_count)
                else
                    entropies[i] = S_i
                end
            end
            S_PCA_matrix[tau_idx, k_idx] = mean(entropies)
        end
    end

    S_PCA_mean = mean(S_PCA_matrix, dims = 2)[:]
    S_PCA_std = std(S_PCA_matrix, dims = 2)[:]

    tau_LPCA_k = fill(NaN, n_k)
    for (k_idx, k_val) in enumerate(zakres_K)
        for (tau_idx, tau) in enumerate(wybrane_tau)
            if S_PCA_matrix[tau_idx, k_idx] < Scrit
                tau_LPCA_k[k_idx] = tau
                break
            end
        end
    end

    tau_LPCA_stable = NaN
    for (tau_idx, tau) in enumerate(wybrane_tau)
        if S_PCA_mean[tau_idx] < Scrit && S_PCA_std[tau_idx] < delta_k
            tau_LPCA_stable = tau
            break
        end
    end

    return (;
        taus = wybrane_tau,
        S_PCA_matrix,
        S_PCA_mean,
        S_PCA_std,
        tau_LPCA_k,
        tau_LPCA_stable,
    )
end

@views function compute_lpca_principal_angles(
        dataset::AbstractMatrix{<:Real};
        k::Int = 20,
        subspace_dim::Int = 1,
        n_slices::Union{Nothing, Int} = nothing,
        feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2))
    )
    taus = sort(unique(dataset[:, 1]))
    if n_slices !== nothing
        idxs = round.(Int, range(1, length(taus), length = n_slices))
        wybrane_tau = taus[idxs]
    else
        wybrane_tau = taus
    end

    mean_angles = Float64[]
    std_angles = Float64[]
    all_angles_per_tau = Vector{Vector{Float64}}()

    for tau in wybrane_tau
        _, X_tau = get_tau_slice(dataset, tau; feature_cols = feature_cols)
        X_norm = normalize_max(X_tau)

        n_points, d = size(X_norm)
        k_actual = min(k, n_points)
        if k_actual < 2
            push!(mean_angles, 0.0)
            push!(std_angles, 0.0)
            push!(all_angles_per_tau, Float64[])
            continue
        end

        tree = KDTree(X_norm')
        knn_idxs, _ = knn(tree, X_norm', k_actual, true)

        m = min(subspace_dim, d)
        V = Vector{Matrix{Float64}}(undef, n_points)

        for i in 1:n_points
            neighbor_idxs = knn_idxs[i]
            neighbors = X_norm[neighbor_idxs, :]
            μ = mean(neighbors, dims = 1)
            Y = neighbors .- μ

            F = svd(Y)
            V[i] = Matrix{Float64}(F.V[:, 1:m])
        end

        angles = Float64[]
        for i in 1:n_points
            if length(knn_idxs[i]) < 2
                continue
            end
            j = knn_idxs[i][2]

            M_proj = V[i]' * V[j]
            sigmas = svdvals(M_proj)
            sigmas = clamp.(sigmas, 0.0, 1.0)

            for σ in sigmas
                push!(angles, acos(σ) * (180.0 / π))
            end
        end

        if isempty(angles)
            push!(mean_angles, 0.0)
            push!(std_angles, 0.0)
            push!(all_angles_per_tau, Float64[])
        else
            push!(mean_angles, mean(angles))
            push!(std_angles, std(angles))
            push!(all_angles_per_tau, angles)
        end
    end

    return (;
        taus = wybrane_tau,
        mean_angles,
        std_angles,
        all_angles_per_tau,
    )
end


function to_2d_local_lpca(dataset::AbstractArray{<:Real, 3})
    dataset_2d = reshape(permutedims(dataset, (2, 1, 3)), :, size(dataset, 3))
    valid_rows = .!isnan.(dataset_2d[:, 1])
    return dataset_2d[valid_rows, :]
end


function dynamic_lpca_analysis(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return dynamic_lpca_analysis(to_2d_local_lpca(dataset), args...; kwargs...)
end
function compute_lpca_entropy(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return compute_lpca_entropy(to_2d_local_lpca(dataset), args...; kwargs...)
end
function compute_stable_lpca_collapse(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return compute_stable_lpca_collapse(to_2d_local_lpca(dataset), args...; kwargs...)
end
function compute_lpca_principal_angles(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return compute_lpca_principal_angles(to_2d_local_lpca(dataset), args...; kwargs...)
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
        relative_difference = relative_difference
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
        max_difference_dimensionless = max_difference_dimensionless
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
        dimension_fractions = dimension_fractions
    )
end
analyze_dimension_distribution(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = analyze_dimension_distribution(to_2d_local_lpca(dataset), args...; kwargs...)

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
        results = results
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
        envelopes = envelopes
    )
end

"""
    PointwiseDimensionSlice

Stores coordinates and corresponding local dimensions at a specific proper time.
"""
struct PointwiseDimensionSlice{T<:Real}
    tau::T
    coordinates::Matrix{T}
    dimensions::Vector{T}
    feature_names::Vector{Symbol}
end

"""
    compute_pointwise_dimensions(dataset, tau; feature_indices, feature_names, k_neighbor, tolerance, normalize_method)
"""
function compute_pointwise_dimensions(
    dataset::AbstractMatrix{<:Real},
    tau::Real;
    feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
    feature_names::AbstractVector{Symbol} = Symbol[],
    k_neighbor::Integer = 24,
    tolerance::Real = 0.01,
    normalize_method::Symbol = :max
)
    tau_col = dataset[:, 1]
    rows = findall(isapprox.(tau_col, tau; atol = 1.0e-5))
    if isempty(rows)
        nearest_idx = argmin(abs.(tau_col .- tau))
        nearest_tau = tau_col[nearest_idx]
        rows = findall(isapprox.(tau_col, nearest_tau; atol = 1.0e-5))
    end

    raw_features = Matrix{Float64}(dataset[rows, feature_indices])
    normalized_features = apply_normalization(raw_features, normalize_method)
    local_dims = dims(normalized_features; k = k_neighbor, tol = tolerance)

    return PointwiseDimensionSlice(
        Float64(tau),
        raw_features,
        local_dims,
        isempty(feature_names) ? [Symbol("x$i") for i in 1:length(feature_indices)] : feature_names
    )
end
compute_pointwise_dimensions(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = compute_pointwise_dimensions(to_2d_local_lpca(dataset), args...; kwargs...)

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
        weights = weights
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
        delta = delta
    )
end
scan_soft_weighted_dimension(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) =
    scan_soft_weighted_dimension(to_2d_local_lpca(dataset), args...; kwargs...)


