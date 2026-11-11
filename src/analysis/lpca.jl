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

@views function compute_local_pr_dimension(points::AbstractMatrix{<:Real}; k::Integer = 16)
    n, d = size(points)
    eff_k = clamp(Int(k), 2, n)

    tree = KDTree(points')
    d_pr = zeros(Float64, n)

    Y_buf = zeros(Float64, eff_k, d)
    C_buf = zeros(Float64, d, d)

    knn_idxs, _ = knn(tree, points', eff_k, true)

    for i in 1:n
        idxs = knn_idxs[i]
        Y = Y_buf[1:eff_k, :]
        for r in 1:eff_k
            for c in 1:d
                Y[r, c] = points[idxs[r], c]
            end
        end

        Y .-= mean(Y, dims = 1)
        C = C_buf[1:d, 1:d]
        mul!(C, Y', Y)
        C ./= (eff_k - 1)

        evals = eigvals!(Symmetric(C))
        for j in 1:d
            evals[j] = max(evals[j], 0.0)
        end

        s1 = sum(evals)
        s2 = sum(evals .^ 2)
        d_pr[i] = s2 > 0.0 ? (s1^2) / s2 : 1.0
    end

    return (
        mean = mean(d_pr),
        std = std(d_pr),
        d_pr = d_pr,
    )
end

function normalize_max(X)
    max_per_column = maximum(abs, X, dims = 1)
    max_per_column[max_per_column .== 0.0] .= 1.0
    return X ./ max_per_column
end

function compute_initial_scales(dataset::AbstractMatrix{<:Real}; tau_col::Integer = 1, feature_cols = 2:size(dataset, 2))
    t0 = minimum(view(dataset, :, tau_col))
    _, X0 = get_tau_slice(dataset, t0; feature_cols = feature_cols)
    scales = [std(X0[:, j]) for j in 1:size(X0, 2)]
    scales[scales .== 0.0] .= 1.0
    return scales
end

function apply_frozen_scales(X::AbstractMatrix{<:Real}, scales::AbstractVector{<:Real})
    return X ./ reshape(scales, 1, :)
end

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
        throw(ArgumentError("Unknown normalization method: $method"))
    end
end

function swiss_roll(n; noise = 0.0)
    t = (3π / 2) .* (1 .+ 2 .* rand(n))
    h = 10 .* rand(n)
    X = hcat(t .* cos.(t), h, t .* sin.(t))
    if noise > 0
        X .+= noise .* randn(size(X))
    end
    return X, t
end

function to_2d_local_lpca(dataset::AbstractArray{<:Real, 3})
    dataset_2d = reshape(permutedims(dataset, (2, 1, 3)), :, size(dataset, 3))
    return dataset_2d[.!isnan.(dataset_2d[:, 1]), :]
end

@views function compute_lpca(
        dataset::AbstractMatrix{<:Real},
        k::Int,
        n_slices::Int;
        feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize::Union{Symbol, Function, AbstractVector{<:Real}} = :standard,
        tol::Real = 0.01
    )
    taus = sort(unique(dataset[:, 1]))
    idxs = round.(Int, range(1, length(taus), length = n_slices))
    selected_taus = taus[idxs]

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
compute_lpca(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) =
    compute_lpca(to_2d_local_lpca(dataset), args...; kwargs...)

function scan_local_pr_dimension(
        dataset::AbstractMatrix{<:Real},
        tau_values::AbstractVector{<:Real};
        feature_indices::AbstractVector{<:Integer} = collect(2:size(dataset, 2)),
        normalize_method::Union{Symbol, Function, AbstractVector{<:Real}} = :standard,
        k::Integer = 16
    )
    n_slices = length(tau_values)
    mean_dims = zeros(Float64, n_slices)
    std_dims = zeros(Float64, n_slices)

    for (idx, tau) in enumerate(tau_values)
        _, raw_slice = get_tau_slice(dataset, tau; feature_cols = feature_indices)
        norm_slice = apply_normalization(raw_slice, normalize_method)
        res = compute_local_pr_dimension(norm_slice; k = k)

        mean_dims[idx] = res.mean
        std_dims[idx] = res.std
    end

    return (
        tau_values = copy(tau_values),
        mean_dims = mean_dims,
        std_dims = std_dims,
        k = k,
    )
end
scan_local_pr_dimension(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) =
    scan_local_pr_dimension(to_2d_local_lpca(dataset), args...; kwargs...)

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
    eff_k = clamp(k, 2, N)

    tree = KDTree(transpose(Z))
    knn_idxs, _ = knn(tree, transpose(Z), eff_k, true)

    d_pr = zeros(Float64, N)
    evals_mean = zeros(Float64, D)
    local_evals = zeros(Float64, D)

    for i in 1:N
        Y = Z[knn_idxs[i], :]
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
        spectral_ratio = evals_mean[end] / max(evals_mean[1], 1.0e-35),
        k = eff_k,
        tau = tau,
    )
end

function k_sweep_frozen(
        dataset::AbstractMatrix{<:Real},
        tau::Real,
        scales::AbstractVector{<:Real};
        k_range = [4:2:300...],
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
        eff_k = clamp(k, 2, N)
        d_pr = zeros(Float64, N)
        ev_accum = zeros(Float64, D)

        for i in 1:N
            Y = Z[knn_idxs_all[i][1:eff_k], :]
            Y .-= mean(Y, dims = 1)
            C = (transpose(Y) * Y) ./ (eff_k - 1)
            ev = max.(sort(eigvals(Symmetric(C)), rev = true), 0.0)
            ev_accum .+= ev
            s1 = sum(ev)
            s2 = sum(ev .^ 2)
            d_pr[i] = s2 > 0.0 ? (s1^2) / s2 : 1.0
        end

        mean_dpr[kid] = mean(d_pr)
        std_dpr[kid] = std(d_pr)
        mean_ratios[kid] = ev_accum[end] / max(ev_accum[1], 1.0e-35)
    end

    return (
        k_values = collect(k_range),
        mean_dpr = mean_dpr,
        std_dpr = std_dpr,
        mean_ratios = mean_ratios,
        tau = tau,
    )
end

function scan_lpca_frozen(
        dataset::AbstractMatrix{<:Real},
        scales::AbstractVector{<:Real};
        tau_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
        k::Integer = 16,
        D::Integer = 2,
        feature_cols = 2:size(dataset, D)
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
        k = k,
    )
end
