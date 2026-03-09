using LinearAlgebra
using Statistics
using MultivariateStats

# """
# Run simple PCA using SVD.
# """
# function run_pca(data::AbstractMatrix{<:Real}; n_components::Integer=2)
#     @assert size(data, 1) > 1 "Need at least two samples."
#     @assert size(data, 2) > 0 "Need at least one feature."
#
#     X = Matrix{Float64}(data)
#     μ = mean(X, dims=1)
#     Xc = X .- μ
#     U, S, Vt = svd(Xc)
#     _, S, Vt = svd(Xc)
#
#     k = min(n_components, size(Vt, 2))
#     components = Vt[:, 1:k]
#     transformed = Xc * components
#     variance = (S .^ 2) ./ (size(X, 1) - 1)
#     ratio = variance ./ sum(variance)
#
#     return (transformed=transformed, components=components, explained_variance_ratio=ratio[1:k])
# end

"""
PCA linear with min-max normalization.

## Definition

```julia

function run_pca(data::AbstractMatrix{<:Real}; n_components::Integer=2)

    return (
        transformed=transformed,
        components=components,
        explained_variance_ratio=ratio[1:k],
    )
end
```
"""
function run_pca(data::AbstractMatrix{<:Real}; n_components::Integer=2)
    @assert size(data, 1) > 1 "Need at least two samples."
    @assert size(data, 2) > 0 "Need at least one feature."

    X = Matrix{Float64}(data)

    mn = minimum(X, dims=1)
    mx = maximum(X, dims=1)
    r = mx .- mn
    r[r .== 0.0] .= 1.0
    Xn = (X .- mn) ./ r

    _, S, Vt = svd(Xn)
    k = min(n_components, size(Vt, 2))

    components = Vt[:, 1:k]
    transformed = Xn * components

    variance = (S .^ 2) ./ (size(Xn, 1) - 1)
    ratio = variance ./ sum(variance)

    return (
        transformed=transformed,
        components=components,
        explained_variance_ratio=ratio[1:k],
    )
end

"""
Kernel PCA with RBF kernel.
```julia

function run_pca_kernel(data::AbstractMatrix{<:Real}; n_components::Integer=2, gamma::Float64=1.0)

    return (
        transformed=transformed,
        components=components,
        explained_variance_ratio=ratio,
    )
end
```
"""
function run_pca_kernel(data::AbstractMatrix{<:Real}; n_components::Integer=2, gamma::Float64=1.0)
    @assert size(data, 1) > 1 "Need at least two samples."
    @assert size(data, 2) > 0 "Need at least one feature."

    X = Matrix{Float64}(data)
    k = min(n_components, size(X, 1), size(X, 2))

    kernel = (x, y) -> exp(-gamma * norm(x - y)^2)
    M = fit(KernelPCA, X'; kernel=kernel, maxoutdim=k)

    transformed = transform(M, X')'

    ev = eigvals(M)
    tot = sum(ev)
    ratio_full = tot <= 1e-12 ? zeros(length(ev)) : ev ./ tot
    ratio = ratio_full[1:k]

    components = try
        projection(M)
    catch
        Matrix{Float64}(undef, size(X, 2), 0)
    end

    return (
        transformed=transformed,
        components=components,
        explained_variance_ratio=ratio,
    )
end

"""
Run PCA separately for each unique time `tau` in a dataset [tau, features...].
"""
function run_pca_per_time(
    dataset::AbstractMatrix{<:Real};
    n_components::Integer=2,
    method::Symbol=:minmax,
    gamma::Float64=1.0,
)
    @assert size(dataset, 2) >= 3 "Dataset must contain at least [tau, T, A]."

    taus = sort(unique(Float64.(dataset[:, 1])))
    pca_results = Dict{Float64, NamedTuple}()
    evr = fill(NaN, length(taus), n_components)

    for (i, tau) in pairs(taus)
        idx = findall(isapprox.(dataset[:, 1], tau; atol=1e-8))
        data_tau = Matrix{Float64}(dataset[idx, 2:end])

        if size(data_tau, 1) > 1
            pca_tau = if method === :minmax
                run_pca(data_tau; n_components=n_components)
            elseif method === :kernel
                run_pca_kernel(data_tau; n_components=n_components, gamma=gamma)
            else
                error("Unknown PCA method. Choose :minmax or :kernel.")
            end

            pca_results[tau] = pca_tau
            n_local = min(length(pca_tau.explained_variance_ratio), n_components)
            evr[i, 1:n_local] .= pca_tau.explained_variance_ratio[1:n_local]
        end
    end

    return (taus=taus, pca_results=pca_results, explained_variance_ratio=evr)
end
