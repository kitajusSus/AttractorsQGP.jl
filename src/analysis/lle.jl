using LinearAlgebra
using KrylovKit
"""
Locally Linear Embedding (LLE)
X: matrix of input data with dimensions (D, N), `D` dimension of features,  `N` number of samples
k: numer of nearest neighbors to consider for each point
d: target dimension of the embedding space
"""
function lle(X::AbstractMatrix{T}; k::Int=10, d::Int=2) where T <: AbstractFloat
    D_dim, N = size(X)
    W = zeros(T, N, N)
    for i in 1:N
        distances = vec(sum((X .- X[:, i]).^2, dims=1))
        neighbors = sortperm(distances)[2:k+1]
        Z = X[:, neighbors] .- X[:, i]
        C = Z' * Z
        C += I(k) * 1e-3 * tr(C)
        w = C \ ones(T, k)
        w ./= sum(w)
        W[i, neighbors] = w
    end

    M = (I(N) - W)' * (I(N) - W)
    eigvals, eigvecs = eigen(Symmetric(M))
    return eigvecs[:, 2:d+1]
end



function run_lle_per_time(
    dataset::AbstractMatrix{<:Real};
    k::Integer=10,
    d::Integer=2,
    atol::Real=1e-8,
    feature_cols::AbstractVector{<:Integer}=collect(2:size(dataset, 2))
)
    @assert size(dataset, 2) >= 3 "Dataset must contain at least [tau, features...]."

    taus = sort(unique(Float64.(dataset[:, 1])))
    lle_results = Dict{Float64, Matrix{Float64}}()

    for tau in taus
        _, Xtau = get_tau_slice(dataset, tau; atol=atol, feature_cols=feature_cols)

        if size(Xtau, 1) > k
            res = lle(Matrix{Float64}(Xtau)'; k=k, d=d)
            lle_results[tau] = res
        end
    end

    return (taus=taus, lle_results=lle_results)
end
