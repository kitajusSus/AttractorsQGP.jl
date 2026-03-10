using ManifoldLearning







"""
Locally Linear Embedding :
"""
function lle(X::AbstractMatrix{T}; k::Int=10, d::Int=2)
    n = size(X, 2)
    W = zeros(T, n, n)
    for i in 1:n
        # Find the k nearest neighbors of X[:, i]
        distances = sum((X .- X[:, i]).^2, dims=1)
        neighbors = sortperm(distances)[2:k+1]  # Exclude the point itself
        Z = X[:, neighbors] .- X[:, i]
        C = Z' * Z
        C += I * 1e-3 * trace(C)  # Regularization
        w = C \ ones(k)
        w /= sum(w)  # Normalize weights
        W[i, neighbors] = w'
    end
    M = (I - W)' * (I - W)
    eigvals, eigvecs = eigen(M)
    return eigvecs[:, 2:d+1]

end
