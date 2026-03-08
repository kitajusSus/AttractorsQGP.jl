using LinearAlgebra
using Statistics

"""
Run simple PCA using SVD.
"""
function run_pca(data::AbstractMatrix{<:Real}; n_components::Integer=2)
    @assert size(data, 1) > 1 "Need at least two samples."
    @assert size(data, 2) > 0 "Need at least one feature."

    X = Matrix{Float64}(data)
    μ = mean(X, dims=1)
    Xc = X .- μ
    U, S, Vt = svd(Xc)

    k = min(n_components, size(Vt, 2))
    components = Vt[:, 1:k]
    transformed = Xc * components
    variance = (S .^ 2) ./ (size(X, 1) - 1)
    ratio = variance ./ sum(variance)

    return (transformed=transformed, components=components, explained_variance_ratio=ratio[1:k])
end
