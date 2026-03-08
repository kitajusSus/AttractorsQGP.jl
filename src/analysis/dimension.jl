"""
Estimate intrinsic dimension with participation ratio.
"""
function estimate_dimension(data::AbstractMatrix{<:Real})
    @assert size(data, 1) > 1 "Need at least two samples."
    X = Matrix{Float64}(data)
    Xc = X .- mean(X, dims=1)
    C = cov(Xc)
    vals = eigvals(Symmetric(C))
    vals = real.(vals)
    vals = vals[vals .> 0]
    @assert !isempty(vals) "Covariance has no positive eigenvalues."
    return (sum(vals)^2) / sum(vals .^ 2)
end
