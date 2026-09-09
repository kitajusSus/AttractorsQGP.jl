using LinearAlgebra
using Statistics
using Distances
using NearestNeighbors

function participation_ratio(vals::AbstractVector{<:Real})
    vals = vals[vals .> 0]
    return (sum(vals)^2) / sum(vals .^ 2)
end


function estimate_intrinsic_dimension(data::AbstractMatrix{<:Real})
    @assert size(data, 1) > 1

    X = Matrix{Float64}(data)
    Xc = X .- mean(X, dims = 1)

    C = cov(Xc)
    vals = eigvals(Symmetric(C))
    vals = real.(vals)

    return participation_ratio(vals)
end

"""
Oblicza LID dla pojedynczego punktu na podstawie wektora odległości k-sąsiadów.
`dists` powinien zawierać odległości od 1-szego do k-tego prawdziwego sąsiada (bez odległości do samego siebie).
"""
function lid_point(dists::AbstractVector{<:Real})
    k = length(dists)

    rk = dists[end] # Ostatni element to maksymalna odległość r_k
    log_sum = 0.0
    for i in 1:k
        ri = dists[i]
        denom = ri == 0.0 ? 1.0e-10 : ri
        log_sum += log(rk / denom)
    end

    # MLE: 1 / ( (1/k) * log_sum ) -> k / log_sum
    return k / log_sum
end

function estimate_lid(X::AbstractMatrix{<:Real}; k::Integer = 20)
    N = size(X, 1)

    tree = KDTree(permutedims(X))
    lid_vals = zeros(N)
    for i in 1:N
        idxs, dists = knn(tree, X[i, :], k + 1, true)
        true_dists = Float64.(dists[2:end])
        lid_vals[i] = lid_point(true_dists)
    end
    return mean(lid_vals), lid_vals
end


# 3. TWO-NN (Facco et al.)

function estimate_twonn(X::AbstractMatrix{<:Real})
    N = size(X, 1)

    tree = KDTree(permutedims(X))

    μ = zeros(N)

    for i in 1:N
        idxs, dists = knn(tree, X[i, :], 3, true)
        r1, r2 = Float64(dists[2]), Float64(dists[3])
        μ[i] = r2 / r1
    end

    μ = μ[μ .> 0]

    # MLE :
    # d = 1 / mean(log μ)
    return 1 / mean(log.(μ))
end

"""
Effective dimension from singular values of Jacobian.
σ_i = singular values of J
"""
function spectral_dimension(σ::AbstractVector{<:Real})
    σ = σ[σ .> 0]

    return (sum(σ .^ 2)^2) / sum(σ .^ 4)
end


function estimate_pinn_dimension(J::AbstractMatrix{<:Real})
    σ = svdvals(Matrix{Float64}(J))
    return spectral_dimension(σ)
end

function scan_intrinsic_dimensions(
        dataset::AbstractMatrix{<:Real};
        k::Integer = 20,
        atol::Real = 1.0e-3,
        feature_cols::Union{Nothing, AbstractVector{<:Integer}} = nothing
    )

    cols = isnothing(feature_cols) ? collect(2:size(dataset, 2)) : feature_cols

    taus = sort(unique(Float64.(dataset[:, 1])))

    lid_out = Float64[]
    twonn_out = Float64[]
    pr_out = Float64[]

    for τ in taus
        _, Xτ = get_tau_slice(dataset, τ; atol = atol, feature_cols = cols)

        if size(Xτ, 1) > k

            X = Matrix{Float64}(Xτ)

            # LID
            lid, _ = estimate_lid(X; k = k)

            # TwoNN
            r2 = estimate_twonn(X)

            # PR
            Xc = X .- mean(X, dims = 1)
            C = cov(Xc)
            vals = eigvals(Symmetric(C))
            vals = real.(vals)
            vals = vals[vals .> 0]
            pr = (sum(vals)^2) / sum(vals .^ 2)

            push!(lid_out, lid)
            push!(twonn_out, r2)
            push!(pr_out, pr)
        end
    end

    return (
        taus = taus,
        lid = lid_out,
        twonn = twonn_out,
        participation_ratio = pr_out,
    )
end
