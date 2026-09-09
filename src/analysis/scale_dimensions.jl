using NearestNeighbors
using Statistics

function neighbor_counts(tree, X, r)
    N = size(X, 1)
    counts = zeros(Int, N)

    for i in 1:N
        idxs = inrange(tree, X[i, :], r)
        counts[i] = length(idxs) - 1  # remove self
    end

    return counts
end


function scale_dimension_point(counts_r1, counts_r2, r1, r2)
    # avoid log(0)
    ϵ = 1.0e-12

    c1 = mean(counts_r1 .+ ϵ)
    c2 = mean(counts_r2 .+ ϵ)

    return (log(c2) - log(c1)) / (log(r2) - log(r1))
end


# d(r, τ)

function estimate_scale_dimension(
        X::AbstractMatrix{<:Real};
        r_grid = nothing
    )

    N = size(X, 1)

    tree = KDTree(permutedims(X))

    # auto scale selection if not provided
    if r_grid === nothing
        dists = pairwise(Euclidean(), X', dims = 2)
        r_min = quantile(vec(dists), 0.05)
        r_max = quantile(vec(dists), 0.5)
        r_grid = exp.(range(log(r_min), log(r_max), length = 20))
    end

    dims = Float64[]

    for i in 1:(length(r_grid) - 1)
        r1 = r_grid[i]
        r2 = r_grid[i + 1]

        c1 = neighbor_counts(tree, X, r1)
        c2 = neighbor_counts(tree, X, r2)

        d = scale_dimension_point(c1, c2, r1, r2)
        push!(dims, d)
    end

    return r_grid[1:(end - 1)], dims
end
