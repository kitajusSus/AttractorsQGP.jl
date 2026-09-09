using LinearAlgebra
using KrylovKit
using Statistics
using NearestNeighbors
using SparseArrays
using LinearMaps


"""
Locally Linear Embedding (LLE)
X: matrix of input data with dimensions (D, N), `D` dimension of features,  `N` number of samples
k: numer of nearest neighbors to consider for each point
d: target dimension of the embedding space
"""
function lle(X::AbstractMatrix{T}; k::Int = 10, d::Int = 2, kdtree = nothing) where {T <: AbstractFloat}
    D_dim, N = size(X)

    I_idx = Vector{Int}(undef, N * k)
    J_idx = Vector{Int}(undef, N * k)
    V = Vector{T}(undef, N * k)

    if isnothing(kdtree)
        kdtree = KDTree(X)
    end

    idxs, _ = knn(kdtree, X, k + 1)

    rhs = ones(T, k)
    w_tmp = Vector{T}(undef, k)

    ptr = 1
    for i in 1:N
        neighbors = @view idxs[i][2:(k + 1)]

        Z = X[:, neighbors] .- X[:, i]
        C = Z' * Z
        C += I(k) * (1.0e-3 * tr(C) + 1.0e-10)

        F_C = cholesky!(Hermitian(C))
        ldiv!(w_tmp, F_C, rhs)
        w_tmp ./= sum(w_tmp)

        for j in 1:k
            I_idx[ptr] = i
            J_idx[ptr] = neighbors[j]
            V[ptr] = w_tmp[j]
            ptr += 1
        end
    end

    W = sparse(I_idx, J_idx, V, N, N)

    A = I - W
    M = A' * A
    sigma = -1.0e-5
    for i in 1:N
        M[i, i] -= sigma
    end
    F = cholesky(Symmetric(M))
    inv_f = x -> F \ x

    _, vecs, _ = eigsolve(inv_f, N, d + 1, :LM; ishermitian = true, maxiter = 300, tol = 1.0e-5)

    return reduce(hcat, vecs[2:(d + 1)])
end

function run_lle_per_time(
        dataset::AbstractMatrix{<:Real};
        k::Integer = 10,
        d::Integer = 2,
        atol::Real = 1.0e-8,
        feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2))
    )

    taus = sort(unique(Float64.(dataset[:, 1])))
    lle_results = Dict{Float64, Matrix{Float64}}()

    for tau in taus
        _, Xtau = get_tau_slice(dataset, tau; atol = atol, feature_cols = feature_cols)

        if size(Xtau, 1) > k
            X_input = Matrix{Float64}(Xtau)'
            res = lle(X_input; k = k, d = d)
            lle_results[tau] = res
        end
    end

    return (taus = taus, lle_results = lle_results)
end


function run_lle_for_selected_taus(
        dataset::AbstractMatrix{<:Real},
        target_taus::AbstractVector{<:Real};
        k::Integer = 20,
        d::Integer = 2,
        atol::Real = 1.0e-3,
        feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2))
    )
    lle_results = Dict{Float64, Matrix{Float64}}()

    for tau in target_taus
        _, Xtau = get_tau_slice(dataset, tau; atol = atol, feature_cols = feature_cols)

        if size(Xtau, 1) > k
            println("Obliczam LLE dla tau = $tau (liczba punktów: $(size(Xtau, 1)))...")
            res = lle(Matrix{Float64}(Xtau)'; k = k, d = d)
            lle_results[tau] = Matrix(res')
        end
    end

    return (taus = target_taus, lle_results = lle_results)
end


function lle_spectrum(X; k = 10, kdtree = nothing, nλ = 10)
    D, N = size(X)

    I_idx = Vector{Int}(undef, N * k)
    J_idx = Vector{Int}(undef, N * k)
    V = Vector{Float64}(undef, N * k)

    if isnothing(kdtree)
        kdtree = KDTree(X)
    end

    idxs, _ = knn(kdtree, X, k + 1)

    rhs = ones(Float64, k)
    w_tmp = Vector{Float64}(undef, k)

    ptr = 1
    for i in 1:N
        neighbor_idx = @view idxs[i][2:(k + 1)]

        Z = X[:, neighbor_idx] .- X[:, i]
        C = Z' * Z
        C += I(k) * (1.0e-3 * tr(C) + 1.0e-10)

        F_C = cholesky!(Hermitian(C))
        ldiv!(w_tmp, F_C, rhs)
        w_tmp ./= sum(w_tmp)

        for j in 1:k
            I_idx[ptr] = i
            J_idx[ptr] = neighbor_idx[j]
            V[ptr] = w_tmp[j]
            ptr += 1
        end
    end

    W = sparse(I_idx, J_idx, V, N, N)

    A = I - W
    M = A' * A
    sigma = -1.0e-5
    for i in 1:N
        M[i, i] -= sigma
    end
    F = cholesky(Symmetric(M))
    inv_f = x -> F \ x

    vals_inv, _, _ = eigsolve(inv_f, N, nλ, :LM; ishermitian = true, maxiter = 300, tol = 1.0e-5)

    vals = (1 ./ real(vals_inv)) .+ sigma
    return sort(vals)
end

"""
    flatten_k_values(k_values)

Flatten a collection of integers/ranges/nested vectors into a flat sequence of integers.
For example, `5:5:50` is returned as is, but `[5:1:200]` is returned as a flat vector `5:200`.
"""
function flatten_k_values(k_values)
    if k_values isa Integer
        return [Int(k_values)]
    end
    if k_values isa AbstractRange{<:Integer}
        return k_values
    end
    if k_values isa AbstractVector{<:Integer}
        return k_values
    end

    flat = Int[]
    for item in k_values
        if item isa AbstractRange
            append!(flat, item)
        elseif item isa Integer
            push!(flat, item)
        elseif item isa AbstractVector || item isa Tuple
            append!(flat, flatten_k_values(item))
        else
            push!(flat, Int(item))
        end
    end
    return flat
end

# i love ai it gave me this blas code implementation
@views function lle_spectrum_over_k(dataset; tau, k_values = 5:5:50, atol = 1.0e-8, nλ = 10)
    k_values = flatten_k_values(k_values)
    _, Xτ = get_tau_slice(dataset, tau; atol = atol)

    X = Matrix{Float64}(Xτ)'

    spectra = Vector{Union{Nothing, Vector{Float64}}}(undef, length(k_values))
    fill!(spectra, nothing)
    tree = KDTree(X)

    old_threads = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)
    try
        Threads.@threads for i in 1:length(k_values)
            k = k_values[i]
            if size(X, 2) > k + 2
                spectra[i] = lle_spectrum(X; k = k, kdtree = tree, nλ = nλ)
            end
        end
    finally
        LinearAlgebra.BLAS.set_num_threads(old_threads)
    end

    valid_spectra = Vector{Float64}[s for s in spectra if !isnothing(s)]
    valid_ks = [k_values[i] for i in 1:length(k_values) if !isnothing(spectra[i])]

    return valid_spectra, valid_ks
end

"""
Liczy średnią i odchylenie standardowe dla spektrum wartości własnych 
lub czegokolwiek innego 
"""
@views function spectrum_statistics(spectra)
    max_len = minimum(length.(spectra))
    S = reduce(hcat, [s[1:max_len] for s in spectra])

    μ = mean(S, dims = 2)[:]
    σ = std(S, dims = 2)[:]

    return μ, σ
end


function scan_lle_spectrum(dataset; taus, k_values = 5:5:50)
    results = Dict()

    for τ in taus
        spectra, ks = lle_spectrum_over_k(dataset; tau = τ, k_values = k_values)

        if length(spectra) < 2
            continue
        end

        μ, σ = spectrum_statistics(spectra)

        results[τ] = (mean = μ, std = σ, k = ks)
    end

    return results
end


function to_2d_local_lle(dataset::AbstractArray{<:Real, 3})
    dataset_2d = reshape(permutedims(dataset, (2, 1, 3)), :, size(dataset, 3))
    valid_rows = .!isnan.(dataset_2d[:, 1])
    return dataset_2d[valid_rows, :]
end


function run_lle_per_time(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return run_lle_per_time(to_2d_local_lle(dataset), args...; kwargs...)
end
function run_lle_for_selected_taus(dataset::AbstractArray{<:Real, 3}, args...; kwargs...)
    return run_lle_for_selected_taus(to_2d_local_lle(dataset), args...; kwargs...)
end
