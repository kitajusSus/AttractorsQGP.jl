include("lib.jl")
module modPCA

using Statistics
using LinearAlgebra
using MultivariateStats
using ..modHydroSim

export PCAResultAtTime, run_pca_at_time, run_pca_over_time

struct PCAResultAtTime
    tau::Float64
    transformed_data::Matrix{Float64}
    explained_variance_ratio::Vector{Float64}
    principal_components::Matrix{Float64}
    valid_mask::BitVector
end

function linear_pca(X::Matrix{Float64}, n::Int; mode::Symbol=:standardize)
    X_scaled = if mode == :standardize
        m, s = mean(X, dims=1), std(X, dims=1)
        s[s.==0] .= 1.0
        (X .- m) ./ s
    elseif mode == :center
        X .- mean(X, dims=1)
    elseif mode == :minmax
        mn, mx = minimum(X, dims=1), maximum(X, dims=1)
        r = mx .- mn
        r[r.==0] .= 1.0
        (X .- mn) ./ r
    else
        copy(X)
    end

    M = fit(PCA, X_scaled'; maxoutdim=n, pratio=1.0)
    return transform(M, X_scaled')', principalvars(M)./var(M), projection(M)
end

function kernel_pca(X::Matrix{Float64}, n::Int; gamma::Float64)
    k = (x,y) -> exp(-gamma * norm(x-y)^2)
    M = fit(KernelPCA, X'; kernel=k, maxoutdim=n)
    ev = eigvals(M)
    tot = sum(ev)
    r = tot <= 1e-10 ? zeros(n) : ev[1:n] ./ tot
    return transform(M, X')', r, projection(M)
end

function extract_features(simres, t, feats::Vector{Symbol})
    u, du, mask = modHydroSim.extract_phase_space_slice(simres, t)
    if !any(mask); return nothing, nothing; end

    t0 = simres.settings.tspan[1]
    data = Vector{Float64}[]

    for f in feats
        if f == :T
            push!(data, u[1][mask])
        elseif f == :A
            push!(data, u[2][mask])
        elseif f == :dTdτ
            push!(data, du[1][mask])
        elseif f == :dAdτ
            push!(data, du[2][mask])
        elseif f == :tau0_T
            push!(data, t0 .* u[1][mask])
        elseif f == :tau0sq_dTdτ
            push!(data, (t0^2) .* du[1][mask])
        end
    end

    mat = hcat(data...)
    fin_mask = all(isfinite, mat, dims=2)[:]
    return mat[fin_mask, :], mask
end

function run_pca_at_time(simres, t, feats, n, params)
    data, mask = extract_features(simres, t, feats)
    if isnothing(data) || size(data, 1) < n; return nothing; end

    real_n = min(n, size(data, 2))

    res, evr, comps = if params[:method] == :kernel
        kernel_pca(data, real_n; gamma=params[:gamma])
    else
        linear_pca(data, real_n; mode=params[:method])
    end

    return PCAResultAtTime(t, res, evr, comps, mask)
end

function run_pca_over_time(simres, feats, n_steps, n_comp, params)
    ts = range(simres.settings.tspan..., length=n_steps)
    res = PCAResultAtTime[]
    for t in ts
        r = run_pca_at_time(simres, t, feats, n_comp, params)
        !isnothing(r) && push!(res, r)
    end
    return res
end

end
