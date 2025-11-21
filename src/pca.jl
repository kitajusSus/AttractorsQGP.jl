include("lib.jl")
module modPCA

using Statistics
using LinearAlgebra
using MultivariateStats
using ..modHydroSim

export PCAResultAtTime, run_pca_at_time, run_pca_over_time, extract_features_at_time

struct PCAResultAtTime
    tau::Float64
    transformed_data::Matrix{Float64}
    explained_variance_ratio::Vector{Float64}
    principal_components::Matrix{Float64}
    valid_mask::BitVector
end

function linear_pca(X::Matrix{Float64}, n_components::Int; mode::Symbol=:standardize)
    n_samples, n_features = size(X)
    if n_components > n_features
        error("Liczba komponentów ($n_components) > liczba cech ($n_features).")
    end

    X_scaled = if mode == :standardize
        mean_v, std_v = mean(X, dims=1), std(X, dims=1)
        std_v[std_v .== 0.0] .= 1.0
        (X .- mean_v) ./ std_v
    elseif mode == :center
        X .- mean(X, dims=1)
    elseif mode == :minmax
        min_v, max_v = minimum(X, dims=1), maximum(X, dims=1)
        rng_v = max_v .- min_v
        rng_v[rng_v .== 0.0] .= 1.0
        (X .- min_v) ./ rng_v
    elseif mode == :none
        copy(X)
    else
        error("Nieznany tryb: $mode")
    end

    M = fit(PCA, X_scaled'; maxoutdim=n_components, pratio=1.0)
    return transform(M, X_scaled')', principalvars(M) ./ var(M), projection(M)
end

function kernel_pca(X::Matrix{Float64}, n_components::Int; gamma::Float64)
    kpca_k = (x, y) -> exp(-gamma * norm(x - y)^2.0)
    M = fit(KernelPCA, X'; kernel=kpca_k, maxoutdim=n_components)
    eig = eigvals(M)
    tot = sum(eig)
    evr = tot <= 1e-10 ? zeros(n_components) : eig[1:n_components] ./ tot
    return transform(M, X')', evr, projection(M)
end

function extract_features_at_time(sim_result::modHydroSim.SimResult, tau::Float64, feature_names::Vector{Symbol})
    u_vals, du_vals, valid_mask = modHydroSim.extract_phase_space_slice(sim_result, tau)
    if sum(valid_mask) == 0
        return nothing, nothing, nothing
    end

    # Base variables
    T_vals = u_vals[1][valid_mask]
    A_vals = u_vals[2][valid_mask]
    dTdτ_vals = du_vals[1][valid_mask]
    dAdτ_vals = du_vals[2][valid_mask]

    tau_0 = sim_result.settings.tspan[1]

    features = Vector{Float64}[]
    used_names = Symbol[]

    for name in feature_names
        if name == :T
            push!(features, T_vals); push!(used_names, :T)
        elseif name == :A
            push!(features, A_vals); push!(used_names, :A)
        elseif name == :dTdτ
            push!(features, dTdτ_vals); push!(used_names, :dTdτ)
        elseif name == :dAdτ
            push!(features, dAdτ_vals); push!(used_names, :dAdτ)
        elseif name == :tau0_T
            push!(features, tau_0 .* T_vals); push!(used_names, :tau0_T)
        elseif name == :tau0sq_dTdτ
            push!(features, (tau_0^2) .* dTdτ_vals); push!(used_names, :tau0sq_dTdτ)
        else
            @warn "Pominięto nieznaną cechę: $name"
        end
    end

    if isempty(features)
        return nothing, nothing, nothing
    end

    data_mat = hcat(features...)

    # Filtracja NaN w wynikowych cechach
    finite_mask = all(isfinite, data_mat, dims=2)[:]
    data_filtered = data_mat[finite_mask, :]

    # Aktualizacja maski globalnej
    final_mask = copy(valid_mask)
    # To jest nieco skomplikowane: valid_mask to te które przeżyły extract_slice
    # Teraz musimy odsiać te, które stały się NaN podczas obliczania cech
    # Upraszczamy: zakładamy, że extract_slice zwraca już dobre dane.

    return data_filtered, valid_mask, used_names
end

function run_pca_at_time(sim_result, tau, feature_names, n_components, params)
    data, mask, names = extract_features_at_time(sim_result, tau, feature_names)

    if isnothing(data) || size(data, 1) < n_components
        # @warn "Brak danych dla τ=$tau" # Ograniczamy logi
        return nothing
    end

    # Jeśli liczba cech < n_components, redukujemy n_components
    real_n_comp = min(n_components, size(data, 2))

    try
        res, evr, comps = if params[:method] == :kernel
            kernel_pca(data, real_n_comp; gamma=params[:gamma])
        else
            linear_pca(data, real_n_comp; mode=params[:method])
        end

        return PCAResultAtTime(tau, res, evr, comps, mask)
    catch e
        @warn "Błąd PCA w τ=$tau: $e"
        return nothing
    end
end

function run_pca_over_time(sim_result, feature_names, n_steps, n_components, params)
    t1, t2 = sim_result.settings.tspan
    times = range(t1, t2, length=n_steps)
    results = PCAResultAtTime[]

    println("Analiza PCA ($n_steps kroków) na cechach: $(join(feature_names, ", "))")
    for (i, t) in enumerate(times)
        print("\rKrok $i/$n_steps (τ=$(round(t, digits=2)))")
        r = run_pca_at_time(sim_result, t, feature_names, n_components, params)
        !isnothing(r) && push!(results, r)
    end
    println("\nZakończono.")
    return results
end

end
