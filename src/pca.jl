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

function linear_pca(X::Matrix{Float64}, n_components::Int; mode::Symbol=:standardize)
    n_samples, n_features = size(X)
    if n_components > n_features
        error(
            "Liczba komponentów ($n_components) nie może być większa niż liczba cech ($n_features).",
        )
    end

    X_scaled = if mode == :standardize
        mean_vector = mean(X, dims=1)
        std_vector = std(X, dims=1)
        std_vector[std_vector.==0.0] .= 1.0
        (X .- mean_vector) ./ std_vector
    elseif mode == :center
        mean_vector = mean(X, dims=1)
        X .- mean_vector
    elseif mode == :minmax
        min_vals = minimum(X, dims=1)
        max_vals = maximum(X, dims=1)
        range_vals = max_vals .- min_vals
        range_vals[range_vals.==0.0] .= 1.0
        (X .- min_vals) ./ range_vals
    elseif mode == :none
        copy(X)
    else
        error("Nieznany tryb skalowania liniowego: $mode")
    end

    X_transposed = X_scaled'
    M_linear = fit(PCA, X_transposed; maxoutdim=n_components, pratio=1.0)
    transformed_data = MultivariateStats.transform(M_linear, X_transposed)'
    explained_variance_ratio = principalvars(M_linear) ./ var(M_linear)
    projection_matrix = projection(M_linear)

    return transformed_data, explained_variance_ratio, projection_matrix
end

function kernel_pca(X::Matrix{Float64}, n_components::Int; gamma::Float64)
    n_samples, n_features = size(X)
    X_transposed = X'
    kpca_kernel = (x, y) -> exp(-gamma * norm(x - y)^2.0)
    M_kernel = fit(KernelPCA, X_transposed; kernel=kpca_kernel, maxoutdim=n_components)
    transformed_data = MultivariateStats.transform(M_kernel, X_transposed)'
    all_eigenvalues = eigvals(M_kernel)
    total_variance = sum(all_eigenvalues)

    explained_variance_ratio = if total_variance <= 1e-10
        zeros(n_components)
    else
        all_eigenvalues[1:n_components] ./ total_variance
    end

    selected_alphas = projection(M_kernel)
    return transformed_data, explained_variance_ratio, selected_alphas
end

"""
    run_pca_at_time(sim_result, tau, feature_indices, n_components, pca_method_params)

Uruchamia analizę PCA dla pojedynczego kroku czasowego `tau`.
"""
function run_pca_at_time(
    sim_result::modHydroSim.SimResult,
    tau::Float64,
    feature_indices::Vector{Int},
    n_components::Int,
    pca_method_params::Dict,
)
    method = pca_method_params[:method]

    u_vals, du_vals, valid_mask =
        modHydroSim.extract_phase_space_slice(sim_result, tau)

    if sum(valid_mask) < n_components
        @warn "Zbyt mało prawidłowych danych (znaleziono $(sum(valid_mask))) w czasie τ=$tau. Pomijanie kroku."
        return nothing
    end

    all_features = [u_vals[1], u_vals[2], du_vals[1], du_vals[2]]

    if any(idx -> idx > length(all_features), feature_indices)
        error("Indeks cechy poza zakresem. Dostępne indeksy: 1-$(length(all_features))")
    end

    data_matrix = hcat([all_features[idx][valid_mask] for idx in feature_indices]...)

    if size(unique(data_matrix, dims=1), 1) < n_components
        @warn "Zbyt mało unikalnych danych (znaleziono $(size(unique(data_matrix, dims=1), 1))) w czasie τ=$tau. Pomijanie kroku."
        return nothing
    end

    local transformed_data, explained_ratio, components
    try
        if method == :kernel
            gamma = pca_method_params[:gamma]
            transformed_data, explained_ratio, components =
                kernel_pca(data_matrix, n_components; gamma=gamma)
        else
            transformed_data, explained_ratio, components =
                linear_pca(data_matrix, n_components; mode=method)
        end

        if any(isnan, transformed_data) || any(isnan, explained_ratio)
            @warn "Wynik PCA dla τ=$tau zawiera NaN. Pomijanie kroku."
            return nothing
        end

        return PCAResultAtTime(
            tau,
            transformed_data,
            explained_ratio,
            components,
            valid_mask,
        )
    catch e
        @warn "Błąd podczas przetwarzania PCA w czasie τ=$tau. Pomijanie kroku. Błąd: $e"
        return nothing
    end
end

"""
    run_pca_over_time(sim_result, feature_indices, n_pca_steps, n_components, pca_method_params)

Uruchamia analizę PCA w pętli dla `n_pca_steps` kroków czasowych.
"""
function run_pca_over_time(
    sim_result::modHydroSim.SimResult,
    feature_indices::Vector{Int},
    n_pca_steps::Int,
    n_components::Int,
    pca_method_params::Dict,
)
    t_start, t_end = sim_result.settings.tspan
    sample_times = range(t_start, stop=t_end, length=n_pca_steps)
    pca_results_vector = PCAResultAtTime[]

    println("Uruchamianie analizy PCA dla $n_pca_steps kroków czasowych...")

    for (i, tau) in enumerate(sample_times)
        print("\rPrzetwarzanie kroku czasowego: $i/$n_pca_steps (τ = $(round(tau, digits=2)) fm/c)")

        result = run_pca_at_time(
            sim_result,
            tau,
            feature_indices,
            n_components,
            pca_method_params
        )

        if !isnothing(result)
            push!(pca_results_vector, result)
        end
    end

    println("\nAnaliza PCA zakończona. Przetworzono $(length(pca_results_vector)) pomyślnych kroków czasowych.")
    return pca_results_vector
end

end
