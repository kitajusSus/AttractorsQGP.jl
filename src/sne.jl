
include("lib.jl")
using .modHydroSim

using Plots
using Statistics
using DataFrames
using CSV
using Printf
using Random
using LinearAlgebra

module ManualTSNE

using LinearAlgebra, Random, Printf, Statistics

export run_tsne


"""
Oblicza macierz kwadratów odległości Euklidesowych.
"""
function calculate_squared_distances(X::Matrix{Float64})
    n_points = size(X, 1)
    dist_sq = zeros(Float64, n_points, n_points)
    sum_X = sum(X .^ 2, dims=2)
    dist_sq = -2 * (X * X') .+ sum_X .+ sum_X'
    # Zapewnienie, że odległości nie są ujemne z powodu błędów numerycznych
    return max.(dist_sq, 0.0)
end

"""
Dla danego `σ` i wektora odległości, oblicza rozkład prawdopodobieństwa P i jego entropię.
"""
function calculate_p_and_entropy(dist_row_sq::Vector{Float64}, sigma::Float64, point_idx::Int)
    p = exp.(-dist_row_sq / (2 * sigma^2))
    p[point_idx] = 0.0
    sum_p = sum(p)
    if sum_p == 0.0
        return fill(1e-12, length(p)), 0.0
    end
    p_normalized = p ./ sum_p
    entropy = -sum(p_normalized .* log.(max.(p_normalized, 1e-12)))
    return p_normalized, entropy
end

"""
Przeprowadza wyszukiwanie binarne, aby znaleźć `σ`, które daje zadaną perpleksję.
"""
function find_sigma_for_perplexity(dist_row_sq::Vector{Float64}, perplexity::Float64, point_idx::Int)
    target_entropy = log(perplexity)
    sigma_min, sigma_max = 0.0, Inf
    sigma_val = 1.0

    for _ in 1:50 # 50 iteracji wyszukiwania binarnego
        (_, current_entropy) = calculate_p_and_entropy(dist_row_sq, sigma_val, point_idx)
        if current_entropy < target_entropy
            sigma_min = sigma_val
            sigma_val = isinf(sigma_max) ? sigma_val * 2 : (sigma_val + sigma_max) / 2.0
        else
            sigma_max = sigma_val
            sigma_val = (sigma_val + sigma_min) / 2.0
        end
    end
    return sigma_val
end

"""
Oblicza macierz P symetrycznych prawdopodobieństw w przestrzeni wysokowymiarowej.
"""
function calculate_P_matrix(X::Matrix{Float64}, perplexity::Float64)
    (n_points, _) = size(X)
    dist_sq = calculate_squared_distances(X)
    P_conditional = zeros(Float64, n_points, n_points)

    println("Obliczanie macierzy P (prawdopodobieństw wysokowymiarowych)...")
    for i in 1:n_points
        print("\rPrzetwarzanie punktu $i/$n_points...")
        sigma_i = find_sigma_for_perplexity(dist_sq[i, :], perplexity, i)
        (p_row, _) = calculate_p_and_entropy(dist_sq[i, :], sigma_i, i)
        P_conditional[i, :] = p_row
    end
    println("\nSymetryzacja macierzy P...")
    # Wzór na symetryczne p_ij = (p_j|i + p_i|j) / 2n
    P = (P_conditional + P_conditional') / (2 * n_points)
    return max.(P, 1e-12) # Unikamy zerowych wartości dla stabilności
end

# --- Krok 2 i 3: Optymalizacja w przestrzeni niskowymiarowej ---

"""
Główna funkcja uruchamiająca algorytm t-SNE.
"""
function run_tsne(X::Matrix{Float64}; perplexity=30.0, max_iter=1000, seed=123,
                  learning_rate=500.0, early_exaggeration=12.0,
                  initial_momentum=0.5, final_momentum=0.8,
                  stop_exaggeration_iter=250)

    (n_points, _) = size(X)
    Random.seed!(seed)

    P = calculate_P_matrix(X, perplexity)

    Y = randn(n_points, 2) * 0.0001
    dY = zeros(size(Y)) # Gradient
    iY = zeros(size(Y)) # Aktualizacja z pędem

    println("Rozpoczynanie optymalizacji t-SNE...")
    for iter in 1:max_iter
        sum_Y = sum(Y .^ 2, dims=2)
        dist_Y_sq = -2 * (Y * Y') .+ sum_Y .+ sum_Y'

        # Wzór na q_ij (bez normalizacji)
        q_unnormalized_inv = 1.0 ./ (1.0 .+ dist_Y_sq)
        q_unnormalized_inv[diagind(q_unnormalized_inv)] .= 0.0

        # Znormalizowana macierz Q
        Q = q_unnormalized_inv ./ sum(q_unnormalized_inv)
        Q = max.(Q, 1e-12)

        current_momentum = iter <= stop_exaggeration_iter ? initial_momentum : final_momentum
        P_eff = iter <= stop_exaggeration_iter ? P * early_exaggeration : P

        # Wzór: dC/dy_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^-1
        PQ_diff = P_eff - Q
        for i in 1:n_points
            dY[i, :] = sum(4 * (PQ_diff[:, i] .* q_unnormalized_inv[:, i]) .* (Y[i, :]' .- Y), dims=1)
        end
        iY = current_momentum .* iY - learning_rate .* dY
        Y .+= iY

        Y .-= mean(Y, dims=1)

        if iter % 50 == 0
            cost = sum(P_eff .* log.(P_eff ./ Q))
            @printf("Iteracja %4d/%d: Koszt KL = %.4f\n", iter, max_iter, cost)
        end
    end

    println("Optymalizacja t-SNE zakończona.")
    return Y
end

end # koniec modułu ManualTSNE


module DimRedAnalysis

using ..modHydroSim
using ..ManualTSNE
using Plots
using Statistics
using DataFrames
using CSV
using Printf
using Random

export SimSettings, full_analysis, prepare_hydro_data, p_embedding, save_embedding_dataset

function prepare_hydro_data(sim_settings::SimSettings, n_time_samples::Int)
    println("--- Krok 1: Generowanie danych z symulacji hydrodynamiki... ---")
    sim_result = run_simulation(settings=sim_settings)
    n_trajectories = length(sim_result.solutions)

    t_start, t_end = sim_result.settings.tspan
    sample_times = range(t_start, stop=t_end, length=n_time_samples)

    D = n_time_samples * 2
    X = zeros(Float64, n_trajectories, D)
    initial_temps = zeros(Float64, n_trajectories)

    println("--- Krok 2: Przekształcanie trajektorii w wektory wysokowymiarowe... ---")
    for i in 1:n_trajectories
        sol = sim_result.solutions[i]
        sampled_points = sol(sample_times)
        flat_vector = vcat([[p[1], p[2]] for p in sampled_points]...)
        X[i, :] = flat_vector
        initial_temps[i] = sol.u[1][1]
    end

    @printf("Przygotowanie danych zakończone. Wymiar macierzy X: %d trajektorii x %d cech.\n", n_trajectories, D)
    return X, initial_temps, sim_result
end

function p_embedding(Y::Matrix{Float64}, labels::Vector{Float64}; title_str="Wynik redukcji wymiarowości")
    println("\n--- Krok 4: Generowanie wykresu wynikowego... ---")
    p = scatter(
        Y[:, 1], Y[:, 2],
        marker_z=labels,
        title=title_str,
        xlabel="Komponent t-SNE 1",
        ylabel="Komponent t-SNE 2",
        label="",
        markersize=5,
        alpha=0.8,
        color=:plasma,
        colorbar_title="  Temp. początkowa [MeV]"
    )
    display(p)
    println("Gotowe. Wyświetlono wykres.")
    return p
end

function save_embedding_dataset(sim_result::SimResult, Y::Matrix{Float64}, initial_temps::Vector{Float64}, filename_prefix::String)
    println("\n--- Krok 5: Zapisywanie wyników do pliku CSV... ---")
    settings = sim_result.settings
    Tmin, Tmax = settings.T_range
    Amin, Amax = settings.A_range
    t_start, t_end = settings.tspan

    df = DataFrame(
        trajectory_id = 1:length(sim_result.solutions),
        initial_T = initial_temps,
        initial_A = [sol.u[1][2] for sol in sim_result.solutions],
        embedding_x = Y[:, 1],
        embedding_y = Y[:, 2]
    )

    file = "$(filename_prefix)_" *
           "T-$(round(Tmin, digits=1))-$(round(Tmax, digits=1))_" *
           "A-$(round(Amin, digits=1))-$(round(Amax, digits=1))_" *
           "tau-$(round(t_start, digits=1))-$(round(t_end, digits=1)).csv"

    CSV.write(file, df)
    println("📁 Zapisano dane do pliku: $file")
end

function full_analysis(;
    hydro_settings::SimSettings = SimSettings(T_range = (300,1200),A_range=(-13,13),n_points=250, tspan=(0.2,1.2)),
    n_time_samples::Int = 10,
    perplexity::Float64 = 15.0,
    max_iter::Int = 1000,
    seed::Int = 5,
    save_results::Bool = true
)
    X, initial_temps, sim_result = prepare_hydro_data(hydro_settings, n_time_samples)


    println("\n--- Krok 3: Uruchamianie  t-SNE... ---")
    Y = ManualTSNE.run_tsne(X, perplexity=perplexity, max_iter=max_iter, seed=seed)

    title = "Wizualizacja t-SNE trajektorii (Perplexity: $(perplexity))"
    p = p_embedding(Y, initial_temps, title_str=title)

    if save_results
        filename_prefix = "tsne_embedding_manual"
        save_embedding_dataset(sim_result, Y, initial_temps, filename_prefix)
    end

    println("\n✅ Analiza zakończona pomyślnie.")
    return (sim_result=sim_result, X=X, Y=Y, plot=p)
end

end
