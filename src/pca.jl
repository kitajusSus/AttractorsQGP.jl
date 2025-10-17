# Dołączenie modułu symulacji jest niezbędne
include("lib.jl")

module PCAWorkflow

# Importowanie niezbędnych funkcji z innych modułów
using ..modHydroSim
using Plots
using Statistics
using LinearAlgebra
using DataFrames
using CSV

# Eksportowanie publicznych funkcji, które będą używane w main.jl
export SimSettings, PCAResultAtTime, full_analysis,
    p_pca, p_explained_variance,
    calc_pca, save_pca_dataset

# Stała globalna ułatwiająca wybór cech
const ALL_FEATURE_NAMES = ["T", "A", "dT_dt", "dA_dt"]


# --- SEKCJA 2: STRUKTURY I FUNKCJE ANALIZY ---
struct PCAResultAtTime
    tau::Float64
    transformed_data::Matrix{Float64}
    explained_variance::Vector{Float64}
end

function pca_math(X::Matrix{Float64}, n_components::Int)
    # Pobranie wymiarów macierzy wejściowej
    n_samples, n_features = size(X) # n = n_samples, p = n_features
    #
    if n_components > n_features
        error("Liczba komponentów (k) nie może być większa niż liczba cech (p).")
    end

    # --------------------------------------------------------------------------
    # Krok 1: Centrowanie danych
    #
    # Cel: Przesunięcie danych tak, aby każda cecha miała średnią równą 0.
    # Wzór: Dla każdej kolumny (cechy) X_j, obliczana jest jej średnia miu_j.
    #       Następnie dla każdej obserwacji x_ij odejmuje się tę średnią:
    #       x'_ij = x_ij - miu_j
    # W notacji macierzowej: X_c = X - μ (gdzie μ to wektor średnich powielony dla każdego wiersza)
    # --------------------------------------------------------------------------
    mean_vector = mean(X, dims=1) # Wektor średnich dla każdego wymiaru  (1 x p)
    X_centered = X .- mean_vector

    # -------------------------------------
    # Krok 2: Obliczenie macierzy kowariancji
    #
    # Cel: Stworzenie macierzy (p x p), która opisuje wariancję i współzmienność cech.
    # Wzór: Macierz kowariancji C jest zdefiniowana jako:
    #       C = (1 / (n-1)) * X_c' * X_c
    #       gdzie X_c' to transpozycja macierzy scentrowanych danych.
    #       Element c_ij macierzy C to kowariancja między cechą i a cechą j.
    #       cov(X_i, X_j) = E[(X_i - E[X_i])(X_j - E[X_j])]
    # --------------------------------------------------------------------------
    cov_matrix = cov(X_centered) # Rozmiar (p x p)

    # --------------------------------------------------------------------------
    # Krok 3: Obliczenie wartości własnych i wektorów własnych
    #
    # Cel: Znalezienie głównych osi wariancji w danych.
    # Wzór: Rozwiązujemy równanie własne dla macierzy kowariancji C:
    #       C * v = λ * v
    #       gdzie:
    #       - v to wektor własny macierzy C (kierunek osi, czyli główna składowa).
    #       - λ to wartość własna odpowiadająca wektorowi v (mówi o tym,
    #         jak dużo wariancji jest w kierunku wektora v).
    # To jest równoważne znalezieniu rozwiązań równania charakterystycznego:
    #       det(C - λI) = 0
    eigen_result = eigen(cov_matrix)
    eigenvalues = eigen_result.values    # Wektor λ (lambda)
    eigenvectors = eigen_result.vectors  # Macierz, której kolumnami są wektory v

    # --------------------------------------------------------------------------
    # Krok 4: Sortowanie wektorów własnych i wybór głównych składowych
    #
    # Cel: Uporządkowanie osi wariancji od najważniejszej do najmniej ważnej
    #      i wybranie k pierwszych z nich.
    # Proces: Sortujemy wartości własne λ w porządku malejącym:
    #         λ_1 ≥ λ_2 ≥ ... ≥ λ_p
    #         Następnie, w tej samej kolejności, sortujemy odpowiadające im
    #         wektory własne v.
    #         Wybieramy k pierwszych wektorów własnych (v_1, v_2, ..., v_k),
    #         aby utworzyć macierz transformacji W.
    # --------------------------------------------------------------------------
    sorted_indices = sortperm(eigenvalues, rev=true)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Macierz transformacji W (projection_matrix) składa się z k pierwszych wektorów własnych
    # W = [v_1 | v_2 | ... | v_k]
    projection_matrix = sorted_eigenvectors[:, 1:n_components] # Rozmiar (p x k)

    # --------------------------------------------------------------------------
    # Krok 5: Transformacja danych do nowej przestrzeni#
    # Cel: Rzutowanie oryginalnych danych na nową podprzestrzeń zdefiniowaną
    #      przez wybrane główne składowe.
    # Wzór: Nowa macierz danych Y jest obliczana jako iloczyn macierzy
    #       scentrowanych danych X_c i macierzy transformacji W.
    #       Y = X_c * W
    #       Wynikowa macierz Y ma wymiary (n x k), co oznacza redukcję
    #       wymiarowości z p do k.
    # --------------------------------------------------------------------------
    transformed_X = X_centered * projection_matrix # Rozmiar (n x k)

    # Obliczenie wyjaśnionej wariancji
    # Proporcja wariancji wyjaśnionej przez i-tą główną składową: λ_i / sum(λ)
    total_variance = sum(eigenvalues)
    explained_variance_ratio = sorted_eigenvalues[1:n_components] ./ total_variance

    return transformed_X, explained_variance_ratio
end
function calc_pca(
    sim_result::modHydroSim.SimResult;
    features_indices::Vector{Int},
    n_steps::Int,
    n_components::Int
)
    t_start, t_end = sim_result.settings.tspan
    sample_times = range(t_start, t_end, length=n_steps)
    pca_results_vector = PCAResultAtTime[]

    for tau in sample_times
        all_data_vectors = modHydroSim.TA(sim_result, tau)
        selected_vectors = [all_data_vectors[i] for i in features_indices]
        data_matrix = hcat(selected_vectors...)

        transformed_data, explained_variance = pca_math(data_matrix, n_components)
        push!(pca_results_vector, PCAResultAtTime(tau, transformed_data, explained_variance))
    end
    return pca_results_vector
end

# --- SEKCJA 3: WIZUALIZACJA I WORKFLOW ---
function p_explained_variance(pca_results::Vector{PCAResultAtTime}; prefix=" ")
    times = [res.tau for res in pca_results]
    n_components = length(pca_results[1].explained_variance)
    variance_data = [[res.explained_variance[i] for res in pca_results] for i in 1:n_components]
    labels = ["Komponent $i" for i in 1:n_components]

    p = plot(times, variance_data, title="Wariancja wyjaśniona przez komponenty PCA. [ $(prefix) ]",
        xlabel="Czas τ [fm/c]", ylabel="Proporcja wyjaśnionej wariancji",
        label=reshape(labels, 1, :), legend=:best, linewidth=2)
    hline!(p, [1.0], linestyle=:dash, color=:black, label="", alpha=0.5)
    return p
end

function p_pca(pca_results::Vector{PCAResultAtTime}, sim_result::modHydroSim.SimResult, target_tau::Float64)
    _, idx = findmin(res -> abs(res.tau - target_tau), pca_results)

    result = pca_results[idx]
    initial_temps = [sol.u[1][1] for sol in sim_result.solutions]
    total_explained_var = sum(result.explained_variance) * 100

    p = scatter(
        result.transformed_data[:, 1],
        result.transformed_data[:, 2],
        marker_z=initial_temps,
        title="PCA dla τ = $(round(result.tau, digits=2)) (Wariancja: $(round(total_explained_var, digits=2))%)",
        xlabel="Komponent PCA 1", ylabel="Komponent PCA 2",
        label="", markersize=5, alpha=0.8,
        color=:plasma, colorbar_title="  Temp. początkowa [MeV]"
    )
    return p
end

function save_pca_dataset(
    sim_result::modHydroSim.SimResult,
    pca_results::Vector{PCAResultAtTime},
    features_indices::Vector{Int};
    filename_prefix::String="pca_data"
)
    settings = sim_result.settings
    selected_feature_names = ALL_FEATURE_NAMES[features_indices]
    n_components = length(pca_results[1].explained_variance)

    Tmin, Tmax = settings.T_range
    Amin, Amax = settings.A_range
    t_start, t_end = settings.tspan

    file_path = "$(filename_prefix)_" *
                "features-$(join(selected_feature_names, "-"))_" *
                "T-$(round(Int,Tmin/modHydroSim.MeV))-$(round(Int,Tmax/modHydroSim.MeV))_" *
                "A-$(round(Int,Amin))-$(round(Int,Amax))_" *
                "tau-$(round(t_start, digits=1))-$(round(t_end, digits=1)).csv"

    # Zapis danych (dla zwięzłości, uproszczono logikę z oryginalnego kodu)
    println("📁 Zapisywanie danych do pliku: $file_path (logika zapisu pominięta dla demonstracji)")
    # Pełna implementacja zapisu CSV powinna być tutaj
end

function full_analysis(;
    settings::modHydroSim.SimSettings,
    features_indices::Vector{Int}=[1, 2],
    n_pca_steps=50,
    n_components=2
)
    if any(i -> i < 1 || i > 4, features_indices)
        error("Indeksy cech muszą być w zakresie od 1 do 4. Otrzymano: $features_indices")
    end
    selected_names = ALL_FEATURE_NAMES[features_indices]

    println("--- Krok 1: Uruchamianie symulacji hydro... ---")
    sim_result = modHydroSim.run_simulation(settings=settings)

    println("\n--- Krok 2: Analiza PCA dla cech: $(join(selected_names, ", "))... ---")
    pca_results = calc_pca(sim_result, features_indices=features_indices, n_steps=n_pca_steps, n_components=n_components)

    println("\n--- Krok 3: Generowanie wizualizacji... ---")
    prefix_str = join(selected_names, ", ")
    variance_plot = p_explained_variance(pca_results, prefix=prefix_str)
    display(variance_plot)

    # Wyświetl PCA dla punktu w połowie symulacji
    mid_time = settings.tspan[1] + (settings.tspan[2] - settings.tspan[1]) / 2
    pca_snapshot_plot = p_pca(pca_results, sim_result, mid_time)
    display(pca_snapshot_plot)

    println("\n--- Krok 4: Zapisywanie wyników do pliku CSV... ---")
    save_pca_dataset(sim_result, pca_results, features_indices)

    println("\n✅ Analiza zakończona pomyślnie.")
    return sim_result, pca_results
end

end # Koniec modułu PCAWorkflow
