# ANALIZA PCA  
#plik pca.jl 
#
include("lib.jl")
using .modHydroSim
using Plots
using Statistics
using LinearAlgebra

module PCAWorkflow

using ..modHydroSim
using Plots
using Statistics
using LinearAlgebra
using DataFrames
using CSV

export SimSettings, PCAResultAtTime, full_analysis, 
       p_pca, p_explained_variance,
       calc_pca
#stała globalna 
const ALL_FEATURE_NAMES = ["T", "A", "dT_dt", "dA_dt"]

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


# --- SEKCJA 2: STRUKTURY I FUNKCJE ANALIZY ---

"""
    PCAResultAtTime
Struktura do przechowywania wyniku analizy PCA dla pojedynczego punktu w czasie.
"""
struct PCAResultAtTime
    tau::Float64
    transformed_data::Matrix{Float64}
    explained_variance::Vector{Float64}
end

"""
    analyze_pca(sim_result::modHydroSim.SimResult; n_steps=50, n_components=2)

Przeprowadza analizę PCA w `n_steps` punktach czasowych w całym zakresie symulacji.
"""
function calc_pca(
    sim_result::modHydroSim.SimResult;
    features_indices::Union{Tuple,Vector{Int}}, # ZMIANA
    n_steps::Int,
    n_components::Int
)
    t_start, t_end = sim_result.settings.tspan
    sample_times = range(t_start, t_end, length=n_steps)
    pca_results_vector = PCAResultAtTime[]

    for tau in sample_times
        # 1. Pobiera WSZYSTKIE dane (T, A, dT/dt, dA/dt)
        all_data_vectors = modHydroSim.TA(sim_result, tau)
        # 2. tylko te wektory, które odpowiadają indeksom podanym przez użytkownika
        selected_vectors = [all_data_vectors[i] for i in features_indices]
        # 3. połączenie  wybranych wektorów w jedną macierz danych
        data_matrix = hcat(selected_vectors...)

        # 4. PCA na przygotowanej macierzy
        transformed_data, explained_variance = pca_math(data_matrix, n_components)

        result = PCAResultAtTime(tau, transformed_data, explained_variance)
        push!(pca_results_vector, result)
    end

    return pca_results_vector
end
# --- SEKCJA 3: WIZUALIZACJA I GŁÓWNY WORKFLOW ---

"""
    p_explained_variance(pca_results::Vector{PCAResultAtTime})

Tworzy wykres pokazujący, jak wariancja wyjaśniona przez PCA
zmienia się w czasie.  wykres na bazie PRL 125
"""
function p_explained_variance(pca_results::Vector{PCAResultAtTime}; prefix = " ")
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



"""
    p_pca(pca_results::Vector{PCAResultAtTime}, sim_result::modHydroSim.SimResult, target_tau::Float64)

Wyszukuje i wyświetla wykres PCA dla konkretnej chwili czasu `target_tau`.
"""
function p_pca(pca_results::Vector{PCAResultAtTime}, sim_result::modHydroSim.SimResult, target_tau::Float64)
    idx = findmin(res -> abs(res.tau - target_tau), pca_results)[2]
    
    if isnothing(idx)
        available_times = [round(r.tau, digits=3) for r in pca_results]
        error("Nie znaleziono wyników dla tau ≈ $target_tau. Dostępne czasy: $available_times")
    end
    
    result = pca_results[idx]
    initial_temps = [sol.u[1][1] for sol in sim_result.solutions]
    total_explained_var = sum(result.explained_variance) * 100
    
    p = scatter(
        result.transformed_data[:, 1], 
        result.transformed_data[:, 2],
        marker_z=initial_temps,
        title="PCA dla τ = $(round(result.tau, digits=2)) (Wariancja: $(round(total_explained_var, digits=2))%)",
        xlabel="Komponent PCA 1",
        ylabel="Komponent PCA 2",
        label="",
        xlims=(-1.5, 1.5),
        ylims=(-0.55, 0.51),
        markersize=5,
        alpha=0.8,
        color=:plasma,
        colorbar_title="  Temp. początkowa [MeV]"
    )
    display(p)
    return p
end

##################FUNKCJA DO ZAPISU DANYCH W CSV#############################

"""
    save_pca_dataset(sim_result::modHydroSim.SimResult, pca_results::Vector{PCAResultAtTime}, filename_prefix::String="pca_dataset")

Zapisuje dane: τ, T, A, PCA1, PCA2, ... dla każdej trajektorii i chwili czasu w pliku CSV.
Nazwa pliku zalezy od zakrecu parametrów T, A i τ.
"""
function save_pca_dataset(
    sim_result::modHydroSim.SimResult,
    pca_results::Vector{PCAResultAtTime},
    features_indices::Union{Tuple,Vector{Int}};
    filename_prefix::String="pca_data" # Opcjonalny prefiks
)
    # --- Krok 1: Zbiera wszystkie potrzebne informacje ---
    settings = sim_result.settings
    selected_feature_names = ALL_FEATURE_NAMES[collect(features_indices)]
    n_components = length(pca_results[1].explained_variance)    

    Tmin, Tmax = settings.T_range
    Amin, Amax = settings.A_range
    t_start, t_end = settings.tspan

    df_columns = vcat([:id, :tau], Symbol.(selected_feature_names), [Symbol("PCA$(k)") for k in 1:n_components])
    df_types = vcat(Int, Float64, fill(Float64, length(selected_feature_names)), fill(Float64, n_components))
    df_all = DataFrame([T[] for T in df_types], df_columns)

# wypełnianie danymi 
    all_data_at_time = [modHydroSim.TA(sim_result, res.tau) for res in pca_results]
    
    for traj_idx in 1:length(sim_result.solutions)
        for (pca_idx, res) in enumerate(pca_results)
      
            all_vectors = all_data_at_time[pca_idx]
            
       
            original_data_row = [all_vectors[feat_idx][traj_idx] for feat_idx in features_indices]
            
       
            pca_data_row = res.transformed_data[traj_idx, :]
            
   
            row_to_push = vcat(traj_idx, res.tau, original_data_row, pca_data_row)
            push!(df_all, row_to_push)
        end
    end
    

    file = "$(filename_prefix)_" * # 1. Prefiks
           "features-$(join(selected_feature_names, "-"))_" * # 2. Analizowane cechy
           "T-$(round(Tmin, digits=1))-$(round(Tmax, digits=1))_" * # 3. Zakres T
           "A-$(round(Amin, digits=1))-$(round(Amax, digits=1))_" * # 4. Zakres A
           "tau-$(round(t_start, digits=1))-$(round(t_end, digits=1)).csv" # 5. Zakres tau
    
    # --- Krok 5: Zapisz plik ---
    CSV.write(file, df_all)
    println("📁 Zapisano dane do pliku: $file")
end
"""
    full_analysis(; settings=SimSettings(), n_pca_steps=50)

Uruchamia cały workflow: symulację, analizę PCA, wyświetla kluczowe wyniki
i zwraca artefakty do dalszej analizy.
"""
function full_analysis(;
    settings::SimSettings,
    features_indices::Union{Tuple,Vector{Int}} = (1, 2), # <-- Prosty interfejs z domyślną wartością
    n_pca_steps=50,
    n_components=2
)
    # Walidacja wejścia
    if any(i -> i < 1 || i > 4, features_indices)
        error("Indeksy cech muszą być w zakresie od 1 do 4. Otrzymano: $features_indices")
    end
    selected_names = ALL_FEATURE_NAMES[collect(features_indices)]

    println("--- Krok 1: Uruchamianie symulacji hydro... ---")
    sim_result = modHydroSim.run_simulation(settings=settings)
    println("Symulacja zakończona.")

    println("\n--- Krok 2: Przeprowadzanie analizy PCA dla cech: $(join(selected_names, ", "))... ---")
    pca_results = calc_pca(sim_result, features_indices=features_indices, n_steps=n_pca_steps, n_components=n_components)
    println("Analiza PCA zakończona.")

    println("\n--- Krok 3: Generowanie wizualizacji... ---")
    prefix_str = join(selected_names, ", ") # Tworzy string, np. "T, A, dT_dt"
    variance_plot = p_explained_variance(pca_results, prefix=prefix_str)
    display(variance_plot)

    println("\n--- Krok 4: Zapisywanie wyników do pliku CSV... ---")
    save_pca_dataset(sim_result, pca_results, features_indices)

    println("\n✅ Analiza zakończona pomyślnie.")
    return sim_result, pca_results
end



end # koniec modułu PCAWorkflow
