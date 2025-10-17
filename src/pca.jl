# Dołączenie modułu symulacji jest niezbędne
include("lib.jl")

# --- Początek modułu PCAWorkflow ---
module PCAWorkflow

# --- SEKCJA 1: Importowanie zależności ---
using ..modHydroSim
using Plots
using Statistics
using LinearAlgebra
using DataFrames
using CSV
using HDF5
using Dates

export run_pca_workflow, calc_pca, pca_snapshot_plot, visualize_pca_from_file


"""
    PCAResultAtTime
Przechowuje wyniki analizy PCA dla pojedynczego kroku czasowego `tau`.
"""
struct PCAResultAtTime
    tau::Float64
    transformed_data::Matrix{Float64}
    explained_variance::Vector{Float64}
    principal_components::Matrix{Float64}
end


function pca_math(X::Matrix{Float64}, n_components::Int)
    n_samples, n_features = size(X)
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

    return transformed_X, explained_variance_ratio, projection_matrix
end

"""
    calc_pca(sim_result::modHydroSim.SimResult; ...)
Orkiestruje procesem analizy PCA dla całego zestawu symulacji w wielu krokach czasowych.
"""
function calc_pca(
    sim_result::modHydroSim.SimResult;
    feature_indices::Vector{Int},
    n_pca_steps::Int,
    n_components::Int
)
    println("\n--- Krok 3: Uruchamianie analizy PCA... ---")
    println("Liczba kroków czasowych PCA: $n_pca_steps")
    println("Liczba głównych komponentów: $n_components")

    t_start, t_end = sim_result.settings.tspan
    sample_times = range(t_start, stop=t_end, length=n_pca_steps)
    pca_results_vector = PCAResultAtTime[]

    # Pętla po każdym wybranym kroku czasowym
    for (i, tau) in enumerate(sample_times)
        print("\rPrzetwarzanie kroku czasowego: $i/$n_pca_steps (τ = $(round(tau, digits=2)) fm/c)")

        # Pobranie "migawki" stanu systemu dla wszystkich symulacji w czasie tau
        all_data_vectors = modHydroSim.TA(sim_result, tau)

        # Wybór cech wskazanych przez użytkownika
        selected_vectors = [all_data_vectors[i] for i in feature_indices]
        data_matrix = hcat(selected_vectors...) # Stworzenie macierzy (n_symulacji x p_cech)

        # Usunięcie wierszy z wartościami NaN lub Inf, które mogły powstać w symulacji
        valid_rows = all(isfinite, data_matrix, dims=2)
        if count(valid_rows) < 2
            println("\nOstrzeżenie: Zbyt mało prawidłowych danych w czasie τ=$tau. Pomijanie kroku.")
            continue
        end

        # Wykonanie obliczeń PCA
        transformed_data, explained_variance, principal_components = pca_math(data_matrix[vec(valid_rows), :], n_components)

        # Zapisanie wyników
        push!(pca_results_vector, PCAResultAtTime(tau, transformed_data, explained_variance, principal_components))
    end
    println("\nAnaliza PCA zakończona. Przetworzono $(length(pca_results_vector)) kroków czasowych.")
    return pca_results_vector
end


# --- SEKCJA 4: Interakcja z użytkownikiem ---

"""
    prompt_for_features(sim_result::modHydroSim.SimResult)
Wyświetla dostępne cechy i prosi użytkownika o wybór tych do analizy.
"""
function prompt_for_features(sim_result::modHydroSim.SimResult)
    # Definicja dostępnych cech na podstawie teorii
    if sim_result.settings.theory == :HJSW
        all_feature_names = ["T", "A", "Z", "dT/dτ", "dA/dτ", "dZ/dτ"]
    else # BRSSS lub MIS
        all_feature_names = ["T", "A", "dT/dτ", "dA/dτ"]
    end

    println("\n Dostępne cechy do analizy PCA:")
    for (i, name) in enumerate(all_feature_names)
        println("  [$i] $name")
    end

    while true
        print("Wybierz indeksy cech do analizy (oddzielone przecinkami, np. 1,2,4): ")
        input = readline()
        try
            indices = [parse(Int, s) for s in split(input, ',')]
            if all(i -> 1 <= i <= length(all_feature_names), indices) && !isempty(indices)
                selected_names = all_feature_names[indices]
                println("Wybrano cechy: $(join(selected_names, ", "))")
                return indices, selected_names
            else
                println("Błąd: Podaj prawidłowe indeksy z zakresu 1-$(length(all_feature_names)).")
            end
        catch e
            println("Błąd: Nieprawidłowy format. Wprowadź liczby oddzielone przecinkami.")
        end
    end
end


# --- SEKCJA 5: Zapisywanie wyników ---

"""
    generate_output_filename_base(...)
Tworzy opisową, unikalną nazwę bazową dla plików wyjściowych.
"""
function generate_output_filename_base(ic_filepath::String, theory::Symbol, selected_features::Vector{String})
    ic_name = splitext(basename(ic_filepath))[1]
    features_str = join(selected_features, "-")
    timestamp = Dates.format(now(), "YYYYmmdd_HHMMSS")

    return "pca_results_$(ic_name)_theory_$(theory)_features_$(features_str)_t_$(timestamp)"
end

function save_pca_results(
    filename_base::String,
    pca_results::Vector{PCAResultAtTime},
    sim_result::modHydroSim.SimResult,
    selected_feature_names::Vector{String},
    ic_filepath::String  # <--- DODANY ARGUMENT
)
    println("\n--- Krok 5: Zapisywanie wyników... ---")

    # --- Zapis do pliku CSV ---
    csv_path = filename_base * ".csv"
    df_rows = []
    initial_states = collect(hcat([sol.u[1] for sol in sim_result.solutions]...)')

    for result in pca_results
        tau = result.tau
        for i in 1:size(result.transformed_data, 1)
            row = (
                tau=tau,
                simulation_id=i,
                T_0=initial_states[i, 1],
                A_0=initial_states[i, 2],
            )
            if size(initial_states, 2) > 2
                row = merge(row, (Z_0=initial_states[i, 3],))
            end
            for j in 1:size(result.transformed_data, 2)
                row = merge(row, (Symbol("PC$j") => result.transformed_data[i, j],))
            end
            for j in 1:length(result.explained_variance)
                row = merge(row, (Symbol("ExplainedVariance_$j") => result.explained_variance[j],))
            end
            push!(df_rows, row)
        end
    end

    df = DataFrame(df_rows)
    CSV.write(csv_path, df)
    println("📁 Zapisano wyniki w formacie CSV do: $csv_path")

    # --- Zapis do pliku HDF5 ---
    h5_path = filename_base * ".h5"
    h5open(h5_path, "w") do file
        # Metadane
        attrs(file)["description"] = "Wyniki analizy PCA dla symulacji hydrodynamicznej."
        # POPRAWKA: Używamy przekazanej nazwy pliku, a nie pola .seed
        attrs(file)["source_ic_file"] = basename(ic_filepath)
        attrs(file)["theory"] = string(sim_result.settings.theory)
        attrs(file)["selected_features"] = join(selected_feature_names, ", ")
        attrs(file)["timestamp"] = string(now())

        # Dane
        g = create_group(file, "pca_results")
        g["tau"] = [res.tau for res in pca_results]
        transformed_data_3d = permutedims(cat([res.transformed_data for res in pca_results]..., dims=3), (3, 1, 2))
        g["transformed_data"] = transformed_data_3d

        explained_variance_matrix = collect(hcat([res.explained_variance for res in pca_results]...)')
        g["explained_variance"] = explained_variance_matrix

        principal_components_3d = permutedims(cat([res.principal_components for res in pca_results]..., dims=3), (3, 1, 2))
        g_pc = create_group(g, "principal_components")
        g_pc["axes"] = principal_components_3d
        attrs(g_pc)["description"] = "Osie głównych komponentów (wektory własne) dla każdego kroku czasowego. Wymiary: (czas, cechy, komponenty)."
    end
    println("🗄️ Zapisano wyniki w formacie HDF5 do: $h5_path")
end

"""
    p_explained_variance(pca_results, prefix)
Tworzy wykres wariancji wyjaśnionej przez komponenty w funkcji czasu.
"""
function p_explained_variance(pca_results::Vector{PCAResultAtTime}; prefix=" ")
    times = [res.tau for res in pca_results]
    n_components = length(pca_results[1].explained_variance)

    p = plot(title="Wariancja wyjaśniona przez komponenty PCA\n[Cechy: $(prefix)]",
        xlabel="Czas τ [fm/c]", ylabel="Proporcja wyjaśnionej wariancji",
        legend=:best)

    # Rysuj wariancję dla każdego komponentu
    for i in 1:n_components
        variance_data = [res.explained_variance[i] for res in pca_results]
        plot!(p, times, variance_data, label="Komponent $i", linewidth=2)

    end

    # Rysuj skumulowaną wariancję
    if n_components > 1
        cumulative_variance = [sum(res.explained_variance) for res in pca_results]
        plot!(p, times, cumulative_variance, label="Suma", linestyle=:dash, color=:black, linewidth=2.5)
    end

    hline!(p, [1.0], linestyle=:dot, color=:grey, label="", alpha=0.7)
    return p
end

"""
    p_pca_snapshot(pca_results, sim_result, target_tau)
Tworzy wykres rozrzutu danych po transformacji PCA dla jednego punktu w czasie.
"""
function p_pca_snapshot(pca_results::Vector{PCAResultAtTime}, sim_result::modHydroSim.SimResult, target_tau::Float64)
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
        color=:plasma, colorbar_title="  Temp. początkowa [MeV]",
        xlims=(-0.31, 0.45),
        ylims=(-0.005, 0.0045)
    )
    return p
end


function run_pca_workflow(
    ic_filepath::String;
    theory::Symbol,
    tspan::Tuple{Float64,Float64}=(0.22, 2.0),
    n_pca_steps::Int=50,
    n_components::Int=2
)
    println("="^60)
    println(" Rozpoczynanie przepływu pracy analizy PCA")
    println(" Plik z warunkami początkowymi: $ic_filepath")
    println(" Teoria: $theory")
    println("="^60)

    # --- Krok 1: Uruchomienie symulacji na podstawie pliku z warunkami początkowymi ---
    println("\n--- Krok 1: Uruchamianie symulacji hydro... ---")
    settings = modHydroSim.SimSettings(theory=theory, tspan=tspan)
    sim_result = modHydroSim.run_simulation(settings=settings, ic_file=ic_filepath)
    if isempty(sim_result.solutions)
        println("Błąd krytyczny: Symulacja nie zwróciła żadnych wyników. Przerwanie analizy.")
        return
    end

    # --- Krok 2: Interaktywny wybór cech do analizy ---
    feature_indices, selected_feature_names = prompt_for_features(sim_result)

    # --- Krok 3: Przeprowadzenie analizy PCA ---
    pca_results = calc_pca(sim_result, feature_indices=feature_indices, n_pca_steps=n_pca_steps, n_components=n_components)
    if isempty(pca_results)
        println("Błąd krytyczny: Analiza PCA nie zwróciła żadnych wyników. Przerwanie analizy.")
        return
    end

    # --- Krok 4: Generowanie wizualizacji ---
    println("\n--- Krok 4: Generowanie wizualizacji... ---")
    variance_plot = p_explained_variance(pca_results, prefix=join(selected_feature_names, ", "))
    display(variance_plot)


    mid_time = tspan[1] + (tspan[2] - tspan[1]) / 2
    pca_1 = p_pca_snapshot(pca_results, sim_result, mid_time / 2)
    display(pca_1)
    pca_2 = p_pca_snapshot(pca_results, sim_result, mid_time * 1.2)
    display(pca_2)
    pca_snapshot_plot = p_pca_snapshot(pca_results, sim_result, mid_time)
    display(pca_snapshot_plot)

    filename_base = generate_output_filename_base(ic_filepath, theory, selected_feature_names)
    save_pca_results(filename_base, pca_results, sim_result, selected_feature_names, ic_filepath)

    println("\n✅ Analiza zakończona pomyślnie.")
    return sim_result, pca_results
end




function visualize_pca_from_file(filepath::String; color_by::Symbol=:T_0)
    println("--- Wczytywanie wyników PCA z pliku: $filepath ---")

    local df::DataFrame
    local metadata = Dict()

    # --- Krok 1: Wczytanie danych w zależności od typu pliku ---
    if endswith(lowercase(filepath), ".csv")
        df = CSV.read(filepath, DataFrame)
        println("Pomyślnie wczytano dane z pliku CSV.")
        metadata["theory"] = "N/A (z CSV)"
        metadata["selected_features"] = "N/A (z CSV)"

    elseif endswith(lowercase(filepath), ".h5")
        h5open(filepath, "r") do file
            # Wczytanie metadanych
            metadata["theory"] = read(attrs(file)["theory"])
            metadata["selected_features"] = read(attrs(file)["selected_features"])

            # Wczytanie danych
            taus = read(file["data/tau"])
            transformed_data = read(file["data/transformed_data"]) # Wymiary: (czas, symulacja, komponent)
            initial_conditions = read(file["initial_conditions/values"])
            ic_names = read(attrs(file["initial_conditions"])["column_names"])

            # Złożenie danych w jeden DataFrame (analogicznie do formatu CSV)
            rows = []
            for (i, tau) in enumerate(taus)
                for j in 1:size(transformed_data, 2)
                    row = (tau=tau, simulation_id=j)
                    # Dodanie warunków początkowych
                    for (k, name) in enumerate(ic_names)
                        row = merge(row, (Symbol(name) => initial_conditions[j, k],))
                    end
                    # Dodanie komponentów PCA
                    for k in 1:size(transformed_data, 3)
                        row = merge(row, (Symbol("PC$k") => transformed_data[i, j, k],))
                    end
                    push!(rows, row)
                end
            end
            df = DataFrame(rows)
        end
        println("Pomyślnie wczytano dane z pliku HDF5.")
    else
        error("Nieobsługiwany format pliku: '$filepath'. Użyj .csv lub .h5.")
    end

    if !hasproperty(df, color_by)
        println("Ostrzeżenie: Wybrana kolumna do kolorowania ':$color_by' nie istnieje. Używam :T_0.")
        color_by = :T_0
        if !hasproperty(df, :T_0)
            error("Brak kolumny ':T_0' w danych. Nie można pokolorować wykresu.")
        end
    end

    # --- Krok 2: Interaktywny wybór kroku czasowego ---
    unique_taus = sort(unique(df.tau))
    println("\nDostępne kroki czasowe (τ) w pliku:")
    for (i, tau) in enumerate(unique_taus)
        println("  [$i] $(round(tau, digits=3))")
    end

    local selected_tau
    while true
        print("Wybierz indeks kroku czasowego do wizualizacji: ")
        input = readline()
        try
            idx = parse(Int, input)
            if 1 <= idx <= length(unique_taus)
                selected_tau = unique_taus[idx]
                break
            else
                println("Błąd: Podaj indeks z zakresu 1-$(length(unique_taus)).")
            end
        catch
            println("Błąd: Nieprawidłowy format. Wprowadź liczbę.")
        end
    end

    # --- Krok 3: Filtrowanie danych i generowanie wykresu ---
    snapshot_df = filter(row -> row.tau == selected_tau, df)

    p = scatter(
        snapshot_df.PC1,
        snapshot_df.PC2,
        marker_z=snapshot_df[!, color_by],
        title="PCA dla τ = $(round(selected_tau, digits=2))\nTeoria: $(metadata["theory"]), Cechy: $(metadata["selected_features"])",
        xlabel="Komponent PCA 1", ylabel="Komponent PCA 2",
        label="", markersize=5, alpha=0.8,
        color=:plasma, colorbar_title="  $(string(color_by))"
    )
    display(p)
    println("\n✅ Wykres został wygenerowany.")
    return p
end
end # Koniec modułu PCAWorkflow
