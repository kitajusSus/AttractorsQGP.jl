include("lib.jl")
include("pca.jl")
include("plt.jl")

using .modHydroSim
using .modPCA
using .modPlots
using GLMakie
using Dates
using DataFrames
using CSV
using HDF5

export run_from_file, main
const ALL_FEATURE_NAMES = ["T", "A", "dTdτ", "dAdτ"]

function prompt_for_features()
    println("\n Dostępne cechy do analizy PCA:")
    for (i, name) in enumerate(ALL_FEATURE_NAMES)
        println("  [$i] $name")
    end

    while true
        print("Wybierz indeksy cech (oddzielone przecinkami, np. 1,3): ")
        input = readline()
        try
            # Zezwól na indeksy 1-based, ale funkcja PCA oczekuje 0-based
            # Funkcja run_pca_over_time oczekuje indeksów 1-based [1,2,3,4]
            indices = [parse(Int, s) for s in split(input, ',')]
            if all(i -> 1 <= i <= length(ALL_FEATURE_NAMES), indices) && !isempty(indices)
                selected_names = ALL_FEATURE_NAMES[indices]
                println("Wybrano cechy: $(join(selected_names, ", "))")
                return indices, selected_names
            else
                println("Błąd: Podaj prawidłowe indeksy z zakresu 1-$(length(ALL_FEATURE_NAMES)).")
            end
        catch e
            println("Błąd: Nieprawidłowy format. Wprowadź liczby oddzielone przecinkami.")
        end
    end
end

function prompt_for_pca_settings()
    println("\n--- Wybór metody PCA ---")
    methods = [:standardize, :center, :minmax, :none, :kernel]
    descriptions = [
        "Liniowa PCA ze standaryzacją (średnia=0, odch. std.=1) - ZALECANE",
        "Liniowa PCA z centrowaniem (odjęcie średniej)",
        "Liniowa PCA z normalizacją Min-Max [0, 1]",
        "Liniowa PCA bez skalowania (surowe dane)",
        "Kernel PCA (kPCA) z jądrem RBF (do nieliniowych zależności)",
    ]

    for (i, desc) in enumerate(descriptions)
        println("  [$i] $(methods[i]) - $desc")
    end

    while true
        print("Wybierz indeks metody (np. 1): ")
        input = readline()
        try
            idx = parse(Int, input)
            if 1 <= idx <= length(methods)
                selected_mode = methods[idx]
                println("Wybrano metodę: $selected_mode")
                return selected_mode
            else
                println("Błąd: Podaj indeks z zakresu 1-$(length(methods)).")
            end
        catch e
            println("Błąd: Nieprawidłowy format. Wprowadź jedną liczbę.")
        end
    end
end

function prompt_for_kernel_parameters()
    println("\n--- Konfiguracja Kernel PCA (kPCA) ---")
    println("kPCA używa jądra RBF: k(x,y) = exp(-gamma * ||x-y||^2)")
    gamma_default = 0.1

    while true
        print("Podaj wartość parametru gamma [domyślnie: $gamma_default]: ")
        input = readline()
        if isempty(input)
            println("Użyto domyślnej wartości gamma = $gamma_default")
            return gamma_default
        end
        try
            gamma = parse(Float64, input)
            if gamma > 0
                println("Ustawiono gamma = $gamma")
                return gamma
            else
                println("Błąd: gamma musi być wartością dodatnią.")
            end
        catch e
            println("Błąd: Nieprawidłowy format. Wprowadź liczbę.")
        end
    end
end

function prompt_for_plot_count()
    println("\n--- Konfiguracja wizualizacji ---")
    while true
        print("Ile statycznych wykresów PCA chcesz wygenerować? (np. 6): ")
        input = readline()
        try
            count = parse(Int, input)
            if count > 0
                println("Zostanie wygenerowanych $count wykresów w równych odstępach czasu.")
                return count
            else
                println("Błąd: Liczba wykresów musi być dodatnia.")
            end
        catch e
            println("Błąd: Nieprawidłowy format. Wprowadź liczbę całkowitą.")
        end
    end
end

# ==============================================================================
# NOWA FUNKCJA (ZGODNIE Z PROŚBĄ)
# ==============================================================================
"""
Przeszukuje podany folder w poszukiwaniu plików .csv,
wyświetla je i prosi użytkownika o wybór.
Zwraca pełną ścieżkę do wybranego pliku lub `nothing`.
"""
function prompt_for_dataset(directory::String)
    println("\n--- Wybierz zbiór danych ---")

    if !isdir(directory)
        println("⚠️  Ostrzeżenie: Folder '$directory' nie istnieje. Próbuję kontynuować...")
        return nothing # Lub można tu rzucić błąd
    end

    # Znajdź pliki .csv (lub inne, np. .hdf5, jeśli potrzebujesz)
    files = filter(f -> endswith(f, ".csv") || endswith(f, ".hdf5"), readdir(directory))

    if isempty(files)
        println("Błąd: Nie znaleziono żadnych plików .csv lub .hdf5 w folderze '$directory'.")
        println("Upewnij się, że pliki z danymi znajdują się w odpowiednim miejscu.")
        return nothing
    end

    println("Znaleziono następujące pliki w folderze '$directory':")
    for (i, filename) in enumerate(files)
        println("  [$i] $filename")
    end

    while true
        print("Wybierz indeks pliku (np. 1): ")
        input = readline()
        try
            idx = parse(Int, input)
            if 1 <= idx <= length(files)
                selected_file = files[idx]
                filepath = joinpath(directory, selected_file)
                println("Wybrano plik: $filepath")
                return filepath
            else
                println("Błąd: Podaj indeks z zakresu 1-$(length(files)).")
            end
        catch e
            println("Błąd: Nieprawidłowy format. Wprowadź jedną liczbę.")
        end
    end
end
# ==============================================================================

function generate_output_filename_base(
    ic_filepath::String,
    theory::Symbol,
    selected_features::Vector{String},
)
    ic_name = splitext(basename(ic_filepath))[1]
    features_str = join(selected_features, "-")
    timestamp = Dates.format(now(), "YYYYmmdd_HHMMSS")
    return "pca_results_$(ic_name)_$(theory)_$(features_str)_$(timestamp)"
end


function run_full_pca_analysis(ic_filepath::String)
    println("="^60)
    println(" Rozpoczynanie analizy PCA z pliku: $ic_filepath")
    println("="^60)

    println("--- Krok 1: Uruchamianie symulacji ---")
    settings = modHydroSim.load_simulation_settings(ic_filepath)
    sim_result = modHydroSim.run_simulation(settings=settings, ic_file=ic_filepath)
    println("Symulacja zakończona.")

    println("\n--- Krok 2: Konfiguracja PCA ---")
    feature_indices, selected_feature_names = prompt_for_features()
    selected_method = prompt_for_pca_settings()

    # Dict{Symbol, Any}
    pca_params = Dict{Symbol,Any}(:method => selected_method)

    if selected_method == :kernel
        pca_params[:gamma] = prompt_for_kernel_parameters()
    end

    num_plots_to_generate = prompt_for_plot_count()
    n_pca_steps = 100
    n_components = 2

    info_text = "Plik: $(basename(ic_filepath)), Metoda: $selected_method, Cechy: $(join(selected_feature_names, ", "))"

    println("\n--- Krok 3: Obliczanie PCA ---")
    pca_results = modPCA.run_pca_over_time(
        sim_result,
        feature_indices,
        n_pca_steps,
        n_components,
        pca_params, # Dict{Symbol, Any}
    )

    if isempty(pca_results)
        println("\nBłąd: Nie udało się wygenerować żadnych wyników PCA. Przerywanie pracy.")
        return
    end

    filename_base = generate_output_filename_base(
        ic_filepath,
        sim_result.settings.theory,
        selected_feature_names,
    )

    println("\n--- Krok 4: Zapisywanie wykresów ---")

    fig_ev = modPlots.plot_explained_variance_evolution(pca_results; info_text=info_text)
    save("$(filename_base)_variance.png", fig_ev)
    println("Zapisano wykres wariancji.")

    fig_grid = modPlots.visualize_pca_static_grid(
        pca_results,
        sim_result,
        num_plots_to_generate;
        info_text=info_text,
    )
    save("$(filename_base)_grid.png", fig_grid)
    println("Zapisano siatkę wykresów PCA.")

    if pca_params[:method] != :kernel
        fig_loadings = modPlots.plot_loadings_evolution(
            pca_results,
            selected_feature_names;
            info_text=info_text,
        )
        save("$(filename_base)_loadings.png", fig_loadings)
        println("Zapisano wykres ładunków.")
    end

    println("\n--- Analiza PCA zakończona pomyślnie ---")
end

function run_phase_space_analysis(ic_filepath::String; tau_list::Vector{Float64}=[0.22, 0.4, 0.7, 1.0])
    println("+"^60)
    println(" Rozpoczynanie analizy wykresów fazowych (SpalHel) z pliku: $ic_filepath")
    println("+"^60)

    settings = modHydroSim.load_simulation_settings(ic_filepath)
    sim_result = modHydroSim.run_simulation(settings=settings, ic_file=ic_filepath)

    println("\n--- Generowanie wykresów fazowych (snapshoty) ---")

    # Użyj nowej funkcji siatki
    fig_grid = modPlots.plot_phase_space_grid(sim_result, tau_list)
    filename = "phase_space_grid_$(basename(ic_filepath)).png"
    save(filename, fig_grid)
    println("Zapisano siatkę wykresów fazowych jako: $filename")

    # Pętla poniżej jest teraz opcjonalna, jeśli chcesz mieć ODDZIELNE pliki
    # (zostawiam ją, ale nowa funkcja siatki jest lepsza)
    #=
    for (i, tau) in enumerate(tau_list)
        if tau >= settings.tspan[1] && tau <= settings.tspan[2]
            fig = modPlots.plot_phase_space_snapshot(sim_result, tau)
            filename = "phase_space_snapshot_tau_$(tau).png"
            save(filename, fig, dpi=300)
            println("Zapisano: $filename")
        else
            println("⚠️  Pominięto τ=$tau (poza zakresem symulacji $(settings.tspan))")
        end
    end
    =#

    println("\n✅ Zakończono generowanie wykresów fazowych")
end


# ==============================================================================
# ZMODYFIKOWANA FUNKCJA main()
# ==============================================================================
function main()
    # Krok 1: Zapytaj o plik z danymi
    csv_file = prompt_for_dataset("datasets")

    if isnothing(csv_file)
        println("Nie wybrano pliku. Zakończono działanie.")
        return
    end

    # Krok 2: Zapytaj, co zrobić z plikiem
    println("\n--- Wybierz akcję dla pliku: $(basename(csv_file)) ---")
    println("  [1] Uruchom pełną analizę PCA")
    println("  [2] Uruchom analizę przestrzeni fazowej (siatka snapshotów)")
    println("  [3] Wygeneruj animację przestrzeni fazowej")
    println("  [Inne] Zakończ")

    print("\nWybór: ")
    choice = readline()

    if choice == "1"
        # Uruchom pełną analizę PCA
        run_full_pca_analysis(csv_file)

    elseif choice == "2"
        tau_list_default = [0.22, 0.4, 0.7, 1.0, 2.0, 5.0]
        println("Używam domyślnej listy czasów: $tau_list_default")
        run_phase_space_analysis(csv_file, tau_list=tau_list_default)

    elseif choice == "3"
        println("--- Generowanie animacji przestrzeni fazowej ---")
        settings = modHydroSim.load_simulation_settings(csv_file)
        sim_result = modHydroSim.run_simulation(settings=settings, ic_file=csv_file)

        output_name = "anim_$(splitext(basename(csv_file))[1]).gif"
        modPlots.animate_phase_space_evolution(sim_result, output_filename=output_name)
        println("Zapisano animację jako: $output_name")

    else
        println("Zakończono.")
    end

    # println("--- Ręczne generowanie pojedynczego wykresu PCA ---")
    # settings = modHydroSim.load_simulation_settings(csv_file)
    # sim_result = modHydroSim.run_simulation(settings=settings, ic_file=csv_file)
    # modPlots.animate_phase_space_evolution(sim_result)
end

main()
