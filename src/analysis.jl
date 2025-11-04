#  REPL: include("analysis.jl"); run_from_file("nazwa_pliku.csv")

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

    println("\n--- Generowanie wykresów fazowych ---")
    for (i, tau) in enumerate(tau_list)
        if tau >= settings.tspan[1] && tau <= settings.tspan[2]
            fig = modPlots.plot_phase_space_snapshot(sim_result, tau)
            filename = "phase_space_snapshot_tau_$(tau).png"
            save(filename, fig)
            println("Zapisano: $filename")
        else
            println("⚠️  Pominięto τ=$tau (poza zakresem symulacji $(settings.tspan))")
        end
    end
    println("\n✅ Zakończono generowanie wykresów fazowych")
end

function main()
    # csv_file = "datasets/SPALHEL_(2.030456852791878, 7.614213197969542)_(-8.0, 20.0)_10000_t_(0.22, 1.0).csv"
    # csv_file = "datasets/DUZEDANE_(400.0, 2500.0)_(-8.0, 20.0)_10000_t_(0.22, 1.0).csv"
    csv_file = "nowe_dane_.csv"
    # run_full_pca_analysis(csv_file)

    run_phase_space_analysis(csv_file, tau_list=[0.22, 0.35, 0.55, 0.6, 0.86, 2, 5, 5.5])

    println("--- Ręczne generowanie pojedynczego wykresu PCA ---")

    settings = modHydroSim.load_simulation_settings(csv_file)
    sim_result = modHydroSim.run_simulation(settings=settings, ic_file=csv_file)

    my_tau = 1.0
    my_features = [1, 3]
    my_feature_names = ["T", "dTdτ"]
    my_pca_params = Dict{Symbol,Any}(:method => :standardize)
    my_info = "Plik: $(basename(csv_file)), Cechy: T, dTdτ"

    fig = modPlots.plot_pca_snapshot(
        sim_result,
        my_tau,
        my_features,
        my_pca_params;
        info_text=my_info
    )

    save("pca_snapshot_tau_$(my_tau).png", fig)
    println("Zapisano pojedynczy wykres PCA dla tau = $my_tau.")


    modPlots.animate_phase_space_evolution(sim_result)


end

main()
