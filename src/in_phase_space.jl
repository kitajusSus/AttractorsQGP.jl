include("lib.jl")
include("pca.jl")

# Użyj modułów
using .modHydroSim
using .PCAWorkflow




function main()
  ic_filepath = "bigMISdataset_(400.0, 1200.0)_(-8.0, 25.0)_10000_t_(0.22, 3.0).csv"

  n_pca_steps = 10 # Można dostosować, aby uzyskać więcej lub mniej "migawwek" w czasie

  n_components = 2 # Zgodnie z artykułem, analiza koncentruje się na redukcji wymiarowości do 1D lub 2D

  pca_method_params = Dict(:method => :standardize)
  num_plots = 6 # Można zmienić


  println("="^60)
  println(" Rozpoczynanie analizy z pliku: $ic_filepath")
  println(" Używana przestrzeń fazowa do PCA: (T, Ṫ)")
  println("="^60)

  settings = modHydroSim.load_settings(ic_filepath)
  println("Wczytano ustawienia:")
  println("  Teoria: $(settings.theory)")
  println("  Zakres czasu (tspan): $(settings.tspan)")
  println("  Liczba punktów (n_points): $(settings.n_points)")
  if settings.theory != :MIS
    @warn "Teoria w pliku CSV to $(settings.theory), ale eksperyment dotyczy MIS. Kontynuuję z ustawieniami z pliku."
  end

  sim_result = modHydroSim.run_simulation(settings=settings, ic_file=ic_filepath)

  feature_indices = [1, 3]
  selected_feature_names = ["T", "Tdot"] # Nazwy odpowiadające indeksom
  println("\nWybrano cechy do PCA: $(join(selected_feature_names, ", ")) (Indeksy: $feature_indices)")

  println("\nRozpoczynanie obliczeń PCA...")
  pca_results = PCAWorkflow.calc_pca(
    sim_result;
    feature_indices=feature_indices,
    n_pca_steps=n_pca_steps,
    n_components=n_components,
    pca_method_params=pca_method_params
  )

  if isempty(pca_results)
    println("\nBłąd: Nie udało się wygenerować żadnych wyników PCA. Przerywanie pracy.")
  else
    println("\nObliczenia PCA zakończone pomyślnie.")

    # --- Krok 3: Zapis wyników i generowanie wizualizacji ---

    # Generuj bazową nazwę pliku wyjściowego
    filename_base = PCAWorkflow.generate_output_filename_base(ic_filepath, settings.theory, selected_feature_names)

    # Zapisz wyniki PCA do plików .csv i .h5
    PCAWorkflow.save_pca_results(
      filename_base,
      pca_results,
      sim_result,
      selected_feature_names,
      ic_filepath,
      pca_method_params
    )

    # Wygeneruj wykres ewolucji wyjaśnionej wariancji (EVR)
    PCAWorkflow.plot_explained_variance_evolution(
      pca_results;
      source_file=ic_filepath,
      feature_names=selected_feature_names,
      pca_method_params=pca_method_params
    )

    # Wygeneruj wykresy ewolucji ładunków (loadings), jeśli użyto metody liniowej
    PCAWorkflow.plot_loadings_evolution(
      pca_results,
      selected_feature_names; # Używamy nazw T i Tdot
      source_file=ic_filepath,
      pca_method_params=pca_method_params
    )

    # Wygeneruj siatkę statycznych wykresów PCA pokazujących rozkład punktów w przestrzeni PC1-PC2 w różnych krokach czasowych
    PCAWorkflow.visualize_pca_static_grid(
      pca_results, sim_result, num_plots;
      source_file=ic_filepath,
      feature_names=selected_feature_names,
      pca_method_params=pca_method_params
    )

    println("\n--- Analiza zakończona. Wyniki zapisano, wykresy wygenerowano w katalogu 'plots'. ---")
  end
end


main()
