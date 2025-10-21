
# Skrypt do testowania 'work flow':
# 1. Generuje pliki danych z zapisanymi ustawieniami.
# 2. Uruchamia analizę PCA bezpośrednio z wygenerowanego pliku.

println("--- Uruchamianie skryptu testowego ---")

# Załaduj niezbędne moduły
include("pca.jl")
using .modHydroSim
using .PCAWorkflow

function run_test_workflow()
  test_settings = SimSettings(
    theory=:BRSSS,
    n_points=10,
    tspan=(0.2, 1.0),
    T_range=(500.0, 800.0),
    A_range=(-5.0, 5.0),
    seed=123
  )

  test_filename_base = "test_ic"
  println("\n[Test] Generowanie plików testowych...")
  generate_and_save_ics(settings=test_settings, output_filename_base=test_filename_base)

  test_csv_path = "$(test_filename_base)_$(test_settings.T_range)_$(test_settings.A_range)_$(test_settings.n_points)_t_$(test_settings.tspan).csv"

  println("\n[Test] Sprawdzanie, czy plik istnieje: ", isfile(test_csv_path) ? "Tak" : "Nie")

  # Krok 3: Uruchom analizę PCA z pliku CSV
  println("\n[Test] Uruchamianie analizy PCA z pliku: $test_csv_path")
  println("UWAGA: Proszę ręcznie wprowadzić dane w konsoli, aby kontynuować test.")
  try
    run_pca_workflow_from_file(test_csv_path)
    println("\n[Test] Analiza PCA zakończona pomyślnie.")
  catch e
    println("\n[Test] Wystąpił błąd podczas analizy PCA: ", e)
  end

  # Krok 4: Sprzątanie (opcjonalne)
  println("\n[Test] Sprzątanie plików testowych...")
  try
    rm(test_csv_path)
    rm("$(test_filename_base).h5")
    println("[Test] Pliki testowe zostały usunięte.")
  catch e
    println("[Test] Nie udało się usunąć plików testowych: ", e)
  end
end

# Uruchomienie testu
run_test_workflow()

println("\n--- Skrypt testowy zakończył działanie ---")
