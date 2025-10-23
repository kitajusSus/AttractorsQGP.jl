include("lib.jl")
using .modHydroSim
module PCAWorkflow

using ..modHydroSim
using Plots
using Statistics
using LinearAlgebra
using DataFrames
using CSV
using HDF5
using Dates


export run_full_pca_analysis, visualize_pca_from_file, run_pca_workflow_from_file

# --- SEKCJA 1: Struktury Danych ---

"""
    PCAResultAtTime
Przechowuje wyniki analizy PCA dla pojedynczego kroku czasowego `tau`.
"""
struct PCAResultAtTime
  tau::Float64
  transformed_data::Matrix{Float64}
  explained_variance_ratio::Vector{Float64}
  # Dla Liniowego PCA, to są wektory własne.
  # Dla Kernel PCA, to są wektory 'alpha' z dekompozycji jądra.
  principal_components::Matrix{Float64}
end



"""
    linear_pca(X, n_components; mode)
Wykonuje LINIOWĄ analizę PCA, używając klasycznej metody opartej
na dekompozycji własnej macierzy kowariancji.
"""
function linear_pca(X::Matrix{Float64}, n_components::Int; mode::Symbol=:standardize)
  n_samples, n_features = size(X)
  if n_components > n_features
    error("Liczba komponentów ($n_components) nie może być większa niż liczba cech ($n_features).")
  end

  X_transformed = copy(X)
  local X_scaled
  if mode == :standardize
    mean_vector = mean(X_transformed, dims=1)
    std_vector = std(X_transformed, dims=1)
    std_vector[std_vector.==0.0] .= 1.0
    X_scaled = (X_transformed .- mean_vector) ./ std_vector
  elseif mode == :center
    mean_vector = mean(X_transformed, dims=1)
    X_scaled = X_transformed .- mean_vector
  elseif mode == :minmax
    min_vals = minimum(X_transformed, dims=1)
    max_vals = maximum(X_transformed, dims=1)
    range_vals = max_vals .- min_vals
    range_vals[range_vals.==0.0] .= 1.0
    X_scaled = (X_transformed .- min_vals) ./ range_vals
  elseif mode == :none
    X_scaled = X_transformed
  else
    error("Nieznany tryb skalowania liniowego: $mode")
  end

  # Wzór: C = (1 / (n - 1)) * X_scaled^T * X_scaled
  # Gdzie n to liczba próbek (n_samples).
  cov_matrix = (1 / (n_samples - 1)) * (X_scaled' * X_scaled)

  # Rozwiązuje problem własny dla macierzy kowariancji.
  # Wzór: C * v_i = λ_i * v_i
  # Gdzie λ_i to wartości własne, a v_i to wektory własne.
  eigen_result = eigen(cov_matrix)
  eigenvalues = eigen_result.values   # Wartości własne (λ)
  eigenvectors = eigen_result.vectors # Wektory własne (v)

  # Sortuje wektory własne w kolejności malejącej, zgodnie z
  # odpowiadającymi im wartościami własnymi.
  sorted_indices = sortperm(eigenvalues, rev=true)
  sorted_eigenvalues = eigenvalues[sorted_indices]
  sorted_eigenvectors = eigenvectors[:, sorted_indices]

  # Tworzy macierz projekcji V, wybierając k pierwszych
  # (najważniejszych) posortowanych wektorów własnych.
  projection_matrix = sorted_eigenvectors[:, 1:n_components]

  # Transformuje oryginalne, przeskalowane dane na nową podprzestrzeń.
  # Wzór: Y = X_scaled * V
  transformed_data = X_scaled * projection_matrix

  # Wariancja wyjaśniona przez i-ty komponent to λ_i / sum(wszystkich λ).
  total_variance = sum(eigenvalues)
  explained_variance_ratio = sorted_eigenvalues[1:n_components] ./ total_variance

  return transformed_data, explained_variance_ratio, projection_matrix
end
"""
    kernel_pca(X, n_components; gamma)
Wykonuje JĄDROWĄ analizę PCA (Kernel PCA) od podstaw.
"""
function kernel_pca(X::Matrix{Float64}, n_components::Int; gamma::Float64)
  n_samples, n_features = size(X)

  # Karnel Gaussowski (Radial Basis Function, RBF):
  # k(xi, xj) = exp(-gamma * ||xi - xj||^2)
  # Macierz K ma wymiary (n_samples x n_samples). K_ij = k(x_i, x_j)
  K = zeros(n_samples, n_samples)
  for i in 1:n_samples
    for j in i:n_samples
      distance_sq = sum((X[i, :] .- X[j, :]) .^ 2)
      K[i, j] = exp(-gamma * distance_sq)
    end
  end
  K = Symmetric(K, :U)

  # Wzór: K_c = K - 1n*K - K*1n + 1n*K*1n
  # gdzie 1n to macierz (n x n) o wszystkich elementach równych 1/n.
  ones_n = ones(n_samples, n_samples) ./ n_samples
  K_centered = K - ones_n * K - K * ones_n + ones_n * K * ones_n

  # Wzór: K_c * alpha_i = lambda_i * alpha_i
  eigen_result = eigen(K_centered)
  eigenvalues = real(eigen_result.values)
  eigenvectors = real(eigen_result.vectors) # Wektory alpha

  sorted_indices = sortperm(eigenvalues, rev=true)
  sorted_lambdas = eigenvalues[sorted_indices]
  sorted_alphas = eigenvectors[:, sorted_indices]

  # Współrzędne oryginalnych danych w nowej przestrzeni są dane przez
  # wektory własne pomnożone przez pierwiastek z wartości własnych.
  selected_lambdas = sorted_lambdas[1:n_components]
  selected_alphas = sorted_alphas[:, 1:n_components]
  transformed_data = selected_alphas * diagm(sqrt.(max.(selected_lambdas, 0)))

  # Proporcja wariancji: lambda_i / sum(all_lambdas).
  total_variance = sum(max.(eigenvalues, 0))
  explained_variance_ratio = max.(selected_lambdas, 0) ./ total_variance

  return transformed_data, explained_variance_ratio, selected_alphas
end



"""
    calc_pca(...)
Orkiestruje procesem analizy PCA, wywołując odpowiednią funkcję (liniową lub jądrową).
"""
function calc_pca(
  sim_result::modHydroSim.SimResult;
  feature_indices::Vector{Int},
  n_pca_steps::Int,
  n_components::Int,
  pca_method_params::Dict
)
  method = pca_method_params[:method]
  println("\n--- Krok 3: Uruchamianie analizy PCA... ---")
  println("Wybrana metoda: $method")

  t_start, t_end = sim_result.settings.tspan
  sample_times = range(t_start, stop=t_end, length=n_pca_steps)
  pca_results_vector = PCAResultAtTime[]

  for (i, tau) in enumerate(sample_times)
    print("\rPrzetwarzanie kroku czasowego: $i/$n_pca_steps (τ = $(round(tau, digits=2)) fm/c)")

    all_data_vectors = modHydroSim.TA(sim_result, tau)
    data_matrix = hcat([all_data_vectors[i] for i in feature_indices]...)

    valid_mask = vec(all(isfinite, data_matrix, dims=2))
    if sum(valid_mask) < n_components
      println("\nOstrzeżenie: Zbyt mało prawidłowych danych w czasie τ=$tau. Pomijanie kroku.")
      continue
    end

    filtered_matrix = data_matrix[valid_mask, :]

    local transformed_data, explained_ratio, components
    if method == :kernel
      gamma = pca_method_params[:gamma]
      transformed_data, explained_ratio, components = kernel_pca(filtered_matrix, n_components, gamma=gamma)
    else
      transformed_data, explained_ratio, components = linear_pca(filtered_matrix, n_components, mode=method)
    end

    push!(pca_results_vector, PCAResultAtTime(tau, transformed_data, explained_ratio, components))
  end
  println("\nAnaliza PCA zakończona. Przetworzono $(length(pca_results_vector)) kroków czasowych.")
  return pca_results_vector
end



function prompt_for_features(sim_result::modHydroSim.SimResult)
  if sim_result.settings.theory == :HJSW
    all_feature_names = ["T", "A", "Z", "dT/dτ", "dA/dτ", "dZ/dτ"]
  else
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


function prompt_for_pca_settings()
  println("\n--- Krok 2: Wybór metody PCA ---")
  methods = [:standardize, :center, :minmax, :none, :kernel]
  descriptions = [
    "Liniowa PCA ze standaryzacją (średnia=0, odch. std.=1) - ZALECANE",
    "Liniowa PCA z centrowaniem (odjęcie średniej)",
    "Liniowa PCA z normalizacją Min-Max [0, 1]",
    "Liniowa PCA bez skalowania (surowe dane)",
    "Kernel PCA (Kernel PCA) z RBF (do nieliniowych zależności)"
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

function prompt_for_kernel_parameters(data_matrix::Matrix{Float64})
  n_features = size(data_matrix, 2)
  gamma_default = 1.0 / n_features
  println("\n--- Konfiguracja Kernel PCA (Kernel PCA) ---")
  println("KERNEL PCA używa kernel RBF: k(x,y) = exp(-gamma * ||x-y||^2)")

  while true
    print("Podaj wartość parametru gamma [domyślnie: $(round(gamma_default, digits=4))]: ")
    input = readline()
    if isempty(input)
      println("Użyto domyślnej wartości gamma = $gamma_default")
      return Dict(:gamma => gamma_default)
    end
    try
      gamma = parse(Float64, input)
      if gamma > 0
        println("Ustawiono gamma = $gamma")
        return Dict(:gamma => gamma)
      else
        println("Błąd: gamma musi być wartością dodatnią.")
      end
    catch e
      println("Błąd: Nieprawidłowy format. Wprowadź liczbę.")
    end
  end
end



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
  ic_filepath::String,
  pca_method_params::Dict
)
  println("\n--- Krok 4: Zapisywanie wyników... ---")

  csv_path = filename_base * ".csv"
  println("Trwa przygotowywanie danych do zapisu w formacie CSV...")

  df_rows = []

  initial_states = collect(hcat([sol.u[1] for sol in sim_result.solutions]...)')
  for result in pca_results
    tau = result.tau
    if size(result.transformed_data, 1) != size(initial_states, 1)
      @warn "Niezgodność liczby symulacji dla tau = $tau. Pomijanie zapisu do CSV dla tego kroku."
      continue
    end

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

      for j in 1:length(result.explained_variance_ratio)
        row = merge(row, (Symbol("ExplainedVariance_$j") => result.explained_variance_ratio[j],))
      end

      push!(df_rows, row)
    end
  end

  df = DataFrame(df_rows)
  CSV.write(csv_path, df)
  println("📁 Zapisano wyniki w formacie CSV do: $csv_path")

  h5_path = filename_base * ".h5"
  h5open(h5_path, "w") do file
    attrs(file)["description"] = "Wyniki analizy PCA dla symulacji hydrodynamicznej."
    attrs(file)["source_ic_file"] = basename(ic_filepath)
    attrs(file)["theory"] = string(sim_result.settings.theory)
    attrs(file)["selected_features"] = join(selected_feature_names, ", ")
    attrs(file)["pca_method"] = string(pca_method_params[:method])
    if pca_method_params[:method] == :kernel
      attrs(file)["kernel_gamma"] = pca_method_params[:gamma]
    end
    attrs(file)["timestamp"] = string(now())

    g = create_group(file, "data")
    g["tau"] = [res.tau for res in pca_results]
    transformed_data_3d = cat([res.transformed_data for res in pca_results]...; dims=3)
    g["transformed_data"] = transformed_data_3d
    explained_variance_matrix = collect(hcat([res.explained_variance_ratio for res in pca_results]...)')
    g["explained_variance"] = explained_variance_matrix

    g_pc = create_group(g, "principal_components")
    principal_components_3d = cat([res.principal_components for res in pca_results]...; dims=3)
    g_pc["values"] = principal_components_3d
  end
  println("🗄️ Zapisano wyniki w formacie HDF5 do: $h5_path")
end

function prompt_for_plot_count()
  println("\n--- Krok 5: Konfiguracja wizualizacji ---")
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

function plot_explained_variance_evolution(
  pca_results::Vector{PCAResultAtTime};
  source_file::String,
  feature_names::Vector{String},
  pca_method_params::Dict
)
  println("\n--- Generowanie wykresu ewolucji wariancji wyjaśnionej... ---")

  if isempty(pca_results)
    println("Brak wyników do narysowania wykresu wariancji.")
    return
  end

  taus = [res.tau for res in pca_results]
  n_components = length(pca_results[1].explained_variance_ratio)

  method_info = "Metoda: $(pca_method_params[:method])"
  if pca_method_params[:method] == :kernel
    method_info *= " (gamma=$(round(pca_method_params[:gamma], digits=4)))"
  end
  settings_info = "Plik: $(basename(source_file)) | $method_info | Cechy: $(join(feature_names, ", "))"

  p = plot(
    title="Ewolucja wariancji wyjaśnionej przez komponenty PCA",
    plot_title=settings_info,
    xlabel="Czas τ [fm/c]",
    ylabel="Proporcja wyjaśnionej wariancji",
    legend=:best,
    ylim=(0, 1.1)
  )

  for i in 1:n_components
    variance_data = [res.explained_variance_ratio[i] for res in pca_results]
    plot!(p, taus, variance_data, label="PC $i", linewidth=2)
  end

  if n_components > 1
    cumulative_variance = [sum(res.explained_variance_ratio) for res in pca_results]
    plot!(p, taus, cumulative_variance, label="Suma", linestyle=:dash, color=:black)
  end

  hline!(p, [1.0], linestyle=:dot, color=:grey, label="", alpha=0.7)

  mkpath("plots")
  filename = "pca_explained_variance_evolution.png"
  savefig(p, joinpath("plots", filename))
  println("Saved plot to: $(joinpath("plots", filename))")
end

function visualize_pca_static_grid(
  pca_results::Vector{PCAResultAtTime},
  sim_result::modHydroSim.SimResult,
  num_plots::Int;
  source_file::String,
  feature_names::Vector{String},
  pca_method_params::Dict
)
  println("\n--- Generowanie siatki wykresów PCA ---")

  initial_temps = [sol.u[1][1] for sol in sim_result.solutions]

  method_info = "Metoda: $(pca_method_params[:method])"
  if pca_method_params[:method] == :kernel
    method_info *= " (gamma=$(round(pca_method_params[:gamma], digits=4)))"
  end
  settings_info = "Plik: $(basename(source_file)) | $method_info | Cechy: $(join(feature_names, ", "))"

  total_steps = length(pca_results)
  indices_to_plot = unique(round.(Int, range(1, stop=total_steps, length=num_plots)))

  plots_array = []
  all_pc1 = vcat([res.transformed_data[:, 1] for res in pca_results]...)
  all_pc2 = vcat([res.transformed_data[:, 2] for res in pca_results]...)

  min_pc1, max_pc1 = minimum(all_pc1), maximum(all_pc1)
  min_pc2, max_pc2 = minimum(all_pc2), maximum(all_pc2)

  x_margin = (max_pc1 - min_pc1) * 0.05
  y_margin = (max_pc2 - min_pc2) * 0.05

  x_margin = x_margin > 0 ? x_margin : 1.0
  y_margin = y_margin > 0 ? y_margin : 1.0

  global_xlims = (min_pc1 - x_margin, max_pc1 + x_margin)
  global_ylims = (min_pc2 - y_margin, max_pc2 + y_margin)

  for idx in indices_to_plot
    result = pca_results[idx]
    current_tau = result.tau
    total_explained_var = sum(result.explained_variance_ratio) * 100

    p = scatter(
      result.transformed_data[:, 1],
      result.transformed_data[:, 2],
      marker_z=initial_temps,
      title="τ = $(round(current_tau, digits=2)) fm/c (Var: $(round(total_explained_var, digits=1))%)",
      xlabel="PC 1",
      ylabel="PC 2",
      label="",
      xlims=global_xlims,
      ylims=global_ylims,
      markersize=4,
      alpha=0.8,
      color=:plasma,
      colorbar=false,
      legend=false
    )
    push!(plots_array, p)
  end

  layout_cols = ceil(Int, sqrt(length(plots_array)))
  layout_rows = ceil(Int, length(plots_array) / layout_cols)

  println("✅ Tworzenie siatki $(layout_rows)x$(layout_cols) wykresów...")

  final_grid = plot(plots_array...,
    layout=(layout_rows, layout_cols),
    plot_title=settings_info,
    size=(350 * layout_cols, 300 * layout_rows)
  )

  mkpath("plots")
  filename = "pca_static_grid.png"
  savefig(final_grid, joinpath("plots", filename))
  println("Saved plot to: $(joinpath("plots", filename))")
end


function plot_loadings_evolution(
  pca_results::Vector{PCAResultAtTime},
  selected_feature_names::Vector{String};
  source_file::String,
  pca_method_params::Dict
)

  if isempty(pca_results) || pca_method_params[:method] == :kernel
    println("EVR NIE DZIAŁA DLA KERNEL.")
    return
  end

  taus = [res.tau for res in pca_results]
  n_features, n_components = size(pca_results[1].principal_components)

  method_info = "Metoda: $(pca_method_params[:method])"
  main_title_info = "Plik: $(basename(source_file)) | $method_info"

  for i in 1:n_components
    p = plot(
      title="Ewolucja ładunków dla PC$i",
      plot_title=main_title_info,
      xlabel="Czas τ [fm/c]",
      ylabel="Wartość ładunku (Loading)",
      legend=:outerright,
      ylim=(-1.1, 1.1)
    )

    for j in 1:n_features
      loading_data = [res.principal_components[j, i] for res in pca_results]
      feature_name = selected_feature_names[j]
      plot!(p, taus, loading_data, label=feature_name, linewidth=2)
    end

    hline!(p, [0.0], linestyle=:dot, color=:grey, label="", alpha=0.7)

    mkpath("plots")
    filename = "pca_loadings_evolution_pc$(i).png"
    savefig(p, joinpath("plots", filename))
    println("Saved plot to: $(joinpath("plots", filename))")
  end

  println("✅ Wykresy  gotowe. Uwaga: Nagłe 'odbicia lustrzane' (zmiany znaku)")
end

function run_full_pca_analysis(sim_result::modHydroSim.SimResult, ic_filepath::String)
  println("--- Uruchamianie pełnego przepływu pracy analizy PCA ---")

  feature_indices, selected_feature_names = prompt_for_features(sim_result)
  selected_method = prompt_for_pca_settings()
  pca_params = Dict{Symbol,Any}(:method => selected_method)

  if selected_method == :kernel
    tau_sample = sim_result.settings.tspan[1]
    sample_data = hcat([modHydroSim.TA(sim_result, tau_sample)[i] for i in feature_indices]...)
    kernel_params = prompt_for_kernel_parameters(sample_data)
    merge!(pca_params, kernel_params)
  end

  n_pca_steps = 50
  n_components = 2

  pca_results = calc_pca(
    sim_result;
    feature_indices=feature_indices,
    n_pca_steps=n_pca_steps,
    n_components=n_components,
    pca_method_params=pca_params
  )

  if isempty(pca_results)
    println("\nBłąd: Nie udało się wygenerować wyników PCA. Przerywanie pracy.")
    return
  end
  plot_explained_variance_evolution(
    pca_results;
    source_file=ic_filepath,
    feature_names=selected_feature_names,
    pca_method_params=pca_params
  )

  plot_loadings_evolution(
    pca_results,
    selected_feature_names;
    source_file=ic_filepath,
    pca_method_params=pca_params
  )


  visualize_pca_static_grid(
    pca_results, sim_result, num_plots_to_generate;
    source_file=ic_filepath,
    feature_names=selected_feature_names,
    pca_method_params=pca_params
  )

  println("\n--- Zakończono przepływ pracy PCA ---")
end


"""
Główna funkcja orkiestrująca, która wczytuje dane i ustawienia z pliku,
uruchamia symulację, a następnie pełną analizę PCA.
"""
function run_pca_workflow_from_file(ic_filepath::String)
  println("="^60)
  println(" Rozpoczynanie analizy PCA z pliku: $ic_filepath")
  println("="^60)

  settings = modHydroSim.load_settings(ic_filepath)
  sim_result = modHydroSim.run_simulation(settings=settings, ic_file=ic_filepath)
  run_full_pca_analysis(sim_result, ic_filepath)

  println("\n\nAnaliza PCA zakończona pomyślnie.")
end

end
