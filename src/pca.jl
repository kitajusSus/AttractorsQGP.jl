include("lib.jl")
include("plt.jl")

using .modHydroSim
using .modPlots

module PCAWorkflow

using ..modHydroSim
using ..modPlots
using Plots
gr()

using Statistics
using LinearAlgebra
using DataFrames
using CSV
using LaTeXStrings
using HDF5
using Dates
using MultivariateStats

export run_full_pca_analysis,
  run_pca_workflow_from_file,
  run_pca_at_time,
  run_SpalHel,
  generate_trajectory_data,
  run_trajectory_animation,
  PCAResultAtTime,
  calc_pca

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
    error("Liczba komponentów ($n_components) nie może być większa niż liczba cech ($n_features).")
  end

  X_transformed = copy(X)
  local X_scaled
  if mode == :standardize
    mean_vector = mean(X_transformed, dims=1)
    std_vector = std(X_transformed, dims=1)
    std_vector[std_vector.==0.0] .=
      1.0
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


  X_transposed = X_scaled'

  M_linear = fit(PCA, X_transposed;
    maxoutdim=n_components, pratio=1.0)

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

  local explained_variance_ratio
  if total_variance <= 1e-10
    explained_variance_ratio = zeros(n_components)
  else
    explained_variance_ratio = all_eigenvalues[1:n_components] ./ total_variance
  end

  selected_alphas = projection(M_kernel)

  return transformed_data, explained_variance_ratio, selected_alphas
end

function calc_pca(
  sim_result::modHydroSim.SimResult,
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

    all_data_vectors, valid_mask = modHydroSim.TA(sim_result, tau)

    if sum(valid_mask) < n_components
      println("\nOstrzeżenie: Zbyt mało prawidłowych danych (znaleziono $(sum(valid_mask))) w czasie τ=$tau. Wymagane co najmniej $n_components. Pomijanie kroku.")

      continue
    end


    data_matrix = hcat([all_data_vectors[i][valid_mask] for i in feature_indices]...)

    if size(unique(data_matrix, dims=1), 1) < n_components
      println("\nOstrzeżenie: Zbyt mało unikalnych danych (znaleziono $(size(unique(data_matrix, dims=1), 1))) w czasie τ=$tau.
 Wymagane co najmniej $n_components. Pomijanie kroku.")
      continue
    end

    local transformed_data, explained_ratio, components
    try
      if method == :kernel
        gamma = pca_method_params[:gamma]
        transformed_data, explained_ratio, components = kernel_pca(data_matrix, n_components, gamma=gamma)
      else
        transformed_data, explained_ratio, components = linear_pca(data_matrix, n_components, mode=method)
      end

      if any(isnan, transformed_data) || any(isnan, explained_ratio)

        println("\nOstrzeżenie: Wynik PCA dla τ=$tau zawiera NaN. Pomijanie kroku.")
        continue
      end

      push!(pca_results_vector, PCAResultAtTime(tau, transformed_data, explained_ratio, components, valid_mask))

    catch e
      println("\nBłąd podczas przetwarzania PCA w czasie τ=$tau. Pomijanie kroku. Błąd: $e")
      continue
    end
  end
  println("\nAnaliza PCA zakończona. Przetworzono $(length(pca_results_vector)) kroków czasowych.")
  return pca_results_vector
end

function prompt_for_features(sim_result::modHydroSim.SimResult)
  if sim_result.settings.theory == :HJSW
    all_feature_names = ["T", "A", "Z", "dTdτ", "dAdτ", "dZdτ"]

  else
    all_feature_names = ["T", "A", "dTdτ", "dAdτ", "SpalHel"]
  end

  println("\n Dostępne cechy do analizy PCA:")
  for (i, name) in enumerate(all_feature_names)
    println("  [$i] $name")
  end

  while true
    print("Wybierz indeksy cech do analizy (oddzielone przecinkami, np.
 1,2,4): ")
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
    print("Wybierz indeks metody (np.
 1): ")
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
      println("Błąd: Nieprawidłowy format. Wprowadź
 jedną liczbę.")
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
      gamma = parse(Float64,
        input)
      if gamma > 0
        println("Ustawiono gamma = $gamma")
        return Dict(:gamma => gamma)
      else
        println("Błąd: gamma musi być wartością dodatnią.")
      end
    catch e
      println("Błąd: Nieprawidłowy format.
 Wprowadź liczbę.")
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

  if isempty(pca_results)
    println("Brak wyników PCA do zapisania.")
    return
  end

  csv_path = filename_base * ".csv"
  println("Trwa przygotowywanie danych do zapisu w formacie CSV...")

  df_rows = []

  initial_states_raw = [sol.u[1] for sol in sim_result.solutions]
  all_initial_states_matrix = collect(hcat(initial_states_raw...)')

  for result in
      pca_results
    tau = result.tau
    valid_mask = result.valid_mask

    valid_initial_states = all_initial_states_matrix[valid_mask, :]

    if size(result.transformed_data, 1) != size(valid_initial_states, 1)
      @warn "Niezgodność liczby symulacji dla tau = $tau.
 Oczekiwano $(size(result.transformed_data, 1)), znaleziono $(size(valid_initial_states, 1)) stanów początkowych. Pomijanie zapisu do CSV dla tego kroku."
      continue
    end

    for i in 1:size(result.transformed_data, 1)
      row = (
        tau=tau,
        simulation_id=i,
        T_0=valid_initial_states[i, 1],
        A_0=valid_initial_states[i, 2],
      )

      if size(valid_initial_states, 2) > 2
        row =
          merge(row, (Z_0=valid_initial_states[i, 3],))
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

  if isempty(df_rows)
    println("Brak danych do zapisania w pliku CSV.")
  else
    df =
      DataFrame(df_rows)
    CSV.write(csv_path, df)
    println("📁 Zapisano wyniki w formacie CSV do: $csv_path")
  end

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

    g_transformed = create_group(g, "transformed_data")
    for (i, res) in enumerate(pca_results)
      g_transformed["tau_$(i)"] = res.transformed_data
    end

    explained_variance_matrix = collect(hcat([res.explained_variance_ratio for res in pca_results]...)')
    g["explained_variance"] = explained_variance_matrix

    g_pc = create_group(g, "principal_components")
    for (i, res) in enumerate(pca_results)
      g_pc["tau_$(i)"] = res.principal_components
    end
  end
  println("🗄️ Zapisano wyniki w formacie HDF5 do: $h5_path")
end

function prompt_for_plot_count()
  println("\n--- Krok 5: Konfiguracja wizualizacji ---")
  while true
    print("Ile statycznych wykresów PCA chcesz wygenerować?
 (np. 6): ")
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

function run_pca_at_time(
  filepath::String,
  tau::Float64;
  feature_names::Vector{String},
  n_components::Int,
  pca_method_params::Dict
)
  method = pca_method_params[:method]
  println("\n--- Uruchamianie analizy PCA z pliku dla τ = $tau fm/c ---")
  println("Wybrana metoda: $method")

  local data_matrix::Matrix{Float64}

  try
    if !(endswith(filepath, ".csv") || endswith(filepath, ".h5"))

      println("Błąd: Nieobsługiwany format pliku. Użyj .csv lub .h5")
      return nothing
    end

    println("Wczytywanie pliku: $filepath ...")

    local df_tau::DataFrame
    if endswith(filepath, ".csv")
      df_all = CSV.read(filepath, DataFrame)
      df_tau = filter(row -> isapprox(row.tau, tau), df_all)
    elseif endswith(filepath, ".h5")
      println("Tryb HDF5: Wczytywanie...")
      df_h5 = h5open(filepath, "r") do file
        all_feature_names = unique(vcat(feature_names, "tau"))

        data_cols = []
        col_names = []
        for name in all_feature_names
          if haskey(file, name)
            push!(data_cols, read(file[name]))
            push!(col_names, name)
          else
            println("Błąd: Brak datasetu '$name' w pliku HDF5.")

            return nothing
          end
        end
        DataFrame(data_cols, col_names)
      end

      if isnothing(df_h5)
        return nothing
      end

      df_tau = filter(row -> isapprox(row.tau, tau), df_h5)
    end

    if isempty(df_tau)
      println("Błąd: Nie znaleziono w pliku żadnych danych dla τ ≈ $tau.")

      return nothing
    end

    println("Znaleziono $(nrow(df_tau)) próbek dla τ ≈ $tau.")

    data_matrix = Matrix(df_tau[!, feature_names])

  catch e
    println("Błąd podczas wczytywania lub filtrowania danych: $e")
    println("Upewnij się, że plik istnieje i zawiera kolumnę 'tau' (lub dataset) oraz cechy: $feature_names")
    return nothing
  end

  valid_mask = vec(all(isfinite, data_matrix, dims=2))
  if sum(valid_mask) < n_components ||
     size(unique(data_matrix[valid_mask, :], dims=1), 1) < n_components
    println("\nOstrzeżenie: Zbyt mało unikalnych/prawidłowych danych w czasie τ=$tau. Nie można wykonać PCA.")
    return nothing
  end

  filtered_matrix = data_matrix[valid_mask, :]
  if size(filtered_matrix, 1) < n_components
    println("\nOstrzeżenie: Po filtrowaniu zostało zbyt mało danych ($(size(filtered_matrix, 1))) w czasie τ=$tau.")
    return nothing
  end

  local transformed_data, explained_ratio, components
  try
    if method == :kernel
      gamma = pca_method_params[:gamma]
      transformed_data, explained_ratio, components = kernel_pca(filtered_matrix, n_components, gamma=gamma)
    else

      transformed_data, explained_ratio, components = linear_pca(filtered_matrix, n_components, mode=method)
    end

    println("Analiza PCA dla τ=$tau zakończona pomyślnie.")
    return PCAResultAtTime(tau, transformed_data, explained_ratio, components, valid_mask)

  catch e
    println("\nBłąd podczas przetwarzania PCA w czasie τ=$tau.
 Błąd: $e")
    return nothing
  end
end

function run_full_pca_analysis(sim_result::modHydroSim.SimResult, ic_filepath::String)
  println("--- Uruchamianie pełnego przepływu pracy analizy PCA ---")

  feature_indices, selected_feature_names = prompt_for_features(sim_result)
  selected_method = prompt_for_pca_settings()
  pca_params = Dict{Symbol,Any}(:method => selected_method)

  if selected_method == :kernel
    tau_sample = sim_result.settings.tspan[1]
    all_data_vectors, valid_mask = modHydroSim.TA(sim_result, tau_sample)

    if sum(valid_mask) > 0
      sample_data = hcat([all_data_vectors[i][valid_mask] for i in feature_indices]...)
      kernel_params = prompt_for_kernel_parameters(sample_data)
      merge!(pca_params, kernel_params)
    else
      println("Brak prawidłowych danych
 do skonfigurowania kernela. Używanie domyślnych parametrów może się nie udać.")
    end
  end

  num_plots_to_generate = prompt_for_plot_count()
  n_pca_steps = 10
  n_components = 2

  pca_results = calc_pca(
    sim_result,
    feature_indices,
    n_pca_steps,
    n_components,
    pca_params
  )

  if isempty(pca_results)
    println("\nBłąd: Nie udało się wygenerować żadnych wyników PCA. Przerywanie pracy.")
    return
  end

  filename_base = generate_output_filename_base(ic_filepath, sim_result.settings.theory, selected_feature_names)
  save_pca_results(
    filename_base,
    pca_results,
    sim_result,
    selected_feature_names,
    ic_filepath,
    pca_params
  )

  modPlots.plot_explained_variance_evolution(
    pca_results;
    source_file=ic_filepath,
    feature_names=selected_feature_names,
    pca_method_params=pca_params
  )

  modPlots.plot_loadings_evolution(
    pca_results, selected_feature_names;
    source_file=ic_filepath,
    pca_method_params=pca_params
  )

  modPlots.visualize_pca_static_grid(
    pca_results, sim_result, num_plots_to_generate;
    source_file=ic_filepath,
    feature_names=selected_feature_names,
    pca_method_params=pca_params
  )

  println("\n--- Zakończono prace funkcji run_full_pca_analysis  ---")
end

function run_pca_workflow_from_file(ic_filepath::String)
  println("="^60)
  println(" Rozpoczynanie analizy PCA z pliku: $ic_filepath")
  println("="^60)

  settings = modHydroSim.load_settings(ic_filepath)
  sim_result = modHydroSim.run_simulation(settings=settings, ic_file=ic_filepath)
  run_full_pca_analysis(sim_result, ic_filepath)

  println("\n\nAnaliza PCA zakończona pomyślnie.")
end

function run_SpalHel(ic_filepath::String;
  tau_list::Vector{Float64}=[0.3, 0.4, 0.5])
  println("+"^60)
  println(" Rozpoczynanie analizy wykresów fazowych z pliku: $ic_filepath")
  println("+"^60)

  settings = modHydroSim.load_settings(ic_filepath)
  sim_result = modHydroSim.run_simulation(settings=settings, ic_file=ic_filepath)

  println("\n--- Generowanie wykresów fazowych ---")
  for tau in tau_list
    if tau >= settings.tspan[1] && tau <= settings.tspan[2]
      modPlots.wykres_fazowy(sim_result, tau=tau)
    else
      println("⚠️  Pominięto τ=$tau (poza zakresem symulacji)")
    end
  end

  println("\n✅ Zakończono generowanie wykresów fazowych")
end


function generate_trajectory_data(
  ic_filepath::String;
  n_time_steps::Int=50
)
  println("function generate_trajectory_data")
  println(" Generowanie danych trajektorii z pliku: $ic_filepath")
  println("+"^60)


  settings = modHydroSim.load_settings(ic_filepath)
  sim_result = modHydroSim.run_simulation(settings=settings, ic_file=ic_filepath)

  println("\n--- Ekstrakcja danych trajektorii ---")

  tspan = settings.tspan
  tau_0 = settings.tspan[1]
  time_points = range(tspan[1], stop=tspan[2], length=n_time_steps)

  rows = []
  for (run_id, sol) in enumerate(sim_result.solutions)

    T_0 = sol.u[1][1]
    A_0 = sol.u[1][2]

    for t in time_points

      if t >= sol.t[1] && t <= sol.t[end]
        state = sol(t)
        T_val = state[1]
        A_val = state[2]

        derivatives = sol(t, Val{1})
        dTdtau_val = derivatives[1]
        dAdtau_val = derivatives[2]

        push!(rows, (
          tau=t,
          Run_ID=run_id,
          T_0=T_0,
          A_0=A_0,
          T=T_val,
          A=A_val,
          dTdtau=dTdtau_val,
          dAdtau=dAdtau_val
        )
        )
      end
    end
  end

  df = DataFrame(rows)
  println("✅ Wygenerowano dane
 trajektorii w pamięci")
  println("✅ Liczba trajektorii: $(length(unique(df.Run_ID)))")
  println("✅ Liczba kroków czasowych: $(length(unique(df.tau)))")

  return df
end

function run_trajectory_animation(
  ic_filepath::String;
  output_gif::String="trajectory_animation.gif",
  n_time_steps::Int=25,
  fps::Int=5,
  xlims::Union{Nothing,Tuple{Float64,Float64}}=nothing,
  ylims::Union{Nothing,Tuple{Float64,Float64}}=nothing
)
  println("+"^60)
  println(" WORKFLOW ANIMACJI TRAJEKTORII")
  println("+"^60)

  df_raw = generate_trajectory_data(
    ic_filepath,
    n_time_steps=n_time_steps
  )

  settings = modHydroSim.load_settings(ic_filepath)
  tau_0 = settings.tspan[1]

  df_scaled_for_plot = DataFrame(
      tau = df_raw.tau,
      Run_ID = df_raw.Run_ID,
      T_0_MeV = df_raw.T_0 ./ modHydroSim.MeV,
      A_0 = df_raw.A_0,
      plot_x_axis = tau_0 .* df_raw.T,
      plot_y_axis = (tau_0^2) .* df_raw.dTdtau
  )

  auto_xlims = if xlims === nothing && !isempty(df_scaled_for_plot.plot_x_axis)
    min_x, max_x = minimum(df_scaled_for_plot.plot_x_axis), maximum(df_scaled_for_plot.plot_x_axis)
    padding_x = (max_x - min_x) * 0.05
    (min_x - padding_x, max_x + padding_x)
  elseif isempty(df_scaled_for_plot.plot_x_axis)
    (0.0, 1.0)
  else
    xlims
  end

  auto_ylims = if ylims === nothing && !isempty(df_scaled_for_plot.plot_y_axis)
    min_y, max_y = minimum(df_scaled_for_plot.plot_y_axis), maximum(df_scaled_for_plot.plot_y_axis)
    padding_y = (max_y - min_y) * 0.05
    (min_y - padding_y, max_y + padding_y)
  elseif isempty(df_scaled_for_plot.plot_y_axis)
    (0.0, 1.0)
  else
    ylims
  end


  println("\n--- Generowanie animacji ---")
  modPlots.test_animation(
    df_scaled_for_plot;
    output_gif=output_gif,
    fps=fps,
    xlims=auto_xlims,
    ylims=auto_ylims
  )

  println("\n✅ Zakończono
 workflow animacji trajektorii")

  return df_scaled_for_plot
end

end

end
