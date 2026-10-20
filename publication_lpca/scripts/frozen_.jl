"""
    frozen_.jl - Funkcja i skrypt do badania Local PCA z zamrożonymi skalami (Frozen Scales)

Użycie w REPL:
    include("publication_lpca/scripts/frozen_.jl")
    wyniki_mis = frozen(:mis)
    wyniki_hjsw = frozen(:hjsw)

Można też przekazywać własne parametry:
    frozen(:mis; K_NEIGHBOR = 24, SAVE_FORMAT = :pdf)
"""

using AttractorsQGP
using Statistics
using LinearAlgebra
using NearestNeighbors
using Printf
using CairoMakie
CairoMakie.activate!()

function frozen(
        MODEL_NAME::Symbol = :mis;
        dataset_path::Union{Nothing, AbstractString} = nothing,
        K_NEIGHBOR::Integer = 16,
        K_SWEEP_RANGE = [4, 6, 8, 12, 16, 20, 24, 32, 48, 64],
        TAU_SLICES_PLOT = [0.2, 0.35, 0.55, 0.85, 1.5, 2.5],
        GENERATE_PLOTS::Bool = true,
        SAVE_FORMAT::Symbol = :pdf,          # :png lub :pdf
        OUTPUT_PLOT_DIR::String = "plots/frozen_lpca"
    )

    CairoMakie.activate!()
    set_publication_style(; cmap = lpca_style, fontsize = 18)

    fname = MODEL_NAME === :mis ? "mis_matched_seed5.h5" : "hjsw_matched_seed5.h5"
    DATASET_PATH = if dataset_path !== nothing
        dataset_path
    else
        possible_paths = [
            joinpath("datasets", "hjsw_lpca", fname),
            joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", fname),
            joinpath(@__DIR__, "datasets", "hjsw_lpca", fname)
        ]
        data_idx = findfirst(isfile, possible_paths)
        if isnothing(data_idx)
            error("Nie znaleziono pliku $fname w folderze datasets/hjsw_lpca/")
        end
        possible_paths[data_idx]
    end

    println("="^75)
    println(" ANALIZA LOCAL PCA Z ZAMROŻONYMI SKALAMI POCZĄTKOWYMI (FROZEN INITIAL SCALES)")
    println("="^75)
    println("Model:            ", MODEL_NAME)
    println("Ścieżka do danych: ", DATASET_PATH)

    println("\nWczytywanie zbioru danych...")
    dataset = load_dataset(DATASET_PATH)
    n_rows, n_cols = size(dataset)
    all_taus = sort(unique(dataset[:, 1]))
    t0 = all_taus[1]
    t_max = all_taus[end]
    feature_indices = collect(2:n_cols)
    D = length(feature_indices)

    println(
        @sprintf(
            "Wczytano: %d wierszy, %d cech (D = %d), %d przekrojów tau od %.2f do %.2f fm/c.",
            n_rows, n_cols, D, length(all_taus), t0, t_max
        )
    )

    # KROK 1: Obliczenie zamrożonych skal
    println("\n" * "-"^75)
    println("KROK 1: Obliczanie skal początkowych x_j = std(X_j(t0)) w chwili t0 = $t0")
    println("-"^75)

    scales = compute_initial_scales(dataset; tau_col = 1, feature_cols = feature_indices)
    _, X0 = get_tau_slice(dataset, t0; feature_cols = feature_indices)
    Z0 = apply_frozen_scales(X0, scales)

    col_names = D == 2 ? ["T [fm^-1]", "A"] : ["T [MeV]", "A", "B"]
    for j in 1:D
        name = j <= length(col_names) ? col_names[j] : "X_$j"
        @printf(
            "  Cecha %-10s: średnia = %9.4f,  skala x_%d = std = %9.4f  ==>  Var po skalowaniu = %.6f\n",
            name, mean(X0[:, j]), j, scales[j], var(Z0[:, j])
        )
    end

    # KROK 2: Analiza w t0 i t_max
    println("\n" * "-"^75)
    println("KROK 2: Weryfikacja wymiaru d_PR dla t0 = $t0 oraz chwili późnej t = $t_max (k = $K_NEIGHBOR)")
    println("-"^75)

    res_t0 = lpca_frozen_slice(dataset, t0, scales; k = K_NEIGHBOR, feature_cols = feature_indices)
    @printf("Chwila początkowa tau = %.2f fm/c:\n", t0)
    @printf("  Średni wymiar d_PR = %.4f ± %.4f (teoria: D = %d)\n", res_t0.mean_dpr, res_t0.std_dpr, D)
    @printf("  Stosunek spektralny lambda_min / lambda_max = %.4e\n", res_t0.spectral_ratio)

    res_late = lpca_frozen_slice(dataset, t_max, scales; k = K_NEIGHBOR, feature_cols = feature_indices)
    @printf("\nChwila asymptotyczna tau = %.2f fm/c:\n", t_max)
    @printf("  Średni wymiar d_PR = %.4f ± %.4f (teoria: 1.0 dla atraktora)\n", res_late.mean_dpr, res_late.std_dpr)
    @printf("  Stosunek spektralny lambda_min / lambda_max = %.4e (zanik poprzeczny)\n", res_late.spectral_ratio)

    # KROK 3: Badanie plateau w funkcji k
    println("\n" * "-"^75)
    println("KROK 3: Poszukiwanie plateau w funkcji liczby sąsiadów k (k-sweep)")
    println("-"^75)

    sweep_t0 = k_sweep_frozen(dataset, t0, scales; k_range = K_SWEEP_RANGE, feature_cols = feature_indices)
    sweep_late = k_sweep_frozen(dataset, t_max, scales; k_range = K_SWEEP_RANGE, feature_cols = feature_indices)

    println("   k  |  d_PR(t0 = $t0)  |  d_PR(t = $t_max)  | Stosunek spektralny (t_max)")
    println("------+-------------------+--------------------+----------------------------")
    for (idx, k_val) in enumerate(K_SWEEP_RANGE)
        @printf(
            "  %3d |  %.4f ± %.4f  |  %.4f ± %.4f  |  %.4e\n",
            k_val,
            sweep_t0.mean_dpr[idx], sweep_t0.std_dpr[idx],
            sweep_late.mean_dpr[idx], sweep_late.std_dpr[idx],
            sweep_late.mean_ratios[idx]
        )
    end

    # KROK 4: Skan ewolucji w czasie
    println("\n" * "-"^75)
    println("KROK 4: Ewolucja wymiaru d_PR(tau) wzdłuż relaksacji hydrodynamicznej")
    println("-"^75)

    step_stride = max(1, div(length(all_taus), 15))
    selected_scan_taus = all_taus[1:step_stride:end]
    if selected_scan_taus[end] != all_taus[end]
        push!(selected_scan_taus, all_taus[end])
    end

    scan_res = scan_lpca_frozen(dataset, scales; tau_values = selected_scan_taus, k = K_NEIGHBOR, feature_cols = feature_indices)

    println("  tau [fm/c] |  Średni d_PR  |   std(d_PR)  | lambda_min / lambda_max")
    println("-------------+---------------+--------------+------------------------")
    for (i, τ) in enumerate(scan_res.tau_values)
        @printf(
            "    %6.2f   |    %7.4f    |    %7.4f   |       %9.3e\n",
            τ, scan_res.mean_dims[i], scan_res.std_dims[i], scan_res.spectral_ratios[i]
        )
    end

    # KROK 5: Wykresy
    figs = nothing
    if GENERATE_PLOTS
        println("\n" * "-"^75)
        println("KROK 5: Generowanie wykresów przestrzeni fazowej z motywem publikacyjnym i lpca_style...")
        println("-"^75)
        mkpath(OUTPUT_PLOT_DIR)
        ext = string(SAVE_FORMAT)

        valid_plot_taus = filter(τ -> any(isapprox.(all_taus, τ; atol = 0.05)), TAU_SLICES_PLOT)
        if isempty(valid_plot_taus)
            valid_plot_taus = all_taus[round.(Int, range(1, length(all_taus), length = 6))]
        end

        fig_grid_phys = plot_frozen_phase_space_grid(
            dataset, valid_plot_taus, scales;
            k = K_NEIGHBOR, feature_indices = feature_indices, scaled_coords = false,
            colormap = lpca_style
        )
        save(joinpath(OUTPUT_PLOT_DIR, "$(MODEL_NAME)_phase_space_grid_physical.$ext"), fig_grid_phys)

        fig_grid_scaled = plot_frozen_phase_space_grid(
            dataset, valid_plot_taus, scales;
            k = K_NEIGHBOR, feature_indices = feature_indices, scaled_coords = true,
            colormap = lpca_style
        )
        save(joinpath(OUTPUT_PLOT_DIR, "$(MODEL_NAME)_phase_space_grid_scaled.$ext"), fig_grid_scaled)

        taus_for_k_plot = [valid_plot_taus[1], valid_plot_taus[max(1, div(length(valid_plot_taus), 2))], valid_plot_taus[end]]
        fig_k_sweep = plot_frozen_k_sweep(
            dataset, taus_for_k_plot, scales;
            k_range = K_SWEEP_RANGE, feature_indices = feature_indices
        )
        save(joinpath(OUTPUT_PLOT_DIR, "$(MODEL_NAME)_k_plateau_analysis.$ext"), fig_k_sweep)

        fig_evo = Figure(size = (850, 500))
        ax_evo = Axis(
            fig_evo[1, 1],
            title = L"\text{Ewolucja lokalnego wymiaru } d_{\text{PR}}(\tau) \text{ w czasie (%$MODEL_NAME)}",
            xlabel = L"\text{Czas własny } \tau\,[\text{fm}/c]",
            ylabel = L"\langle d_{\text{PR}} \rangle"
        )
        main_color = AttractorsQGP.LPCA_COLORS[4]
        lines!(ax_evo, scan_res.tau_values, scan_res.mean_dims, color = main_color, linewidth = 2.8, label = L"\text{Lokalne PCA } (k = %$(K_NEIGHBOR))")
        scatter!(ax_evo, scan_res.tau_values, scan_res.mean_dims, color = main_color, markersize = 9)
        hlines!(ax_evo, [1.0], color = :gray50, linestyle = :dash, linewidth = 1.5, label = L"\text{Atraktor 1D } (d = 1)")
        hlines!(ax_evo, [Float64(D)], color = :black, linestyle = :dot, linewidth = 1.5, label = L"\text{Pełna przestrzeń } (d = %$D)")
        axislegend(ax_evo, position = :rt)
        save(joinpath(OUTPUT_PLOT_DIR, "$(MODEL_NAME)_dimension_evolution_tau.$ext"), fig_evo)

        println("  [OK] Zapisano komplet wykresów w: $OUTPUT_PLOT_DIR")
        figs = (grid_phys = fig_grid_phys, grid_scaled = fig_grid_scaled, k_sweep = fig_k_sweep, evolution = fig_evo)
    end

    println("\n" * "="^75)
    println(" ZAKOŃCZONO SUKCESEM!")
    println("="^75)

    return (
        model = MODEL_NAME,
        scales = scales,
        res_t0 = res_t0,
        res_late = res_late,
        sweep_t0 = sweep_t0,
        sweep_late = sweep_late,
        scan = scan_res,
        figures = figs,
    )
end
