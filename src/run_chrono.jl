using CairoMakie
using Dates
using Random
using Statistics

module HoloRunner
    using CairoMakie
    using Dates

    const SRC_DIR = @__DIR__

    println(">>> [1/5] Ładowanie modułów fizycznych...")

    include(joinpath(SRC_DIR, "lib.jl"))

    include(joinpath(SRC_DIR, "pca.jl"))

    include(joinpath(SRC_DIR, "chrono.jl"))

    using .modHydroSim
    using .modChromoPlots
    using .modPCA

    export run_holographic_analysis

    function run_holographic_analysis()
        start_time = now()
        println("\n>>> START ANALIZY HOLOGRAFICZNEJ: $start_time")

        # --- A. Konfiguracja Folderów ---
        timestamp = Dates.format(now(), "yyyy-mm-dd_HHMM")
        output_dir = joinpath(SRC_DIR, "..", "plots", "holography_$timestamp")
        mkpath(output_dir)
        println(">>> Katalog wyjściowy: $output_dir")

        # Ustawiamy styl publikacyjny
        modChromoPlots.set_publication_theme()

        # --- B. Część Teoretyczna (Analityczna) ---
        println(">>> [2/5] Generowanie wykresów analitycznych (Theory)...")

        # Wykres 1: Struktura Atraktora (Hydro vs Holo Mode)
        # Pokazuje jak mody o wymiarze Delta = 3 (mezony) zanikają na tle hydro
        fig_theory = modChromoPlots.plot_analytical_transeries(
            params=modChromoPlots.DEFAULT_HOLO_PARAMS,
            w_range=(0.2, 8.0)
        )
        save(joinpath(output_dir, "01_theory_structure_Delta3.png"), fig_theory)

        fig_scaling = modChromoPlots.plot_holographic_scaling(
            sigmas=[1.0, 5.0, 20.0]
        )
        save(joinpath(output_dir, "02_holographic_scaling.png"), fig_scaling)

        println(">>> [3/5] Uruchamianie symulacji numerycznej (:Holo)...")

        try
            settings = modHydroSim.SimSettings(
                theory=:Holo,      # Używamy nowej teorii zdefiniowanej w lib.jl
                n_points=30,       # 30 trajektorii wystarczy do wizualizacji
                tspan=(0.2, 5.0),
                T_range=(0.3, 0.6), # Temperatury typowe dla QGP [GeV]
                A_range=(-2.0, 6.0) # Szerszy zakres anizotropii
            )

            sim_results = modHydroSim.run_simulation(settings=settings)

            println(">>> [4/5] Generowanie wykresów numerycznych...")

            # Wykres 3: Zbieżność do atraktora (Numerics vs Analytics)
            fig_conv = modChromoPlots.plot_chromo_attractor_convergence(
                sim_results,
                show_theory=true
            )
            save(joinpath(output_dir, "03_numerical_convergence.png"), fig_conv)

            # Wykres 4: Przestrzeń fazowa (A vs Phi) - Snapshots
            # Sprawdzamy czy mamy pole Phi w wynikach (u[3])
            if length(sim_results.solutions[1].u[1]) >= 3
                fig_grid = modChromoPlots.plot_chromo_phase_space_grid(
                    sim_results,
                    [0.4, 1.0, 3.0], # Czasy snapshots
                    x_key=:A,
                    y_key=:phi,      # Oś Y to pole holograficzne
                    color_by=:T
                )
                save(joinpath(output_dir, "04_phase_space_evolution.png"), fig_grid)
            end

        catch e
            println("!!! OSTRZEŻENIE: Nie udało się uruchomić symulacji numerycznej.")
            println("!!! Powód: $e")
            println("!!! (Czy zaktualizowałeś plik src/lib.jl o teorię :Holo?)")
        end

        # --- D. Podsumowanie ---
        println(">>> [5/5] Zakończono pomyślnie.")
        println(">>> Wyniki zapisano w: $output_dir")

        return output_dir
    end
end

HoloRunner.run_holographic_analysis()


