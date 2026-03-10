
# API Reference


## Modele i Parametry Transportu

Definicje abstrakcyjnych i konkretnych modeli hydrodynamicznych stosowanych w symulacjach.

```@docs
AtractorsQGP.AbstractHydroModel
AtractorsQGP.HydroParams
AtractorsQGP.BRSSSModel
AtractorsQGP.MISModel

```

## Symulacja i Rozwiązywanie Równań

Główne funkcje odpowiedzialne za generowanie warunków początkowych, rozwiązywanie równań hydrodynamiki

```@docs
AtractorsQGP.run_main
AtractorsQGP.generate_initial_conditions
AtractorsQGP.solve_hydro
AtractorsQGP.generate_trajectories
AtractorsQGP.build_dataset

```

## Analiza Danych i PCA

```@docs
AtractorsQGP.run_pca
AtractorsQGP.run_pca_kernel
AtractorsQGP.run_pca_per_time
AtractorsQGP.run_pca_for_tau
AtractorsQGP.run_evolution_pca_workflow
AtractorsQGP.estimate_dimension
AtractorsQGP.explained_variance_ratio_from_svd

```

## Wizualizacja

Funkcje do tworzenia wykresów

```@docs
AtractorsQGP.set_publication_theme
AtractorsQGP.plot_thermodynamics_evolution
AtractorsQGP.plot_pca_evr_over_time
AtractorsQGP.plot_phase_space_grid
AtractorsQGP.plot_pca_summary

```

## Operacje Wejścia/Wyjścia (I/O)

Funkcje do zapisu i odczytu danych w formatach CSV, HDF5 oraz JLS.

```@docs
AtractorsQGP.save_dataset
AtractorsQGP.load_dataset
AtractorsQGP.save_dataset_csv
AtractorsQGP.load_dataset_csv
AtractorsQGP.save_dataset_h5
AtractorsQGP.load_dataset_h5

```

## Stałe i Jednostki

Stałe fizyczne oraz narzędzia do konwersji temperatury między `fm⁻¹` a `MeV`.

```@docs
AtractorsQGP.HBARC_MEV_FM
AtractorsQGP.to_temperature_unit
AtractorsQGP.temperature_to_fm

```

## Pozostałe Funkcje

Wszystkie pozostałe funkcje wewnętrzne i pomocnicze.

```@autodocs
Modules = [AtractorsQGP]
Order   = [:type, :function]

```

