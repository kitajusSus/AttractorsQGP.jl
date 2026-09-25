"""
    Script 12: Dedicated Phase Space Generator for Publication
    Produces separated directories for MIS and HJSW:
    - publication_lpca/plots/phase_space/mis/
    - publication_lpca/plots/phase_space/hjsw/

    Generates:
    1. Standalone single-slice phase space plots for each tau in [0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80, 1.00, 2.50]:
       - Discrete local dimension dims()
       - Continuous participation ratio dimension pr()
       - Across standardizations: zscore, minmax, max, physical, dimensionless
       - Title on top tau = ... fm/c in large, clear font (titlesize = 24)
       - Subsampled: 10,000 points calculated, 5,000 points plotted
    2. Comprehensive 3x3 grids with wide subplot gaps (55px) and enlarged tau titles
    3. Complete LaTeX and Markdown documentation (README.md and FIGURES_INFO.tex) in each folder.
"""

using AttractorsQGP
using CairoMakie
using LaTeXStrings
using Printf
using Statistics

if !isdefined(Main, :PublicationLPCA)
    include(joinpath(@__DIR__, "..", "src", "PublicationLPCA.jl"))
end
using .PublicationLPCA

function run_phase_space_generator(;
        mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
        hjsw_dataset_path::AbstractString = isfile(joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_10000_points.h5")) ?
            joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_10000_points.h5") :
            joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
        base_output_dir::AbstractString = joinpath(@__DIR__, "..", "plots", "phase_space")
    )
    mis_dir = joinpath(base_output_dir, "mis")
    hjsw_dir = joinpath(base_output_dir, "hjsw")
    mkpath(mis_dir)
    mkpath(hjsw_dir)
    CairoMakie.activate!()

    println("=== Starting 12_phase_space_publication_generator ===")

    grid_taus = Float64[0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80, 1.00, 2.50]
    k_eval = 24
    tol_eval = 0.01
    colormap_choice = :managua100

    mis_configs = [
        (name = "zscore", method = :zscore, return_norm = true, xlabel = L"T_z", ylabel = L"\mathcal{A}_z", dataset_type = :physical),
        (name = "minmax", method = :minmax, return_norm = true, xlabel = L"T_{\mathrm{minmax}}", ylabel = L"\mathcal{A}_{\mathrm{minmax}}", dataset_type = :physical),
        (name = "max", method = :max, return_norm = true, xlabel = L"T / T_{\mathrm{max}}", ylabel = L"\mathcal{A} / \mathcal{A}_{\mathrm{max}}", dataset_type = :physical),
        (name = "physical", method = :none, return_norm = false, xlabel = L"T\,[\mathrm{fm}^{-1}]", ylabel = L"\mathcal{A}", dataset_type = :physical),
        (name = "dimensionless", method = :max, return_norm = false, xlabel = L"w = \tau T", ylabel = L"\mathcal{A}", dataset_type = :dimensionless),
    ]

    hjsw_configs = [
        (name = "zscore", method = :zscore, return_norm = true, xlabel = L"T_z", ylabel = L"\mathcal{A}_z", zlabel = L"\mathcal{B}_z", dataset_type = :physical),
        (name = "minmax", method = :minmax, return_norm = true, xlabel = L"T_{\mathrm{minmax}}", ylabel = L"\mathcal{A}_{\mathrm{minmax}}", zlabel = L"\mathcal{B}_{\mathrm{minmax}}", dataset_type = :physical),
        (name = "max", method = :max, return_norm = true, xlabel = L"T / T_{\mathrm{max}}", ylabel = L"\mathcal{A} / \mathcal{A}_{\mathrm{max}}", zlabel = L"\mathcal{B} / \mathcal{B}_{\mathrm{max}}", dataset_type = :physical),
        (name = "physical", method = :none, return_norm = false, xlabel = L"T\,[\mathrm{MeV}]", ylabel = L"\mathcal{A}", zlabel = L"\mathcal{B}", dataset_type = :physical),
        (name = "dimensionless", method = :max, return_norm = false, xlabel = L"w = \tau T", ylabel = L"\mathcal{A}", zlabel = L"\mathcal{B}", dataset_type = :dimensionless),
    ]

    # =========================================================================
    # Part 1: Conformal MIS Model (2D Phase Space)
    # =========================================================================
    println("\n[1/2] Processing Conformal MIS Phase Space (plots/phase_space/mis/)...")
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)

    for cfg in mis_configs
        ds = cfg.dataset_type == :dimensionless ? mis_variants.dimensionless : mis_variants.physical
        println("  - Generating MIS [$(cfg.name)]...")

        # 1A. Full 3x3 Grids
        fig_grid_dims = plot_colored_phase_space_grid_2d(
            ds,
            grid_taus;
            feature_indices = [2, 3],
            x_label = cfg.xlabel,
            y_label = cfg.ylabel,
            k_neighbor = k_eval,
            tolerance = tol_eval,
            normalize_method = cfg.method,
            return_normalized = cfg.return_norm,
            colormap = colormap_choice,
            n_calc = 10_000,
            n_plot = 5_000
        )
        save(joinpath(mis_dir, "grid_mis_$(cfg.name)_dims.pdf"), fig_grid_dims)

        fig_grid_pr = plot_pr_phase_space_grid_2d(
            ds,
            grid_taus;
            feature_indices = [2, 3],
            x_label = cfg.xlabel,
            y_label = cfg.ylabel,
            colormap = colormap_choice,
            color_limits = (1.0, 2.0),
            normalize_method = cfg.method,
            return_normalized = cfg.return_norm,
            k = k_eval,
            n_calc = 10_000,
            n_plot = 5_000
        )
        save(joinpath(mis_dir, "grid_mis_$(cfg.name)_pr.pdf"), fig_grid_pr)

        # 1B. Individual Single Slices for each tau
        for tau in grid_taus
            tau_str = @sprintf("%.2f", tau)
            slice_data = compute_pointwise_dimensions(
                ds,
                tau;
                feature_indices = [2, 3],
                k_neighbor = k_eval,
                tolerance = tol_eval,
                normalize_method = cfg.method,
                return_normalized = cfg.return_norm,
                max_points = 10_000
            )

            fig_slice_dims = plot_colored_phase_space_slice_2d(
                slice_data;
                x_col_idx = 1,
                y_col_idx = 2,
                x_label = cfg.xlabel,
                y_label = cfg.ylabel,
                colormap = colormap_choice,
                n_plot = 5_000
            )
            save(joinpath(mis_dir, "mis_$(cfg.name)_slice_tau_$(tau_str)_dims.pdf"), fig_slice_dims)

            fig_slice_pr = plot_pr_phase_space_slice_2d(
                ds,
                tau;
                feature_indices = [2, 3],
                x_label = cfg.xlabel,
                y_label = cfg.ylabel,
                colormap = colormap_choice,
                color_limits = (1.0, 2.0),
                normalize_method = cfg.method,
                return_normalized = cfg.return_norm,
                k = k_eval,
                n_calc = 10_000,
                n_plot = 5_000
            )
            save(joinpath(mis_dir, "mis_$(cfg.name)_slice_tau_$(tau_str)_pr.pdf"), fig_slice_pr)
        end
    end

    # Write documentation for MIS folder
    write_mis_docs(mis_dir, grid_taus, mis_configs)

    # =========================================================================
    # Part 2: HJSW Model (3D Phase Space)
    # =========================================================================
    println("\n[2/2] Processing HJSW Phase Space (plots/phase_space/hjsw/)...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)

    for cfg in hjsw_configs
        ds = cfg.dataset_type == :dimensionless ? hjsw_variants.dimensionless : hjsw_variants.physical
        println("  - Generating HJSW [$(cfg.name)]...")

        # 2A. Full 3x3 Grids
        fig_grid_dims = plot_colored_phase_space_grid_3d(
            ds,
            grid_taus;
            feature_indices = [2, 3, 4],
            x_label = cfg.xlabel,
            y_label = cfg.ylabel,
            z_label = cfg.zlabel,
            k_neighbor = k_eval,
            tolerance = tol_eval,
            normalize_method = cfg.method,
            return_normalized = cfg.return_norm,
            colormap = colormap_choice,
            n_calc = 10_000,
            n_plot = 5_000
        )
        save(joinpath(hjsw_dir, "grid_hjsw_$(cfg.name)_dims.pdf"), fig_grid_dims)

        fig_grid_pr = plot_pr_phase_space_grid_3d(
            ds,
            grid_taus;
            feature_indices = [2, 3, 4],
            x_label = cfg.xlabel,
            y_label = cfg.ylabel,
            z_label = cfg.zlabel,
            colormap = colormap_choice,
            color_limits = (1.0, 3.0),
            normalize_method = cfg.method,
            return_normalized = cfg.return_norm,
            k = k_eval,
            n_calc = 10_000,
            n_plot = 5_000
        )
        save(joinpath(hjsw_dir, "grid_hjsw_$(cfg.name)_pr.pdf"), fig_grid_pr)

        # 2B. Individual Single Slices for each tau
        for tau in grid_taus
            tau_str = @sprintf("%.2f", tau)
            slice_data = compute_pointwise_dimensions(
                ds,
                tau;
                feature_indices = [2, 3, 4],
                k_neighbor = k_eval,
                tolerance = tol_eval,
                normalize_method = cfg.method,
                return_normalized = cfg.return_norm,
                max_points = 10_000
            )

            fig_slice_dims = plot_colored_phase_space_slice_hjsw_3d(
                slice_data;
                x_label = cfg.xlabel,
                y_label = cfg.ylabel,
                z_label = cfg.zlabel,
                colormap = colormap_choice,
                n_plot = 5_000
            )
            save(joinpath(hjsw_dir, "hjsw_$(cfg.name)_slice_tau_$(tau_str)_dims.pdf"), fig_slice_dims)

            fig_slice_pr = plot_pr_phase_space_slice_3d(
                ds,
                tau;
                feature_indices = [2, 3, 4],
                x_label = cfg.xlabel,
                y_label = cfg.ylabel,
                z_label = cfg.zlabel,
                colormap = colormap_choice,
                color_limits = (1.0, 3.0),
                normalize_method = cfg.method,
                return_normalized = cfg.return_norm,
                k = k_eval,
                n_calc = 10_000,
                n_plot = 5_000
            )
            save(joinpath(hjsw_dir, "hjsw_$(cfg.name)_slice_tau_$(tau_str)_pr.pdf"), fig_slice_pr)
        end
    end

    # Write documentation for HJSW folder
    write_hjsw_docs(hjsw_dir, grid_taus, hjsw_configs)

    println("=== 12_phase_space_publication_generator Completed Successfully! ===")
end

function write_mis_docs(dir::AbstractString, taus, configs)
    # 1. README.md
    open(joinpath(dir, "README.md"), "w") do io
        write(io, """# Conformal MIS Phase Space Figures (2D: T, A)

Ten katalog zawiera wygenerowane wektorowe wykresy PDF przestrzeni fazowej modelu Conformal MIS.
Wykresy zostały podzielone na pojedyncze przekroje czasowe oraz zbiorcze siatki 3x3, umożliwiając łatwe składanie własnych tabel lub figur w LaTeX.

## Konwencja nazewnictwa plików
- **Pojedynczy przekrój:** `mis_{standaryzacja}_slice_tau_{tau}_{miara}.pdf`
  - Np. `mis_zscore_slice_tau_0.25_dims.pdf`
  - Np. `mis_zscore_slice_tau_0.25_pr.pdf`
- **Siatka 3x3 (9 przekrojów):** `grid_mis_{standaryzacja}_{miara}.pdf`
  - Np. `grid_mis_zscore_dims.pdf`

## Standaryzacje:
1. `zscore` – Standaryzacja Z-Score (rekomendowana, jednostki wariancji)
2. `minmax` – Skalowanie Min-Max do [0, 1]
3. `max` – Skalowanie Abs-Max do [-1, 1]
4. `physical` – Surowe jednostki fizyczne (T w fm^-1, A)
5. `dimensionless` – Współrzędne bezwymiarowe w = tau*T

## Miary:
- `dims`: Dyskretny wymiar lokalny (kropki żółte: d=1, błękitne: d=2)
- `pr`: Ciągły Participation Ratio d_PR (mapa Managua od 1.0 do 2.0)

## Punkty:
- Do obliczeń LPCA (k-NN, k=24): 10 000 punktów.
- Do wizualizacji scatter: dokładnie 5 000 reprezentatywnych punktów.
""")
    end

    # 2. FIGURES_INFO.tex
    open(joinpath(dir, "FIGURES_INFO.tex"), "w") do io
        write(io, raw"""% Szablony LaTeX dla przestrzeni fazowej Conformal MIS

% Przykład 1: Pojedynczy przekrój tau = 0.25 fm/c (dims vs pr)
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{phase_space/mis/mis_zscore_slice_tau_0.25_dims.pdf}
        \caption{MIS: \emph{dims} ($\tau = 0.25\,\mathrm{fm}/c$)}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{phase_space/mis/mis_zscore_slice_tau_0.25_pr.pdf}
        \caption{MIS: \emph{pr} ($\tau = 0.25\,\mathrm{fm}/c$)}
    \end{subfigure}
    \caption{Przestrzeń fazowa Conformal MIS w chwili $\tau = 0.25\,\mathrm{fm}/c$ w standaryzacji Z-Score.}
    \label{fig:mis_phase_space_tau_025}
\end{figure}

% Przykład 2: Zestawienie 3 wybranych chwil czasowych obok siebie
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\linewidth]{phase_space/mis/mis_zscore_slice_tau_0.20_dims.pdf}
        \caption{$\tau = 0.20\,\mathrm{fm}/c$}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\linewidth]{phase_space/mis/mis_zscore_slice_tau_0.45_dims.pdf}
        \caption{$\tau = 0.45\,\mathrm{fm}/c$}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\linewidth]{phase_space/mis/mis_zscore_slice_tau_2.50_dims.pdf}
        \caption{$\tau = 2.50\,\mathrm{fm}/c$}
    \end{subfigure}
    \caption{Ewolucja i kolaps rozmaitości $2\mathrm{D} \to 1\mathrm{D}$ w modelu MIS dla $\tau \in \{0.20, 0.45, 2.50\}\,\mathrm{fm}/c$.}
    \label{fig:mis_3slices}
\end{figure}
""")
    end
end

function write_hjsw_docs(dir::AbstractString, taus, configs)
    # 1. README.md
    open(joinpath(dir, "README.md"), "w") do io
        write(io, """# HJSW Phase Space Figures (3D: T, A, B)

Ten katalog zawiera wygenerowane wektorowe wykresy 3D PDF przestrzeni fazowej modelu HJSW.
Wykresy zostały podzielone na pojedyncze przekroje czasowe oraz zbiorcze siatki 3x3, umożliwiając łatwe składanie własnych tabel lub figur w LaTeX.

## Konwencja nazewnictwa plików
- **Pojedynczy przekrój 3D:** `hjsw_{standaryzacja}_slice_tau_{tau}_{miara}.pdf`
  - Np. `hjsw_zscore_slice_tau_0.25_dims.pdf`
  - Np. `hjsw_zscore_slice_tau_0.25_pr.pdf`
- **Siatka 3x3 (9 przekrojów 3D):** `grid_hjsw_{standaryzacja}_{miara}.pdf`
  - Np. `grid_hjsw_zscore_dims.pdf`

## Standaryzacje:
1. `zscore` – Standaryzacja Z-Score (rekomendowana, jednostki wariancji)
2. `minmax` – Skalowanie Min-Max do [0, 1]
3. `max` – Skalowanie Abs-Max do [-1, 1]
4. `physical` – Surowe jednostki fizyczne (T w MeV, A, B)
5. `dimensionless` – Współrzędne bezwymiarowe w = tau*T

## Miary:
- `dims`: Dyskretny wymiar lokalny (kropki: granatowe d=3, błękitne d=2, złote d=1)
- `pr`: Ciągły Participation Ratio d_PR (mapa Managua od 1.0 do 3.0)

## Punkty:
- Do obliczeń LPCA (k-NN, k=24): 10 000 punktów.
- Do wizualizacji scatter 3D: dokładnie 5 000 reprezentatywnych punktów.
""")
    end

    # 2. FIGURES_INFO.tex
    open(joinpath(dir, "FIGURES_INFO.tex"), "w") do io
        write(io, raw"""% Szablony LaTeX dla przestrzeni fazowej HJSW (3D)

% Przykład 1: Pojedynczy przekrój 3D dla tau = 0.25 fm/c (dims vs pr)
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{phase_space/hjsw/hjsw_zscore_slice_tau_0.25_dims.pdf}
        \caption{HJSW: \emph{dims} ($\tau = 0.25\,\mathrm{fm}/c$)}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{phase_space/hjsw/hjsw_zscore_slice_tau_0.25_pr.pdf}
        \caption{HJSW: \emph{pr} ($\tau = 0.25\,\mathrm{fm}/c$)}
    \end{subfigure}
    \caption{Przestrzeń fazowa HJSW 3D w chwili $\tau = 0.25\,\mathrm{fm}/c$ ukazująca powstawanie 2-wymiarowej rozmaitości pośredniej.}
    \label{fig:hjsw_phase_space_tau_025}
\end{figure}

% Przykład 2: Zestawienie 3 etapów kolapsu hierarchicznego 3 -> 2 -> 1
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\linewidth]{phase_space/hjsw/hjsw_zscore_slice_tau_0.20_dims.pdf}
        \caption{$\tau = 0.20\,\mathrm{fm}/c$ ($3\mathrm{D}$)}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\linewidth]{phase_space/hjsw/hjsw_zscore_slice_tau_0.35_dims.pdf}
        \caption{$\tau = 0.35\,\mathrm{fm}/c$ ($2\mathrm{D}$)}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\linewidth]{phase_space/hjsw/hjsw_zscore_slice_tau_2.50_dims.pdf}
        \caption{$\tau = 2.50\,\mathrm{fm}/c$ ($1\mathrm{D}$)}
    \end{subfigure}
    \caption{Hierarchiczny kolaps przestrzeni fazowej HJSW $3\mathrm{D} \to 2\mathrm{D} \to 1\mathrm{D}$ w standaryzacji Z-Score.}
    \label{fig:hjsw_hierarchical_collapse_3slices}
\end{figure}
""")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_phase_space_generator()
end
