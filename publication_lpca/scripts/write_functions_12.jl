function write_mis_docs(dir::AbstractString, taus, configs)
    # 1. README.md
    open(joinpath(dir, "README.md"), "w") do io
        write(
            io, """# Conformal MIS Phase Space Figures (2D: T, A)

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
            """
        )
    end

    # 2. FIGURES_INFO.tex
    return open(joinpath(dir, "FIGURES_INFO.tex"), "w") do io
        write(
            io, raw"""% Szablony LaTeX dla przestrzeni fazowej Conformal MIS

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
            """
        )
    end
end

function write_hjsw_docs(dir::AbstractString, taus, configs)
    # 1. README.md
    open(joinpath(dir, "README.md"), "w") do io
        write(
            io, """# HJSW Phase Space Figures (3D: T, A, B)

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
            """
        )
    end

    # 2. FIGURES_INFO.tex
    return open(joinpath(dir, "FIGURES_INFO.tex"), "w") do io
        write(
            io, raw"""% Szablony LaTeX dla przestrzeni fazowej HJSW (3D)

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
            """
        )
    end
end
