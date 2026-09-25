# Przewodnik po Wykresach Publikacyjnych (PDF) i Gotowe Kody LaTeX
**Data aktualizacji:** 25 września 2026  
**Lokalizacja plików wektorowych:** [`publication_lpca/plots/`](file:///home/kitajusus/github/Atractors-in-QGP/publication_lpca/plots/)  
**Format wyjściowy:** Wyłącznie wektorowy **`.pdf`** (**bez tytułów wewnątrz wykresów** – tytuły i opisy znajdują się w podpisach LaTeX `\caption{...}`).

---

## Spis Treści
1. [Wprowadzenie i Standardy Publikacyjne](#1-wprowadzenie-i-standardy-publikacyjne)
2. [Kategoria I: Siatki ewolucji przestrzeni fazowej 2D i 3D (`best_practices_lpca/`)](#2-kategoria-i-siatki-ewolucji-przestrzeni-fazowej-2d-i-3d)
   - [Model Conformal MIS (2D)](#a-model-conformal-mis-2d)
   - [Model HJSW (3D)](#b-model-hjsw-3d)
   - [Wielometodowe nałożenia ewolucji $\langle d \rangle(\tau)$](#c-wielometodowe-nałożenia-ewolucji-średniego-wymiaru-langle-d-rangletau)
3. [Kategoria II: Ewolucja wymiaru w czasie $d(\tau)$ dla $K \in [3, 50]$ (`d_vs_tau_k_dependency/`)](#3-kategoria-ii-ewolucja-wymiaru-w-czasie-dtau-dla-k-in-3-50)
   - [Wariant wieloliniowy: Z pasmem $\pm 1\sigma$ vs Czyste linie (Lines Only)](#a-wariant-wieloliniowy-kolor-linii-koduje-k)
   - [Wariant uśredniony: Z pasmem niepewności $\sigma_K$ vs Pojedyncza linia średniej (Line Only)](#b-wariant-uśredniony-po-k-in-3-50)
4. [Kategoria III: Zależność wymiaru od wielkości otoczenia $d(K)$ dla przekrojów czasowych (`k_sweep_curves/`)](#4-kategoria-iii-zależność-wymiaru-od-wielkości-otoczenia-dk)
5. [Kategoria IV: Badanie czułości skali $k$-NN w parach podwajania (`k_dependency/`)](#5-kategoria-iv-badanie-czułości-skali-k-nn-w-parach-podwajania)
6. [Ściąga z Makr i Szerokości w LaTeX](#6-ściąga-z-makr-i-szerokości-w-latex)

---

## 1. Wprowadzenie i Standardy Publikacyjne

Wszystkie wykresy w niniejszym katalogu zostały wygenerowane z zachowaniem rygorystycznych standardów czasopism fizycznych (np. *Physical Review C/D*, *JHEP*, *EPJC*):
1. **Brak wewnętrznych tytułów w plikach PDF:**  
   Wykresy nie posiadają napisów nagłówkowych u góry ramki. Cała identyfikacja rysunku, parametry numeryczne i interpretacja fizyczna są realizowane przez środowisko `\caption{...}` w LaTeX-u.
2. **Ścisłe rozdzielenie miar `dims()` i `pr()`:**  
   Dyskretny wymiar spektralny $\mathrm{dims}() \in \{1, 2, 3\}$ (odcięcie progiem $\text{tol} = 0.01$) oraz ciągły wskaźnik Participation Ratio $\mathrm{PR}() \in [1, 3]$ są zapisane w **dwóch całkowicie niezależnych plikach PDF**. Pozwala to na zestawianie ich obok siebie w podwójnych panelach (`(a)` dyskretny, `(b)` ciągły).
3. **Ucięcie osi czasu:**  
   Wszystkie krzywe ewolucji czasowej są ucięte przy $\tau \le 6.0\,\mathrm{fm}/c$.
4. **Dwustopniowe próbkowanie punktów ($N_{\text{calc}} = 10\,000$, $N_{\text{plot}} = 5\,000$):**  
   W celu zachowania wysokiej dokładności statystycznej estymacji wymiaru ($k$-NN, lokalna macierz kowariancji, wskaźnik Participation Ratio), procedury LPCA i PR wykonują obliczenia na gęstym zbiorze $N_{\text{calc}} = 10\,000$ punktów na dany przekrój czasowy $\tau$. Do samej prezentacji graficznej (wykresu punktowego) losowane jest bez zwracania dokładnie $N_{\text{plot}} = 5\,000$ punktów. Rozwiązanie to zapobiega efektowi zatarcia geometrii punktów (overplotting / okluzja sfer w 3D) i ogranicza wagę wektorowych plików PDF o ponad 60\% przy zachowaniu pełnej precyzji lokalnych sąsiedztw.

---

## 2. Kategoria I: Ewolucja przestrzeni fazowej 2D i 3D

Wszystkie wykresy przestrzeni fazowej zostały posegregowane w dedykowanych katalogach:
- [`publication_lpca/plots/phase_space/mis/`](file:///home/kitajusus/github/Atractors-in-QGP/publication_lpca/plots/phase_space/mis/) – model Conformal MIS (2D)
- [`publication_lpca/plots/phase_space/hjsw/`](file:///home/kitajusus/github/Atractors-in-QGP/publication_lpca/plots/phase_space/hjsw/) – model HJSW (3D)

W każdym z tych katalogów znajdują się:
1. **Pojedyncze przekroje czasowe dla każdej z 9 chwil:** $\tau \in \{0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80, 1.00, 2.50\}\,\mathrm{fm}/c$.  
   Pojedyncze pliki mają powiększony nagłówek $\tau = \dots\,\mathrm{fm}/c$ (rozmiar fontu 24) i pozwalają na swobodne układanie własnych paneli i siatek w LaTeX-u.  
   - Format: `{model}_{standaryzacja}_slice_tau_{tau}_{miara}.pdf` (np. `mis_zscore_slice_tau_0.25_dims.pdf`)
2. **Zbiorcze siatki $3 \times 3$:** z powiększonymi tytułami paneli i poszerzonymi odstępami między wykresami (55 px).  
   - Format: `grid_{model}_{standaryzacja}_{miara}.pdf` (np. `grid_mis_zscore_dims.pdf`)
3. **Pliki instrukcji i szablonów LaTeX:** `README.md` oraz `FIGURES_INFO.tex` z gotowymi kodami `subfigure` do wklejenia do pracy.

### A. Model Conformal MIS (2D)

| Siatka 3x3 `dims()` | Siatka 3x3 `pr()` | Pojedyncze przekroje [PDF] | Metoda standaryzacji |
| :--- | :--- | :--- | :--- |
| `grid_mis_zscore_dims.pdf` | `grid_mis_zscore_pr.pdf` | `mis_zscore_slice_tau_{tau}_...` | Z-Score (rekomendowana) |
| `grid_mis_dimensionless_dims.pdf` | `grid_mis_dimensionless_pr.pdf` | `mis_dimensionless_slice_tau_{tau}_...` | Bezwymiarowa ($w = \tau T$) |
| `grid_mis_minmax_dims.pdf` | `grid_mis_minmax_pr.pdf` | `mis_minmax_slice_tau_{tau}_...` | Min-Max ($[0, 1]^2$) |
| `grid_mis_max_dims.pdf` | `grid_mis_max_pr.pdf` | `mis_max_slice_tau_{tau}_...` | Abs-Max ($[-1, 1]^2$) |
| `grid_mis_physical_dims.pdf` | `grid_mis_physical_pr.pdf` | `mis_physical_slice_tau_{tau}_...` | Fizyczna (surowa) |

#### Kod LaTeX do zestawienia 4 metod standaryzacji (MIS 2D):
```latex
\begin{figure}[htbp]
    \centering
    % Wiersz 1: Dimensionless
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{best_practices_lpca/grid_mis_dimensionless_dims.pdf}
        \caption{Dimensionless (\emph{dims})}
        \label{fig:mis_dim_dims}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{best_practices_lpca/grid_mis_dimensionless_pr.pdf}
        \caption{Dimensionless (\emph{pr})}
        \label{fig:mis_dim_pr}
    \end{subfigure}

    \vspace{0.8em}

    % Wiersz 2: Max
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{best_practices_lpca/grid_mis_max_dims.pdf}
        \caption{Abs-Max (\emph{dims})}
        \label{fig:mis_max_dims}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{best_practices_lpca/grid_mis_max_pr.pdf}
        \caption{Abs-Max (\emph{pr})}
        \label{fig:mis_max_pr}
    \end{subfigure}

    \caption{Ewolucja przestrzeni fazowej w modelu Conformal MIS (cz. 1): lewa kolumna przedstawia dyskretny wymiar lokalny \emph{dims} ($\text{tol}=0.01$), prawa wskaźnik ciągły Participation Ratio \emph{pr}. Pierwszy panel odpowiada stanowi początkowemu $\tau_0 = 0.20\,\mathrm{fm}/c$.}
    \label{fig:mis_phase_space_part1}
\end{figure}

\begin{figure}[htbp]\ContinuedFloat
    \centering
    % Wiersz 3: Min-Max
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{best_practices_lpca/grid_mis_minmax_dims.pdf}
        \caption{Min-Max (\emph{dims})}
        \label{fig:mis_minmax_dims}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{best_practices_lpca/grid_mis_minmax_pr.pdf}
        \caption{Min-Max (\emph{pr})}
        \label{fig:mis_minmax_pr}
    \end{subfigure}

    \vspace{0.8em}

    % Wiersz 4: Z-Score
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{best_practices_lpca/grid_mis_zscore_dims.pdf}
        \caption{Z-Score (\emph{dims})}
        \label{fig:mis_zscore_dims}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{best_practices_lpca/grid_mis_zscore_pr.pdf}
        \caption{Z-Score (\emph{pr})}
        \label{fig:mis_zscore_pr}
    \end{subfigure}

    \caption{Ewolucja przestrzeni fazowej w modelu Conformal MIS (cz. 2, dokończenie): warianty Min-Max oraz Z-Score.}
    \label{fig:mis_phase_space_part2}
\end{figure}
```

---

### B. Model HJSW (3D)

W modelu HJSW punkty fazowe leżą w 3D: $(T, \mathcal{A}, \mathcal{B})$ lub $(w, \mathcal{A}, \mathcal{B})$. Wykresy 3D zoptymalizowano kątowo (oś $\mathcal{B}$ po lewej, temperatura rosnąca od lewej do prawej ku przodowi).

| Siatka 3x3 `dims()` | Siatka 3x3 `pr()` | Pojedyncze przekroje 3D [PDF] | Metoda standaryzacji |
| :--- | :--- | :--- | :--- |
| `grid_hjsw_zscore_dims.pdf` | `grid_hjsw_zscore_pr.pdf` | `hjsw_zscore_slice_tau_{tau}_...` | Z-Score (rekomendowana) |
| `grid_hjsw_dimensionless_dims.pdf` | `grid_hjsw_dimensionless_pr.pdf` | `hjsw_dimensionless_slice_tau_{tau}_...` | Bezwymiarowa ($w = \tau T$) |
| `grid_hjsw_minmax_dims.pdf` | `grid_hjsw_minmax_pr.pdf` | `hjsw_minmax_slice_tau_{tau}_...` | Min-Max ($[0, 1]^3$) |
| `grid_hjsw_max_dims.pdf` | `grid_hjsw_max_pr.pdf` | `hjsw_max_slice_tau_{tau}_...` | Abs-Max ($[-1, 1]$) |
| `grid_hjsw_physical_dims.pdf` | `grid_hjsw_physical_pr.pdf` | `hjsw_physical_slice_tau_{tau}_...` | Fizyczna (surowa) |

#### Kod LaTeX do zestawienia HJSW 3D:
```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{phase_space/hjsw/grid_hjsw_zscore_dims.pdf}
        \caption{HJSW Z-Score (\emph{dims})}
        \label{fig:hjsw_zscore_dims}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{phase_space/hjsw/grid_hjsw_zscore_pr.pdf}
        \caption{HJSW Z-Score (\emph{pr})}
        \label{fig:hjsw_zscore_pr}
    \end{subfigure}

    \caption{Hierarchiczny kolaps wymiarowości $3 \to 2 \to 1$ w modelu HJSW w standaryzacji Z-Score. W chwili $\tau_0 = 0.20\,\mathrm{fm}/c$ układ wypełnia pełną przestrzeń 3D. W $\tau = 0.25\,\mathrm{fm}/c$ szybki mod niehydrodynamiczny ulega kompresji 58-krotnej, wprowadzając układ w 2-wymiarową rozmaitość pośrednią ($d \approx 2.0$), która następnie wolno kolapsuje hydrodynamicznie ku 1D atraktorowi.}
    \label{fig:hjsw_phase_space_zscore}
\end{figure}
```

---

### C. Wielometodowe nałożenia ewolucji średniego wymiaru $\langle d \rangle(\tau)$

Lokalizacja: [`publication_lpca/plots/best_practices_lpca/`](file:///home/kitajusus/github/Atractors-in-QGP/publication_lpca/plots/best_practices_lpca/)  
Na jednym wykresie zestawiono trajektorie dla 4 metod normalizacji (`none`, `max`, `minmax`, `zscore`):
- `mean_dimension_vs_tau_mis_dims.pdf` & `mean_dimension_vs_tau_mis_pr.pdf`
- `mean_dimension_vs_tau_hjsw_dims.pdf` & `mean_dimension_vs_tau_hjsw_pr.pdf`

#### Kod LaTeX:
```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{best_practices_lpca/mean_dimension_vs_tau_mis_dims.pdf}
        \caption{MIS: \emph{dims}}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{best_practices_lpca/mean_dimension_vs_tau_mis_pr.pdf}
        \caption{MIS: \emph{pr}}
    \end{subfigure}

    \vspace{0.8em}

    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{best_practices_lpca/mean_dimension_vs_tau_hjsw_dims.pdf}
        \caption{HJSW: \emph{dims}}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{best_practices_lpca/mean_dimension_vs_tau_hjsw_pr.pdf}
        \caption{HJSW: \emph{pr}}
    \end{subfigure}

    \caption{Porównanie wrażliwości metod normalizacji przestrzeni fazowej w czasie $\tau \in [0.2, 6.0]\,\mathrm{fm}/c$. Brak standaryzacji (Raw Units) powoduje fałszywy pozorny kolaps wymiaru w chwili początkowej z powodu dominacji skali temperatury nad anizotropią.}
    \label{fig:mean_dimension_all_methods}
\end{figure}
```

---

## 3. Kategoria II: Ewolucja wymiaru w czasie $d(\tau)$ dla $K \in [3, 50]$

Lokalizacja: [`publication_lpca/plots/d_vs_tau_k_dependency/`](file:///home/kitajusus/github/Atractors-in-QGP/publication_lpca/plots/d_vs_tau_k_dependency/)  
Wykresy te pokazują zależność wyznaczanego wymiaru od wielkości otoczenia $K \in [3, 50]$ wzdłuż osi czasu własnego $\tau \in [0.20, 6.0]\,\mathrm{fm}/c$.

Dostępne są **dwa warianty wizualne**:
1. **Z pasmem błędu $\pm 1\sigma$:** Pokazuje rozrzut statystyczny w zespole symulacji.
2. **Czyste linie (Lines Only / Line Only):** Maksymalna przejrzystość, bez cieniowania – idealne do dokładnego śledzenia przebiegu każdej krzywej $K$.

---

### A. Wariant wieloliniowy (Kolor linii koduje $K$)
Krzywe dla $K \in \{3, 5, 8, 12, 16, 24, 35, 50\}$.

| Z pasmem $\pm 1\sigma$ [PDF] | Czyste linie (Lines Only) [PDF] | Model | Reprezentacja / Standaryzacja | Miara |
| :--- | :--- | :--- | :--- | :--- |
| `d_vs_tau_mis_zscore_multi_k_dims.pdf` | `d_vs_tau_mis_zscore_multi_k_lines_dims.pdf` | MIS | Z-Score (`zscore`) | `dims()` |
| `d_vs_tau_mis_zscore_multi_k_pr.pdf` | `d_vs_tau_mis_zscore_multi_k_lines_pr.pdf` | MIS | Z-Score (`zscore`) | `pr()` |
| `d_vs_tau_mis_minmax_multi_k_dims.pdf` | `d_vs_tau_mis_minmax_multi_k_lines_dims.pdf` | MIS | Min-Max (`minmax`) | `dims()` |
| `d_vs_tau_mis_minmax_multi_k_pr.pdf` | `d_vs_tau_mis_minmax_multi_k_lines_pr.pdf` | MIS | Min-Max (`minmax`) | `pr()` |
| `d_vs_tau_mis_max_multi_k_dims.pdf` | `d_vs_tau_mis_max_multi_k_lines_dims.pdf` | MIS | Abs-Max (`max`) | `dims()` |
| `d_vs_tau_mis_max_multi_k_pr.pdf` | `d_vs_tau_mis_max_multi_k_lines_pr.pdf` | MIS | Abs-Max (`max`) | `pr()` |
| `d_vs_tau_mis_physical_multi_k_dims.pdf` | `d_vs_tau_mis_physical_multi_k_lines_dims.pdf` | MIS | Physical (`physical`) | `dims()` |
| `d_vs_tau_mis_physical_multi_k_pr.pdf` | `d_vs_tau_mis_physical_multi_k_lines_pr.pdf` | MIS | Physical (`physical`) | `pr()` |
| `d_vs_tau_mis_dimensionless_multi_k_dims.pdf` | `d_vs_tau_mis_dimensionless_multi_k_lines_dims.pdf` | MIS | Dimensionless (`dimensionless`) | `dims()` |
| `d_vs_tau_mis_dimensionless_multi_k_pr.pdf` | `d_vs_tau_mis_dimensionless_multi_k_lines_pr.pdf` | MIS | Dimensionless (`dimensionless`) | `pr()` |
| `d_vs_tau_hjsw_zscore_multi_k_dims.pdf` | `d_vs_tau_hjsw_zscore_multi_k_lines_dims.pdf` | HJSW | Z-Score (`zscore`) | `dims()` |
| `d_vs_tau_hjsw_zscore_multi_k_pr.pdf` | `d_vs_tau_hjsw_zscore_multi_k_lines_pr.pdf` | HJSW | Z-Score (`zscore`) | `pr()` |
| `d_vs_tau_hjsw_minmax_multi_k_dims.pdf` | `d_vs_tau_hjsw_minmax_multi_k_lines_dims.pdf` | HJSW | Min-Max (`minmax`) | `dims()` |
| `d_vs_tau_hjsw_minmax_multi_k_pr.pdf` | `d_vs_tau_hjsw_minmax_multi_k_lines_pr.pdf` | HJSW | Min-Max (`minmax`) | `pr()` |
| `d_vs_tau_hjsw_max_multi_k_dims.pdf` | `d_vs_tau_hjsw_max_multi_k_lines_dims.pdf` | HJSW | Abs-Max (`max`) | `dims()` |
| `d_vs_tau_hjsw_max_multi_k_pr.pdf` | `d_vs_tau_hjsw_max_multi_k_lines_pr.pdf` | HJSW | Abs-Max (`max`) | `pr()` |
| `d_vs_tau_hjsw_physical_multi_k_dims.pdf` | `d_vs_tau_hjsw_physical_multi_k_lines_dims.pdf` | HJSW | Physical (`physical`) | `dims()` |
| `d_vs_tau_hjsw_physical_multi_k_pr.pdf` | `d_vs_tau_hjsw_physical_multi_k_lines_pr.pdf` | HJSW | Physical (`physical`) | `pr()` |
| `d_vs_tau_hjsw_dimensionless_multi_k_dims.pdf` | `d_vs_tau_hjsw_dimensionless_multi_k_lines_dims.pdf` | HJSW | Dimensionless (`dimensionless`) | `dims()` |
| `d_vs_tau_hjsw_dimensionless_multi_k_pr.pdf` | `d_vs_tau_hjsw_dimensionless_multi_k_lines_pr.pdf` | HJSW | Dimensionless (`dimensionless`) | `pr()` |

*(Uwaga: dla zachowania pełnej kompatybilności wstecznej dla wariantu Z-Score pliki są zapisywane także pod nazwami bez członu `_zscore_`, np. `d_vs_tau_mis_multi_k_dims.pdf`).*

#### Kod LaTeX dla czystych linii (Lines Only):
```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{d_vs_tau_k_dependency/d_vs_tau_mis_multi_k_lines_dims.pdf}
        \caption{MIS: \emph{dims} ($K \in [3, 50]$)}
        \label{fig:d_tau_mis_lines_dims}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{d_vs_tau_k_dependency/d_vs_tau_mis_multi_k_lines_pr.pdf}
        \caption{MIS: \emph{pr} ($K \in [3, 50]$)}
        \label{fig:d_tau_mis_lines_pr}
    \end{subfigure}

    \vspace{0.8em}

    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{d_vs_tau_k_dependency/d_vs_tau_hjsw_multi_k_lines_dims.pdf}
        \caption{HJSW: \emph{dims} ($K \in [3, 50]$)}
        \label{fig:d_tau_hjsw_lines_dims}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{d_vs_tau_k_dependency/d_vs_tau_hjsw_multi_k_lines_pr.pdf}
        \caption{HJSW: \emph{pr} ($K \in [3, 50]$)}
        \label{fig:d_tau_hjsw_lines_pr}
    \end{subfigure}

    \caption{Trajektorie ewolucji wymiaru $d(\tau)$ dla wybranych wartości $K \in \{3, 5, 8, 12, 16, 24, 35, 50\}$ (standaryzacja Z-Score). Wariant \emph{Lines Only} bez pasm błędu ukazuje, że dla dowolnego $K \ge 12$ krzywe leżą niemal dokładnie na sobie, co dowodzi niezmienniczości wyznaczanego wymiaru względem skali $k$-NN.}
    \label{fig:d_vs_tau_multi_k_lines}
\end{figure}
```

---

### B. Wariant uśredniony po $K \in [3, 50]$
Pojedyncza krzywa reprezentująca średnią $\langle d \rangle_K(\tau)$ uśrednioną po całym spektrum 48 wartości $K \in [3, 50]$.

| Z pasmem $\pm 1\sigma_K$ i obwiednią [PDF] | Czysta pojedyncza linia (Line Only) [PDF] | Model | Reprezentacja / Standaryzacja | Miara |
| :--- | :--- | :--- | :--- | :--- |
| `d_vs_tau_mis_zscore_k_averaged_dims.pdf` | `d_vs_tau_mis_zscore_k_averaged_line_only_dims.pdf` | MIS | Z-Score (`zscore`) | `dims()` |
| `d_vs_tau_mis_zscore_k_averaged_pr.pdf` | `d_vs_tau_mis_zscore_k_averaged_line_only_pr.pdf` | MIS | Z-Score (`zscore`) | `pr()` |
| `d_vs_tau_mis_minmax_k_averaged_dims.pdf` | `d_vs_tau_mis_minmax_k_averaged_line_only_dims.pdf` | MIS | Min-Max (`minmax`) | `dims()` |
| `d_vs_tau_mis_minmax_k_averaged_pr.pdf` | `d_vs_tau_mis_minmax_k_averaged_line_only_pr.pdf` | MIS | Min-Max (`minmax`) | `pr()` |
| `d_vs_tau_mis_max_k_averaged_dims.pdf` | `d_vs_tau_mis_max_k_averaged_line_only_dims.pdf` | MIS | Abs-Max (`max`) | `dims()` |
| `d_vs_tau_mis_max_k_averaged_pr.pdf` | `d_vs_tau_mis_max_k_averaged_line_only_pr.pdf` | MIS | Abs-Max (`max`) | `pr()` |
| `d_vs_tau_mis_physical_k_averaged_dims.pdf` | `d_vs_tau_mis_physical_k_averaged_line_only_dims.pdf` | MIS | Physical (`physical`) | `dims()` |
| `d_vs_tau_mis_physical_k_averaged_pr.pdf` | `d_vs_tau_mis_physical_k_averaged_line_only_pr.pdf` | MIS | Physical (`physical`) | `pr()` |
| `d_vs_tau_mis_dimensionless_k_averaged_dims.pdf` | `d_vs_tau_mis_dimensionless_k_averaged_line_only_dims.pdf` | MIS | Dimensionless (`dimensionless`) | `dims()` |
| `d_vs_tau_mis_dimensionless_k_averaged_pr.pdf` | `d_vs_tau_mis_dimensionless_k_averaged_line_only_pr.pdf` | MIS | Dimensionless (`dimensionless`) | `pr()` |
| `d_vs_tau_hjsw_zscore_k_averaged_dims.pdf` | `d_vs_tau_hjsw_zscore_k_averaged_line_only_dims.pdf` | HJSW | Z-Score (`zscore`) | `dims()` |
| `d_vs_tau_hjsw_zscore_k_averaged_pr.pdf` | `d_vs_tau_hjsw_zscore_k_averaged_line_only_pr.pdf` | HJSW | Z-Score (`zscore`) | `pr()` |
| `d_vs_tau_hjsw_minmax_k_averaged_dims.pdf` | `d_vs_tau_hjsw_minmax_k_averaged_line_only_dims.pdf` | HJSW | Min-Max (`minmax`) | `dims()` |
| `d_vs_tau_hjsw_minmax_k_averaged_pr.pdf` | `d_vs_tau_hjsw_minmax_k_averaged_line_only_pr.pdf` | HJSW | Min-Max (`minmax`) | `pr()` |
| `d_vs_tau_hjsw_max_k_averaged_dims.pdf` | `d_vs_tau_hjsw_max_k_averaged_line_only_dims.pdf` | HJSW | Abs-Max (`max`) | `dims()` |
| `d_vs_tau_hjsw_max_k_averaged_pr.pdf` | `d_vs_tau_hjsw_max_k_averaged_line_only_pr.pdf` | HJSW | Abs-Max (`max`) | `pr()` |
| `d_vs_tau_hjsw_physical_k_averaged_dims.pdf` | `d_vs_tau_hjsw_physical_k_averaged_line_only_dims.pdf` | HJSW | Physical (`physical`) | `dims()` |
| `d_vs_tau_hjsw_physical_k_averaged_pr.pdf` | `d_vs_tau_hjsw_physical_k_averaged_line_only_pr.pdf` | HJSW | Physical (`physical`) | `pr()` |
| `d_vs_tau_hjsw_dimensionless_k_averaged_dims.pdf` | `d_vs_tau_hjsw_dimensionless_k_averaged_line_only_dims.pdf` | HJSW | Dimensionless (`dimensionless`) | `dims()` |
| `d_vs_tau_hjsw_dimensionless_k_averaged_pr.pdf` | `d_vs_tau_hjsw_dimensionless_k_averaged_line_only_pr.pdf` | HJSW | Dimensionless (`dimensionless`) | `pr()` |

*(Dla wariantu Z-Score pliki są zapisywane także pod nazwami bez członu `_zscore_`, np. `d_vs_tau_mis_k_averaged_dims.pdf`).*

#### Kod LaTeX dla wariantu z pasmem niepewności wyboru $K$ ($\pm 1\sigma_K$):
```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{d_vs_tau_k_dependency/d_vs_tau_mis_k_averaged_dims.pdf}
        \caption{MIS: Średnia $\langle d \rangle_K$ (\emph{dims})}
        \label{fig:d_tau_mis_avg_dims}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{d_vs_tau_k_dependency/d_vs_tau_mis_k_averaged_pr.pdf}
        \caption{MIS: Średnia $\langle d_{\mathrm{PR}} \rangle_K$ (\emph{pr})}
        \label{fig:d_tau_mis_avg_pr}
    \end{subfigure}

    \vspace{0.8em}

    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{d_vs_tau_k_dependency/d_vs_tau_hjsw_k_averaged_dims.pdf}
        \caption{HJSW: Średnia $\langle d \rangle_K$ (\emph{dims})}
        \label{fig:d_tau_hjsw_avg_dims}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{d_vs_tau_k_dependency/d_vs_tau_hjsw_k_averaged_pr.pdf}
        \caption{HJSW: Średnia $\langle d_{\mathrm{PR}} \rangle_K$ (\emph{pr})}
        \label{fig:d_tau_hjsw_avg_pr}
    \end{subfigure}

    \caption{Trajektoria ewolucji wymiaru uśredniona po pełnym zakresie $K \in [3, 50]$. Ciemniejsze pasmo reprezentuje niepewność arbitralnego doboru sąsiadów $\pm 1\sigma_K$, a jaśniejsze obwiednię $[\min_K, \max_K]$. Wąskość pasma potwierdza minimalny wpływ wyboru parametru $K$ na ostateczne wnioski fizyczne.}
    \label{fig:d_vs_tau_k_averaged}
\end{figure}
```

---

## 4. Kategoria III: Zależność wymiaru od wielkości otoczenia $d(K)$

Lokalizacja: [`publication_lpca/plots/k_sweep_curves/`](file:///home/kitajusus/github/Atractors-in-QGP/publication_lpca/plots/k_sweep_curves/)  
Przedstawia wymiar w funkcji $K \in [3, 50]$ dla 8 przekrojów czasowych $\tau \in [0.20, 2.50]\,\mathrm{fm}/c$.

| Model | Standaryzacja Z-Score [PDF] | Min-Max [PDF] | Abs-Max [PDF] | Współrzędne Fizyczne [PDF] | Bezwymiarowe [PDF] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HJSW `dims()`** | `k_sweep_hjsw_zscore_dims.pdf` | `k_sweep_hjsw_minmax_dims.pdf` | `k_sweep_hjsw_max_dims.pdf` | `k_sweep_hjsw_physical_dims.pdf` | `k_sweep_hjsw_dimensionless_dims.pdf` |
| **HJSW `pr()`** | `k_sweep_hjsw_zscore_pr.pdf` | `k_sweep_hjsw_minmax_pr.pdf` | `k_sweep_hjsw_max_pr.pdf` | `k_sweep_hjsw_physical_pr.pdf` | `k_sweep_hjsw_dimensionless_pr.pdf` |
| **MIS `dims()`** | `k_sweep_mis_zscore_dims.pdf` | `k_sweep_mis_minmax_dims.pdf` | `k_sweep_mis_max_dims.pdf` | `k_sweep_mis_physical_dims.pdf` | `k_sweep_mis_dimensionless_dims.pdf` |
| **MIS `pr()`** | `k_sweep_mis_zscore_pr.pdf` | `k_sweep_mis_minmax_pr.pdf` | `k_sweep_mis_max_pr.pdf` | `k_sweep_mis_physical_pr.pdf` | `k_sweep_mis_dimensionless_pr.pdf` |

#### Kod LaTeX:
```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{k_sweep_curves/k_sweep_hjsw_zscore_dims.pdf}
        \caption{HJSW: $d(K)$ dla \emph{dims}}
        \label{fig:k_sweep_hjsw_dims}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{k_sweep_curves/k_sweep_hjsw_zscore_pr.pdf}
        \caption{HJSW: $d_{\mathrm{PR}}(K)$ dla \emph{pr}}
        \label{fig:k_sweep_hjsw_pr}
    \end{subfigure}

    \caption{Zależność estymowanego wymiaru od wielkości otoczenia $K \in [3, 50]$ dla 8 przekrojów czasowych $\tau \in [0.20, 2.50]\,\mathrm{fm}/c$ w modelu HJSW (Z-Score). Plateau w przedziale $K \in [10, 40]$ wyznacza optymalne okno pomiarowe.}
    \label{fig:k_sweep_curves_hjsw}
\end{figure}
```

---

## 5. Kategoria IV: Badanie czułości skali $k$-NN w parach podwajania

Lokalizacja: [`publication_lpca/plots/k_dependency/`](file:///home/kitajusus/github/Atractors-in-QGP/publication_lpca/plots/k_dependency/)  
Tradycyjna metoda testowania stabilności LPCA: cieniowane pasmo pomiędzy $K_{\mathrm{base}}$ a $2 K_{\mathrm{base}}$ dla par $\{(3,6), (6,12), (12,24), (24,48), (100,200)\}$.

| Model | Plik `dims()` [PDF] | Plik `pr()` [PDF] | Układ współrzędnych |
| :--- | :--- | :--- | :--- |
| **HJSW** | `k_dependency_hjsw_physical_coordinates_TAB_dims.pdf` | `k_dependency_hjsw_physical_coordinates_TAB_pr.pdf` | Fizyczne $(T, \mathcal{A}, \mathcal{B})$ |
| **HJSW** | `k_dependency_hjsw_dimensionless_coordinates_wAB_dims.pdf` | `k_dependency_hjsw_dimensionless_coordinates_wAB_pr.pdf` | Bezwymiarowe $(w, \mathcal{A}, \mathcal{B})$ |
| **MIS** | `k_dependency_mis_physical_coordinates_TA_dims.pdf` | `k_dependency_mis_physical_coordinates_TA_pr.pdf` | Fizyczne $(T, \mathcal{A})$ |
| **MIS** | `k_dependency_mis_dimensionless_coordinates_wA_dims.pdf` | `k_dependency_mis_dimensionless_coordinates_wA_pr.pdf` | Bezwymiarowe $(w, \mathcal{A})$ |

---

## 6. Ściąga z Makr i Szerokości w LaTeX

Aby układ tabel i podrysunków był idealny i nie wychodził poza margines:

| Układ paneli | Rekomendowana szerokość | Odstęp | Komentarz |
| :--- | :--- | :--- | :--- |
| **2 panele obok siebie** | `width=0.48\textwidth` | `\hfill` | Idealne dla pary `dims` i `pr` na stronie pionowej A4. |
| **4 panele w poziomie (Landscape)** | `width=0.23\textheight` | `\setlength{\tabcolsep}{3pt}` | Dla pakietu `rotating` (`sidewaysfigure`). |
| **Pojedyncza szeroka figura** | `width=0.85\textwidth` | `\centering` | Gdy wykres ma stać samodzielnie na środku kolumny. |
| **Dwułamowy artykuł (RevTeX / IEEE)** | `width=\linewidth` | w `figure` (1 kolumna) lub `figure*` (obie kolumny) | Zastąp `0.48\textwidth` przez `\linewidth`. |

### Wymagane pakiety w preambule LaTeX:
```latex
\usepackage{graphicx}
\usepackage{subcaption} % dla \begin{subfigure} i \ContinuedFloat
\usepackage{booktabs}   % dla estetycznych tabel
\usepackage{rotating}   % opcjonalnie, dla tabel/figur obróconych o 90 stopni (sidewaysfigure)
```
