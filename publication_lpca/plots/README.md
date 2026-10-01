# Katalog Wykresów Publikacyjnych LPCA (Local PCA)

Ten katalog zawiera wszystkie wykresy i siatki wektorowe w formacie PDF przygotowane do publikacji naukowej:
**"Identifying attractors with local PCA"** (K. Bezubik, M. Spaliński, M. Waśko).

---

## 1. Zorganizowana Struktura Katalogów

Wszystkie rysunki zostały posegregowane w logicznej hierarchii:
- **Według modelu fizycznego:** `mis/` (Conformal MIS, 2D) oraz `hjsw/` (Heller-Janik-Spaliński-Witaszczyk, 3D).
- **Według metody standaryzacji / układu współrzędnych:**
  - `dimensionless/` – bezwymiarowe współrzędne atraktora ($w = \tau T$)
  - `physical/` – surowe jednostki fizyczne ($T$, $\mathcal{A}$ lub $T$, $\mathcal{A}$, $\mathcal{B}$)
  - `max/` – skalowanie Abs-Max do przedziału $[-1, 1]$
  - `minmax/` – normalizacja Min-Max do przedziału $[0, 1]$
  - `zscore/` – standaryzacja Z-Score (jednostki wariancji)
- **Według miary wymiaru lokalnego:**
  - `dims/` – dyskretny wymiar lokalny (odcięcie progiem $\mathrm{tol} = 0.01$)
  - `pr/` – ciągły wskaźnik Participation Ratio $d_{\mathrm{PR}} = (\mathrm{Tr}\,C)^2 / \mathrm{Tr}(C^2)$
- **Wykresy przekrojowe i porównawcze:** `comparisons/` oraz `comparisons_mis_vs_hjsw/`.

Dla wygody na poziomie głównym utworzono również dowiązania symboliczne umożliwiające bezpośrednie przeglądanie według normalizacji:
`dimensionless/`, `physical/`, `max/`, `minmax/`, `zscore/`.

```text
publication_lpca/plots/
├── mis/                                    # Model Conformal MIS (2D: T, A)
│   ├── dimensionless/                      # Współrzędne w = tau * T
│   │   ├── dims/                           # Dyskretny wymiar dims()
│   │   └── pr/                             # Ciągły Participation Ratio pr()
│   ├── physical/                           # Surowe jednostki fizyczne (T w fm^-1, A)
│   │   ├── dims/
│   │   └── pr/
│   ├── max/                                # Skalowanie Abs-Max [-1, 1]
│   │   ├── dims/
│   │   └── pr/
│   ├── minmax/                             # Skalowanie Min-Max [0, 1]
│   │   ├── dims/
│   │   └── pr/
│   ├── zscore/                             # Standaryzacja Z-score
│   │   ├── dims/
│   │   └── pr/
│   └── comparisons/                        # Porównania normalizacji, niezmienniczość, rozkłady
│
├── hjsw/                                   # Model HJSW (3D: T, A, B)
│   ├── dimensionless/                      # Współrzędne w = tau * T
│   │   ├── dims/
│   │   └── pr/
│   ├── physical/                           # Surowe jednostki fizyczne (T w MeV, A, B)
│   │   ├── dims/
│   │   └── pr/
│   ├── max/                                # Skalowanie Abs-Max [-1, 1]
│   │   ├── dims/
│   │   └── pr/
│   ├── minmax/                             # Skalowanie Min-Max [0, 1]
│   │   ├── dims/
│   │   └── pr/
│   ├── zscore/                             # Standaryzacja Z-score
│   │   ├── dims/
│   │   └── pr/
│   └── comparisons/                        # Porównania normalizacji, niezmienniczość, rozkłady
│
├── comparisons_mis_vs_hjsw/                # Bezpośrednie zestawienia MIS vs HJSW
│   ├── eigenvalue_tolerance_cutoff_sensitivity.pdf
│   └── pr_unified_3methods_comparison.pdf
│
└── [dimensionless, physical, max, minmax, zscore]/ # Skróty bezpośredniego dostępu do modeli
```

---

## 2. Zawartość folderów `{model}/{standaryzacja}/{miara}/`

W każdym dedykowanym folderze (np. `mis/dimensionless/dims/` lub `hjsw/zscore/pr/`) znajdują się:
1. **Siatka zbiorcza $3 \times 3$:**  
   `grid_{model}_{standaryzacja}_{miara}.pdf` (9 przekrojów czasowych od $\tau = 0.20$ do $2.50\,\mathrm{fm}/c$)
2. **Pojedyncze przekroje czasowe (Standalone Slices):**  
   `{model}_{standaryzacja}_slice_tau_{tau}_{miara}.pdf` dla $\tau \in \{0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80, 1.00, 2.50\}$
3. **Krzywe ewolucji wymiaru w czasie $d(\tau)$:**  
   - `d_vs_tau_{model}_{standaryzacja}_multi_k_{miara}.pdf` (krzywe dla $K \in \{3, 5, 8, 12, 16, 24, 35, 50\}$ z pasmem $\pm 1\sigma$)
   - `d_vs_tau_{model}_{standaryzacja}_multi_k_lines_{miara}.pdf` (same linie, bez pasm)
   - `d_vs_tau_{model}_{standaryzacja}_k_averaged_{miara}.pdf` (krzywa uśredniona po $K \in [3, 50]$ z obwiednią niepewności $[K_{\min}, K_{\max}]$)
   - `d_vs_tau_{model}_{standaryzacja}_k_averaged_line_only_{miara}.pdf` (sama linia średniej po $K$)
4. **Zależność wymiaru od sąsiedztwa $d(K)$:**  
   `k_sweep_{model}_{standaryzacja}_{miara}.pdf` (zależność wymiaru w funkcji $K \in [3, 50]$ dla różnych $\tau$)
5. **Dwuwymiarowe mapy spektralne:**  
   `map_lpca_{model}_{standaryzacja}.pdf` (mapa ciepła $d(\tau, K)$)

---

## 3. Kompatybilność wsteczna z kodami LaTeX

Stare foldery skryptowe (`best_practices_lpca/`, `d_vs_tau_k_dependency/`, `k_sweep_curves/`, `phase_space/`, `k_dependency/`, `normalization/`, `parameterization_focus/`, `dimension_distribution/`, `coordinate_invariance/`, `participation_ratio/`) zachowano w postaci **relatywnych dowiązań symbolicznych (symlinks)** do plików w nowej strukturze. Żaden istniejący kod LaTeX ani skrypt korzystający ze starych ścieżek nie ulegnie uszkodzeniu.
