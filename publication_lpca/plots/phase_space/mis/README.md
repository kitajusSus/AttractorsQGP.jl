# Conformal MIS Phase Space Figures (2D: T, A)

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
