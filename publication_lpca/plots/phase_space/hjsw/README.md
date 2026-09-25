# HJSW Phase Space Figures (3D: T, A, B)

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
