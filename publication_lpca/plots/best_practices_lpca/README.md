# Dobre praktyki i zestawienia porównawcze (best_practices_lpca)

Katalog zawiera syntetyczne zestawienia porównawcze metod standaryzacji przestrzeni fazowej oraz ewolucji wymiaru lokalnego.

## 1. Wykresy bezpośredniego nałożenia ewolucji wymiaru (Direct Overlay)
Przedstawiają trajektorie $\langle d \rangle(\tau)$ oraz $\langle d_{\mathrm{PR}} \rangle(\tau)$ dla 4 metod standaryzacji na jednym panelu ($k=24$):
- Czerwony: Surowe jednostki (Raw Units, brak standaryzacji)
- Niebieski: Abs-Max Scaling ($[-1, 1]$)
- Zielony: Min-Max Scaling ($[0, 1]$)
- Pomarańczowy: Z-Score Standardization

### Pliki:
- `mean_dimension_vs_tau_mis_dims.pdf` – MIS dla `dims()`
- `mean_dimension_vs_tau_mis_pr.pdf` – MIS dla `pr()`
- `mean_dimension_vs_tau_hjsw_dims.pdf` – HJSW dla `dims()`
- `mean_dimension_vs_tau_hjsw_pr.pdf` – HJSW dla `pr()`

## 2. Zbiorcze siatki przestrzeni fazowej 3x3
- `grid_mis_{metoda}_{miara}.pdf`
- `grid_hjsw_{metoda}_{miara}.pdf`
*(Uwaga: wyodrębnione pojedyncze przekroje czasowe dla każdego tau znajdują się w dedykowanych katalogach `publication_lpca/plots/phase_space/mis/` oraz `publication_lpca/plots/phase_space/hjsw/`).*
