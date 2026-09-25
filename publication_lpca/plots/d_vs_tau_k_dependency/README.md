# Ewolucja wymiaru w czasie d(tau) dla K w [3, 50] (d_vs_tau_k_dependency)

Katalog zawiera wektorowe wykresy PDF przedstawiające trajektorie lokalnego wymiaru w funkcji czasu własnego $\tau \in [0.20, 5.00]\,\mathrm{fm}/c$ dla zróżnicowanych wartości $K \in [3, 50]$.

## Reguły nazewnictwa plików
Każdy plik ma schemat: `d_vs_tau_{model}_{standaryzacja}_{wariant}_{miara}.pdf`

- **Modele:** `mis` (Conformal MIS, 2D), `hjsw` (HJSW, 3D)
- **Standaryzacje:**
  1. `zscore` – Z-Score ($\mu=0, \sigma=1$)
  2. `minmax` – Skalowanie Min-Max do $[0, 1]$
  3. `max` – Skalowanie Abs-Max do $[-1, 1]$
  4. `physical` – Surowe jednostki fizyczne (brak standaryzacji)
  5. `dimensionless` – Współrzędne bezwymiarowe ($w = \tau T$)
- **Miary wymiarowości:**
  - `dims` – Dyskretny wymiar lokalny LPCA z progiem $\lambda / \sum \lambda > 0.01$
  - `pr` – Ciągły wymiar Participation Ratio $d_{\mathrm{PR}}$
- **Warianty wykresów:**
  1. `multi_k`: Zbiór krzywych dla $K \in \{3, 5, 8, 12, 16, 24, 35, 50\}$ z pasmem błędu $\pm 1\sigma$
  2. `multi_k_lines`: Czyste linie dla $K \in \{3, 5, 8, 12, 16, 24, 35, 50\}$ (bez pasma błędu)
  3. `k_averaged`: Pojedyncza krzywa $\langle d \rangle_K(\tau)$ uśredniona po $K \in [3, 50]$ z pasmem $\pm 1\sigma_K$ i obwiednią $[\min_K, \max_K]$
  4. `k_averaged_line_only`: Czysta pojedyncza linia $\langle d \rangle_K(\tau)$ bez pasm błędu

*(Uwaga: dla wariantu Z-Score istnieją także aliasy bez członu `_zscore_`, np. `d_vs_tau_mis_multi_k_dims.pdf`).*
