# Krzywe zależności wymiaru od skali sąsiedztwa d(K) dla K w [3, 50] (k_sweep_curves)

Katalog zawiera wykresy wektorowe PDF przedstawiające zależność wymiaru lokalnego od liczby najbliższych sąsiadów $K \in [3, 50]$ dla 8 reprezentatywnych przekrojów czasowych $\tau \in \{0.20, 0.25, 0.35, 0.50, 0.75, 1.00, 1.50, 2.50\}\,\mathrm{fm}/c$.

## Nazewnictwo plików
`k_sweep_{model}_{standaryzacja}_{miara}.pdf`

- **Modele:** `mis`, `hjsw`
- **Standaryzacje:**
  1. `zscore` – Standaryzacja Z-Score (rekomendowana)
  2. `minmax` – Skalowanie do $[0, 1]$
  3. `max` – Skalowanie do $[-1, 1]$
  4. `physical` – Surowe jednostki fizyczne
  5. `dimensionless` – Współrzędne bezwymiarowe ($w = \tau T$)
- **Miary:**
  - `dims` – Dyskretny wymiar lokalny
  - `pr` – Ciągły wymiar Participation Ratio

## Mapy dwuwymiarowe LPCA (Heatmap)
`map_lpca_{model}_{standaryzacja}.pdf` oraz `map_lpca_{model}_{standaryzacja}.png`
- Generowane z użyciem funkcji `plot_map_lpca`
- Przedstawiają mapę wymiaru lokalnego $d(\tau, K)$ na płaszczyźnie czas własny $\tau$ vs liczba sąsiadów $K \in [10, 80]$
- Pokazują jednoczesną stabilność wymiaru względem skali $K$ oraz jego redukcję w czasie (kolaps atraktorowy $2\mathrm{D}\to 1\mathrm{D}$ dla MIS oraz $3\mathrm{D}\to 2\mathrm{D}\to 1\mathrm{D}$ dla HJSW)

Każdy wykres krzywych zawiera barwne krzywe dla poszczególnych chwil $\tau$ wraz z cieniowanym pasmem $\pm 1\sigma$ dyspersji zespołu.
Płaskowyż widoczny w zakresie $K \in [10, 40]$ dowodzi niezmienniczości wyznaczanego wymiaru i stabilności metody LPCA.
