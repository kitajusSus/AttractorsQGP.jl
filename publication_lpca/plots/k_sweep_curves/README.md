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

Każdy wykres zawiera barwne krzywe dla poszczególnych chwil $\tau$ wraz z cieniowanym pasmem $\pm 1\sigma$ dyspersji zespołu.
Płaskowyż widoczny w zakresie $K \in [10, 40]$ dowodzi niezmienniczości wyznaczanego wymiaru i stabilności metody LPCA.
