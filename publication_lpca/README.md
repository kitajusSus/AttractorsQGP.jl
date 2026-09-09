# Local PCA (LPCA) Systematic Study for Hydrodynamic Attractor Identification

This directory contains the self-contained publication code, datasets, figures, and numerical results for the manuscript:
**"Identifying attractors with local PCA"** (K. Bezubik, M. Spaliński, M. Waśko).

---

## 1. Directory Structure

```text
publication_lpca/
├── README.md                          # Documentation of findings and figure catalog
├── src/                               # Core analysis library written in idiomatic Julia (Julia Style Guide)
│   ├── PublicationLPCA.jl             # Module interface (CairoMakie vector/raster backend)
│   ├── data_preparation.jl           # Coordinate transformations: (T, A), (w, A), (10T, A)
│   ├── k_dependency_analysis.jl       # Neighbor count K sensitivity & uncertainty bands
│   ├── normalization_analysis.jl      # Normalization comparison: :none, :max, :minmax, :zscore
│   ├── coordinate_invariance.jl       # Scale and chart invariance proofs and evaluations
│   └── dimension_distribution.jl      # Statistical decomposition of dims(): mean, median, P(d=m)
├── scripts/                           # Runnable experiment drivers
│   ├── 01_k_dependency.jl             # K-sweep and uncertainty band evaluation
│   ├── 02_normalization_tests.jl      # Normalization method comparisons (:none, :max, :minmax, :zscore)
│   ├── 03_coordinate_invariance.jl    # Scale (10T) and coordinate chart (w=tau*T) invariance tests
│   ├── 04_dimension_statistics.jl     # In-depth dims() analysis: mean vs median, fractions P(d=m), tol
│   ├── 05_phase_space_grids.jl        # 2D & 3D time-slice phase space grids (plot_phase_space_grid)
│   └── run_all_experiments.jl         # Master driver running full pipeline
├── plots/                             # Generated publication figures (PDF and PNG, NO axis titles)
│   ├── k_dependency/                  # K-neighbor dependency bands
│   ├── normalization/                 # Normalization comparisons (multipanel and fixed-K overlays)
│   ├── coordinate_invariance/         # Invariance proofs (dual-panel and standalone single-panel)
│   ├── dimension_distribution/        # Decomposition: mean/median/IQR, discrete fractions, tolerance sweeps
│   └── phase_space/                   # 2D & 3D phase space grids and LPCA colored scatter maps
└── results/                           # Numerical data tables (CSV)
    ├── k_dependency_summary.csv
    ├── normalization_comparison.csv
    ├── coordinate_invariance.csv
    └── dimension_statistics.csv
```

---

## 2. Graphic Standards for Scientific Papers

- **No Top/Axis Titles**: Following standard journal guidelines (e.g. *Physical Review*), all titles within the plot canvas have been completely removed. Captions are handled directly by LaTeX `\caption{...}`.
- **Descriptive File Naming**: Every `.pdf` (and corresponding `.png`) has an unambiguous, self-explanatory filename.
- **Standalone and Combined Plots**: Analyses are provided both as combined multi-panel figures and as separate single-panel figures for flexible layout in one-column or two-column paper formats.

---

## 3. Key Scientific Findings

### A. Sensitivity to Nearest Neighbors ($K$-Dependency)
- **Stability Window**: A robust plateau for local dimension estimation exists in the range $K \in [12, 48]$. In this regime, doubling $K$ results in narrow uncertainty bands ($|\Delta \langle d \rangle / \langle d \rangle| < 5\%$).
- **Small-$K$ Limit ($K < 10$)**: Susceptible to finite-density fluctuations and boundary effects, leading to slight noise at early proper times.
- **Large-$K$ Limit ($K \ge 100$)**: The local Euclidean neighborhood exceeds the local tangent space radius of curvature, introducing manifold curvature artifacts.

### B. Impact of Coordinate Normalization
- **Failure of Raw Units (`:none`)**: In physical units, temperature $T \sim 10^3\,\mathrm{MeV}$ (or several $\mathrm{fm}^{-1}$) vastly exceeds the dimensionless anisotropy $\mathcal{A} \in [-1, 6]$ and non-hydrodynamic mode $\mathcal{B} \in [-1, 1]$. Euclidean distance $\|x - y\|_2$ is dominated by $T$, causing $k$-NN to select points of similar temperature regardless of anisotropy.
- **Effective Normalizations (`:max`, `:minmax`, `:zscore`)**:
  - Abs-Max (`:max`) rescales columns to $[-1, 1]$ preserving zero-crossings and symmetry without introducing arbitrary offsets.
  - Min-Max (`:minmax`) scales strictly to $[0, 1]$.
  - Z-score (`:zscore`) centers around the mean with unit variance.
  All three normalized methods recover the true dimensionality reduction ($2 \to 1$ for MIS, $3 \to 2 \to 1$ for HJSW).

### C. Proof of Coordinate and Scale Invariance
- **Scale Invariance ($T \to 10T$)**: Under column-wise normalization (`:max`), scaling any coordinate by a constant scalar factor leaves the normalized point cloud unchanged:
  $$\frac{c \cdot X_{i,j}}{\max_k |c \cdot X_{k,j}|} = \frac{X_{i,j}}{\max_k |X_{k,j}|}$$
  Numerical verification demonstrates **exact zero difference** ($\Delta \langle d \rangle \equiv 0.00\times 10^0$) between $(T, \mathcal{A})$ and $(10T, \mathcal{A})$. Under raw units (`:none`), scaling artificially alters the dimension by up to $\Delta d = 1.00$.
- **Chart Transformation ($w = \tau T$)**: At each fixed proper time slice $\tau = \mathrm{const}$, the map $T \mapsto w = \tau T$ is a strictly linear coordinate dilation. Consequently, normalized LPCA yields identical dimension trajectories in $(T, \mathcal{A}, \mathcal{B})$ and $(w, \mathcal{A}, \mathcal{B})$.

### D. Deconstructing the `dims()` Function & Statistical Distribution
- **Discrete Nature**: The `dims()` algorithm computes an integer eigenvalue count $d_i \in \{1, \dots, D\}$ for each point.
- **Continuous Mean $\langle d \rangle$ vs Discrete Populations $P(d = m)$**:
  - The fractional values in $\langle d \rangle(\tau)$ reflect the coexistence of points that have already collapsed onto the 1D attractor with points still in the transient phase.
  - In Conformal MIS: $P(d=2)$ transitions smoothly from $100\%$ at $\tau = 0.2\,\mathrm{fm}/c$ to $0\%$ at $\tau \approx 3.5\,\mathrm{fm}/c$, with $P(d=1) \to 100\%$.
  - In HJSW: A two-stage transition is observed: $P(d=3) \to P(d=2) \to P(d=1)$, reflecting the decay of the transient non-hydrodynamic mode $\mathcal{B}$.
- **Median vs Mean**: The median $d_{1/2}$ exhibits sharp integer step transitions, while the mean $\langle d \rangle$ smoothly tracks the volume fraction of the attractor basin.

---

## 4. Complete Publication Figures Catalog

### K-Dependency
- `plots/k_dependency/k_dependency_mis_physical_coordinates_TA.pdf`: Bands across $K$ pairs in $(T, \mathcal{A})$ space (MIS).
- `plots/k_dependency/k_dependency_mis_dimensionless_coordinates_wA.pdf`: Bands across $K$ pairs in $(w, \mathcal{A})$ space (MIS).
- `plots/k_dependency/k_dependency_hjsw_physical_coordinates_TAB.pdf`: Bands across $K$ pairs in $(T, \mathcal{A}, \mathcal{B})$ space (HJSW).
- `plots/k_dependency/k_dependency_hjsw_dimensionless_coordinates_wAB.pdf`: Bands across $K$ pairs in $(w, \mathcal{A}, \mathcal{B})$ space (HJSW).

### Normalization Comparisons
- `plots/normalization/normalization_methods_multipanel_mis.pdf`: 4-panel grid comparing `:none`, `:max`, `:minmax`, `:zscore` for MIS.
- `plots/normalization/normalization_methods_multipanel_hjsw.pdf`: 4-panel grid comparing normalization methods for HJSW.
- `plots/normalization/normalization_methods_direct_overlay_fixed_k24_mis.pdf`: Direct overlay of all normalization curves at $K=24$ for MIS.
- `plots/normalization/normalization_methods_direct_overlay_fixed_k24_hjsw.pdf`: Direct overlay of all normalization curves at $K=24$ for HJSW.

### Coordinate and Scale Invariance
- `plots/coordinate_invariance/coordinate_and_scale_invariance_mis.pdf`: Dual-panel comparison: normalized vs raw for MIS.
- `plots/coordinate_invariance/coordinate_invariance_normalized_max_mis.pdf`: Single-panel plot showing exact collapse of $(T, \mathcal{A})$, $(w, \mathcal{A})$, and $(10T, \mathcal{A})$ for MIS.
- `plots/coordinate_invariance/coordinate_invariance_raw_none_mis.pdf`: Single-panel plot showing severe scale-dependence under raw coordinates (MIS).
- `plots/coordinate_invariance/coordinate_and_scale_invariance_hjsw.pdf`: Dual-panel comparison for HJSW.
- `plots/coordinate_invariance/coordinate_invariance_normalized_max_hjsw.pdf`: Single-panel exact invariance plot for HJSW.
- `plots/coordinate_invariance/coordinate_invariance_raw_none_hjsw.pdf`: Single-panel raw scale distortion plot for HJSW.

### Dimension Distribution & `dims()` Statistics
- `plots/dimension_distribution/dimension_statistics_and_fractions_mis.pdf`: Dual-panel: (a) Mean vs Median/IQR, (b) State populations $P(d=1), P(d=2)$ for MIS.
- `plots/dimension_distribution/dimension_mean_and_quantiles_mis.pdf`: Single-panel Mean vs Median/IQR for MIS.
- `plots/dimension_distribution/dimension_discrete_populations_mis.pdf`: Single-panel discrete populations $P(d=1), P(d=2)$ for MIS.
- `plots/dimension_distribution/dimension_statistics_and_fractions_hjsw.pdf`: Dual-panel for HJSW showing $3 \to 2 \to 1$ transition.
- `plots/dimension_distribution/dimension_mean_and_quantiles_hjsw.pdf`: Single-panel Mean vs Median/IQR for HJSW.
- `plots/dimension_distribution/dimension_discrete_populations_hjsw.pdf`: Single-panel discrete populations $P(d=1), P(d=2), P(d=3)$ for HJSW.
- `plots/dimension_distribution/eigenvalue_tolerance_cutoff_sensitivity.pdf`: Dual-panel eigenvalue tolerance comparison ($\mathrm{tol} \in [0.001, 0.02]$).
- `plots/dimension_distribution/eigenvalue_tolerance_cutoff_sensitivity_mis.pdf`: Single-panel tolerance sweep for MIS.
- `plots/dimension_distribution/eigenvalue_tolerance_cutoff_sensitivity_hjsw.pdf`: Single-panel tolerance sweep for HJSW.

### Phase Space Grids & Pointwise Local Dimension Maps (PDF only)
- `plots/phase_space/phase_space_grid_mis_coords_T_A.pdf`: Standard 2D 9-slice grid in $(T, \mathcal{A})$ for MIS.
- `plots/phase_space/phase_space_grid_mis_coords_w_A.pdf`: Standard 2D 9-slice grid in $(w, \mathcal{A})$ for MIS.
- `plots/phase_space/phase_space_grid_hjsw_coords_T_A_B.pdf`: Standard 3D 9-slice grid in $(T, \mathcal{A}, \mathcal{B})$ for HJSW.
- `plots/phase_space/phase_space_grid_hjsw_coords_w_A_B.pdf`: Standard 3D 9-slice grid in $(w, \mathcal{A}, \mathcal{B})$ for HJSW.
- `plots/phase_space/phase_space_colored_grid_mis_coords_T_A.pdf`: 2D 9-slice grid in $(T, \mathcal{A})$ with points colored by local dimension $d \in \{1, 2\}$.
- `plots/phase_space/phase_space_colored_grid_mis_coords_w_A.pdf`: 2D 9-slice grid in $(w, \mathcal{A})$ with points colored by local dimension $d \in \{1, 2\}$.
- `plots/phase_space/phase_space_colored_grid_hjsw_projections_A_B.pdf`: 2D 9-slice grid of the $(\mathcal{A}, \mathcal{B})$ plane in HJSW with points colored by $d \in \{1, 2, 3\}$, resolving the two-stage collapse.
- `plots/phase_space/phase_space_colored_grid_hjsw_coords_w_A.pdf`: 2D 9-slice grid in $(w, \mathcal{A})$ for HJSW with points colored by $d \in \{1, 2, 3\}$.
- `plots/phase_space/phase_space_colored_slice_mis_coords_*_tau_*.pdf`: Standalone high-resolution 2D slices for MIS at $\tau \in \{0.25, 0.65, 2.50\}\,\mathrm{fm}/c$.
- `plots/phase_space/phase_space_colored_slice_hjsw_coords_*_tau_*.pdf`: Standalone high-resolution 3D slices for HJSW at $\tau \in \{0.25, 0.65, 2.50\}\,\mathrm{fm}/c$ colored by $d \in \{1, 2, 3\}$.

### Parameterization Focus Tri-Panel Analysis (Dense K-Sweep)
- `plots/parameterization_focus/parameterization_focus_mis_none.pdf`: Tri-panel comparison for MIS in raw physical units (`:none`), showing strong divergence between coordinates.
- `plots/parameterization_focus/parameterization_focus_mis_max.pdf`: Tri-panel comparison for MIS under abs-max normalization (`:max`), showing exact coordinate and scale invariance.
- `plots/parameterization_focus/parameterization_focus_mis_minmax.pdf`: Tri-panel comparison for MIS under min-max normalization (`:minmax`).
- `plots/parameterization_focus/parameterization_focus_mis_zscore.pdf`: Tri-panel comparison for MIS under z-score standardization (`:zscore`).
- `plots/parameterization_focus/parameterization_focus_hjsw_none.pdf`: Tri-panel comparison for HJSW in raw physical units (`:none`).
- `plots/parameterization_focus/parameterization_focus_hjsw_max.pdf`: Tri-panel comparison for HJSW under abs-max normalization (`:max`).
- `plots/parameterization_focus/parameterization_focus_hjsw_minmax.pdf`: Tri-panel comparison for HJSW under min-max normalization (`:minmax`).
- `plots/parameterization_focus/parameterization_focus_hjsw_zscore.pdf`: Tri-panel comparison for HJSW under z-score standardization (`:zscore`).
