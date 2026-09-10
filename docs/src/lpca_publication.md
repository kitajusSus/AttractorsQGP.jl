# Identifying Hydrodynamic Attractors with Local PCA (LPCA)

This documentation guide explains how to systematically estimate the effective dimensionality of expanding Quark-Gluon Plasma (QGP) manifolds using **Local Principal Component Analysis (LPCA)** and the new **Soft-Weighted LPCA** algorithm.

---

## 1. Physical Background & Methodology

In relativistic heavy-ion collisions, the expanding fireball undergoes rapid longitudinal expansion (Bjorken flow). The system transitions from far-from-equilibrium initial states toward a universal, low-dimensional **hydrodynamic attractor**:
- **Conformal MIS Model**: 2D phase space $(T, \mathcal{A}) \longrightarrow$ collapses to 1D attractor line ($2 \to 1$).
- **HJSW Model**: 3D phase space $(T, \mathcal{A}, \mathcal{B}) \longrightarrow$ two-stage collapse: fast decay of transient non-hydrodynamic mode $\mathcal{B} \sim e^{-\Omega_I \tau T}$ ($3 \to 2$), followed by slow attraction onto the 1D hydrodynamic manifold $\mathcal{A} \to \mathcal{A}_{\text{attr}}(w)$ ($2 \to 1$).

### Challenges of Raw LPCA
1. **Coordinate Disparity**: Physical temperature $T \sim 10^3\,\mathrm{MeV}$ dominates anisotropy $\mathcal{A} \in [-1, 6]$ in Euclidean distance without column normalization. Rescaling (`:max` or `:zscore`) restores exact scale invariance.
2. **Discrete Popping (Threshold Jitter)**: Hard eigenvalue cutoff $\Theta(\lambda_j/\lambda_1 - \text{tol})$ produces sudden integer jumps.
3. **Desynchronization**: Cold and hot fluid elements at the same proper time $\tau$ have different relaxation ages $w = \tau T$.

### The Soft-Weighted LPCA Solution
- **Sigmoidal Spectral Cutoff**:
  ```math
  d_i^{\text{soft}} = 1 + \sum_{j=2}^D \sigma\left(\frac{\lambda_j / \lambda_1 - \text{tol}}{\Delta}\right), \quad \sigma(z) = \frac{1}{1 + e^{-z}}
  ```
- **Density Weighting**:
  ```math
  W_i^{\text{dens}} = \exp\left(-\frac{R_K(x_i)^2}{2\sigma_R^2}\right)
  ```
  Eliminates spurious boundary distortions from the edges of the initial condition box.
- **Weighted Ensemble Average & Band**:
  ```math
  \langle d \rangle_W = \frac{\sum_i W_i d_i^{\text{soft}}}{\sum_i W_i}, \quad \sigma_W = \sqrt{\frac{\sum_i W_i (d_i^{\text{soft}} - \langle d \rangle_W)^2}{\sum_i W_i}}
  ```

---

## 2. Interactive Julia REPL Quickstart

Start Julia in the project root:
```bash
julia --project=.
```

### Loading Data and Packages
```julia
using AttractorsQGP
using CairoMakie  # Use CairoMakie for vector PDF / PNG export
# using GLMakie   # Or use GLMakie for interactive 3D window rotation

# Load dataset (supports HDF5 .h5, CSV .csv, or Serialization .jls)
dataset_mis = load_dataset("datasets/hjsw_lpca/mis_matched_seed5.h5")
dataset_hjsw = load_dataset("datasets/hjsw_lpca/hjsw_matched_seed5.h5")
```

### A. Quick Soft-Weighted LPCA Scan
```julia
# 1. Define evaluation time slices tau in fm/c
tau_grid = [0.2, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

# 2. Run Soft-Weighted LPCA on HJSW data [columns: tau, T, A, B]
hjsw_res = scan_soft_weighted_dimension(
    dataset_hjsw,
    tau_grid;
    feature_indices = [2, 3, 4], # T, A, B
    k = 20,                      # number of nearest neighbors
    tol = 0.02,                  # spectral cutoff threshold
    delta = 0.005                # sigmoidal transition softness
)

# 3. Plot dimension with ±1σ_W weighted band and hard LPCA comparison
fig = plot_soft_weighted_dimension(hjsw_res; compare_hard = true)

# Display or save
save("hjsw_soft_dimension.pdf", fig)
```

### B. Testing Nearest Neighbor ($K$) Stability
```julia
# Test sensitivity to K across [6, 12, 24, 48]
fig_k = plot_soft_k_dependency(
    dataset_hjsw,
    [6, 12, 24, 48],
    tau_grid;
    feature_indices = [2, 3, 4],
    tol = 0.02,
    delta = 0.005
)
save("hjsw_soft_k_dependency.pdf", fig_k)
```

### C. Continuous Phase Space Dimension Mapping
```julia
# 2D continuous dimension map for MIS at tau = 0.65 fm/c
fig_2d = plot_soft_phase_space_slice_2d(
    dataset_mis,
    0.65;
    feature_indices = [2, 3],
    x_label = L"T\,[\mathrm{fm}^{-1}]",
    y_label = L"\mathcal{A}",
    color_limits = (1.0, 2.0)
)
save("mis_slice_continuous_tau_0.65.pdf", fig_2d)

# 3D continuous dimension map for HJSW (T, A, B)
fig_3d = plot_soft_phase_space_slice_3d(
    dataset_hjsw,
    0.65;
    feature_indices = [2, 3, 4],
    color_limits = (1.0, 3.0)
)
save("hjsw_slice_continuous_3d.pdf", fig_3d)

# 9-slice grid across time evolution
grid_taus = [0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.5, 2.5]
fig_grid = plot_soft_phase_space_grid_2d(
    dataset_mis,
    grid_taus;
    feature_indices = [2, 3],
    color_limits = (1.0, 2.0)
)
save("mis_grid_continuous.pdf", fig_grid)
```

---

## 3. How to Tweak Parameters & What They Mean

| Parameter | Default | Physical & Numerical Intuition |
| :--- | :--- | :--- |
| `k` | `16 – 24` | **Nearest neighbors**. Small $K < 8$ suffers from sampling noise. Large $K > 60$ covers too large a volume, picking up manifold curvature $\lambda \sim \kappa^2 r^4$. The optimal plateau is $K \in [12, 36]$. |
| `tol` | `0.015 – 0.02` | **Relative eigenvalue cutoff** ($\lambda_j / \lambda_1$). Defines when a transverse thickness is considered collapsed relative to the main tangent directions. |
| `delta` ($\Delta$) | `0.005` | **Softness bandwidth**. Controls the width of the logistic transition. $\Delta \to 0$ recovers the hard step function $\Theta$. $\Delta \approx 0.005$ provides smooth differentiability without distorting the integer plateau. |
| `normalize_method` | `:max` | **Coordinate rescaling**. `:max` scales columns to $[-1, 1]$, preserving zero-crossings and providing exact scale and coordinate chart invariance. Other options: `:minmax`, `:zscore`, `:none`. |
| `use_density_weights`| `true` | When `true`, applies Gaussian weighting based on $R_K(x_i)$, downweighting boundary points on the edges of the initial condition box. |
| `w_focus` / `sigma_w` | `nothing` | Optional filter to select fluid elements centered around a specific relaxation age $w_0 = \tau T$. |

---

## 4. Reproducing Manuscript Figures from the REPL

You can reproduce any paper figure with a single high-level function call:

### Figure 1: $K$-Dependency Uncertainty Bands
```julia
k_pairs = [(3, 6), (6, 12), (12, 24), (24, 48), (100, 200)]
k_results = evaluate_k_dependency(dataset_mis, k_pairs, tau_grid; feature_indices = [2, 3])
fig1 = plot_k_dependency_bands(k_results)
save("fig1_k_dependency.pdf", fig1)
```

### Figure 2: Normalization Method Comparison (4 Panels)
```julia
methods = [:none, :max, :minmax, :zscore]
norm_results = compare_normalization_methods(dataset_mis, methods, k_pairs, tau_grid; feature_indices = [2, 3])
fig2 = plot_normalization_multipanel(norm_results; methods_to_plot = methods)
save("fig2_normalizations.pdf", fig2)
```

### Figure 3: Coordinate & Scale Invariance Proof ($T \to 10T$, $w = \tau T$)
```julia
variants = prepare_dataset_variants(dataset_mis, :mis)
fig3 = plot_coordinate_invariance(variants, 24, tau_grid)
save("fig3_invariance.pdf", fig3)
```

### Figure 4: Statistical Decomposition & Discrete Fractions $P(d = m)$
```julia
stats = analyze_dimension_distribution(dataset_hjsw, 24, tau_grid; feature_indices = [2, 3, 4])
fig4 = plot_dimension_distribution(stats; model_name = "HJSW Model")
save("fig4_dimension_stats.pdf", fig4)
```

### Figure 5: Parameterization Focus 2x2 with Manifold Tunnel Envelopes
```julia
dense_k = collect(3:3:96)
sweep = compute_parameterization_k_sweep(variants, dense_k, tau_grid; normalize_method = :max)
fig5 = plot_parameterization_focus_2x2(sweep)
save("fig5_parameterization_focus.pdf", fig5)
```

---

## 5. Participation Ratio (PR) Dimension (Parameter-Free Method)

The Participation Ratio (PR) calculates the effective number of active phase space dimensions from the eigenvalue spectrum of the local covariance matrix without introducing any arbitrary cutoff thresholds (`tol`) or sigmoid parameters (`delta`):

```math
d_{\mathrm{PR}}(x_i) = \frac{(\mathrm{Tr}\, C_i)^2}{\mathrm{Tr}(C_i^2)} = \frac{\left(\sum_{j=1}^D \lambda_j\right)^2}{\sum_{j=1}^D \lambda_j^2}
```

- When $D$ modes are equally active: $d_{\mathrm{PR}} \approx D$.
- When $m$ modes dominate and the rest vanish: $d_{\mathrm{PR}} \approx m$.
- On the 1D hydrodynamic attractor: $d_{\mathrm{PR}} \to 1.0$.

### Interactive REPL Recipe

```julia
using AttractorsQGP
using CairoMakie

# Load HJSW dataset
dataset_hjsw = load_hydro_dataset("datasets/hjsw_lpca/hjsw_matched_seed5.h5")
tau_grid = [0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.5, 5.0, 7.5, 10.0]

# Compute Participation Ratio evolution across tau
pr_scan = scan_local_pr_dimension(dataset_hjsw, tau_grid; feature_indices = [2, 3, 4], k = 20)

# Optional: compare against Soft-Weighted LPCA
soft_scan = scan_soft_weighted_dimension(dataset_hjsw, tau_grid; feature_indices = [2, 3, 4], k = 20)

fig = plot_local_pr_dimension(pr_scan; soft_scan = soft_scan)
save("pr_vs_soft_hjsw.pdf", fig)

# Plot local PR dimension distribution in 3D phase space (using :devon colormap)
fig_3d = plot_pr_phase_space_slice_3d(dataset_hjsw, 0.65; feature_indices = [2, 3, 4], colormap = :devon)
save("pr_phase_space_slice_hjsw_3d_tau_0.65.pdf", fig_3d)
```

---

## 6. Running Automated Script Pipelines

To re-run the entire batch of publication experiments (Steps 1 through 8) and generate all figures at once:
```bash
julia --project=. publication_lpca/scripts/run_all_experiments.jl
```
All vector PDFs will be placed in `publication_lpca/plots/`, and numerical data tables in `publication_lpca/results/`.
