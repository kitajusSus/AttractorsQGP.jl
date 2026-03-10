
# AtractorsQGP.jl User Guide

This guide provides a comprehensive walkthrough of the core functionalities offered by AtractorsQGP.jl. It covers theoretical background, fundamental data structures, simulation workflows, and advanced analysis tools.

## 1. Theoretical Background and Models

The package solves the hydrodynamic evolution of the Quark-Gluon Plasma (QGP) under the assumption of longitudinal Bjorken expansion. The state of the system at any proper time $\tau$ is described by a 2D phase space:

* $T$: Temperature
* $\mathcal{A}$: Pressure anisotropy, defined related to the shear stress tensor.

The evolution is governed by the following coupled ordinary differential equations (ODEs):

$$\frac{dT}{d\tau}=\frac{T}{\tau}\left(-\frac{1}{3}+\frac{\mathcal{A}}{18}\right)$$

$$\frac{d\mathcal{A}}{d\tau}=\frac{1}{\tau_\pi\tau}\left[8\frac{\eta}{s}-\tau T\left(\mathcal{A}+\frac{\lambda_1}{12\eta/s}\mathcal{A}^2\right)-\frac{2}{9}\tau_\pi\mathcal{A}^2\right]$$

### Data Structures for Models

The transport parameters ($\eta/s$, $\tau_\pi$, $\lambda_1$) are stored in the `HydroParams` struct:

```julia
struct HydroParams{T<:Real}
    eta_over_s::T
    tau_pi::T
    lambda1::T
end

```

You can initialize specific models that automatically configure these parameters:

```julia
using AtractorsQGP

# BRSSS Model (includes all terms)
model_brsss = BRSSSModel(eta_over_s = 0.08, tau_pi = 0.1, lambda1 = 0.05)

# MIS Model (Müller-Israel-Stewart) - strictly enforces lambda1 = 0
model_mis = MISModel(eta_over_s = 1/(4*pi))

```

## 2. Generating Initial Conditions

To study the attractor behavior, we generate an ensemble of initial conditions. The `generate_initial_conditions` function creates a set of random starting points in the $(T,\mathcal{A})$ phase space.

```julia
# Generate 1000 initial conditions
# Note: T_range is provided in MeV, but internally converted to fm⁻¹
ics = generate_initial_conditions(1000;
    T_range=(400.0, 2500.0),
    A_range=(-10.0, 20.0),
    temperature_unit=:MeV,
    seed=5
)

```

The resulting `ics` object is a `Vector` of `SVector{2, Float64}`, ensuring high performance during the ODE solving phase.

## 3. Solving Hydrodynamic Equations

The core simulation relies on DifferentialEquations.jl to solve the ODEs. You can generate trajectories over a specified proper time span ($\tau$).

```julia
tspan = (0.22, 1.2) # proper time in fm/c

# Solve for all initial conditions (supports multithreading)
solutions = generate_trajectories(model_brsss, ics, tspan; parallel=:threads)

```

### The Dataset Matrix

For ease of analysis (like PCA), the vector of ODE solutions is typically flattened into a dense 2D matrix using `build_dataset`.

```julia
dataset = build_dataset(solutions; temperature_unit=:fm)

```

The resulting dataset is a `Matrix{Float64}` with exactly 3 columns:

1. **tau** ($\tau$)
2. **T** (Temperature)
3. **A** ($\mathcal{A}$, Anisotropy)

Each row represents a single snapshot of a trajectory at a specific time.

## 4. Dimensionality Reduction (PCA)

To understand how the dimensionality of the system evolves and collapses onto the attractor, the package performs Principal Component Analysis (PCA) independently for each time slice $\tau$.

```julia
# Standard linear PCA (using min-max normalization by default)
pca_results = run_pca_per_time(dataset; n_components=2, method=:minmax)

# Kernel PCA for non-linear dimensionality reduction (using RBF kernel)
pca_kernel_results = run_pca_per_time(dataset; n_components=2, method=:kernel, gamma=0.5)

```

The returned object contains the `explained_variance_ratio` (EVR), which quantitatively shows how much variance is captured by the principal components as time progresses.

## 5. Dynamical Systems Analysis

Beyond PCA, AtractorsQGP.jl provides tools to analyze the system from the perspective of chaos theory and dynamical systems.

### Potencial Way for upgrades
While I am doing my research sometimes I find out about interesting algorithms:

#### Lyapunov Exponent

Measures the rate of separation of infinitesimally close trajectories. A negative LLE strongly indicates the presence of an attracting manifold.

```julia
u0 = [2.0, 5.0] # [T0 in fm⁻¹, A0]
lle = run_LLE(model_brsss, u0, (0.22, 5.0); perturbation=1e-6)
println("Local Lyapunov Exponent: ", lle)

```

### Intrinsic Dimension

Estimates the intrinsic geometric dimension of the data cloud at a given time using the participation ratio of the covariance matrix eigenvalues.

```julia
_, X_tau = get_tau_slice(dataset, 0.5)
dim = estimate_dimension(X_tau)

```

## 6. Visualization
My own way of ploting data used in my Thesis


```julia
# Plot the thermodynamic evolution of T and A
fig1 = plot_thermodynamics_evolution(dataset)

# Plot the Explained Variance Ratio (EVR) over time
fig2 = plot_pca_evr_over_time(dataset; n_components=2, method=:minmax)

# Plot phase space snapshots at specific times
fig3 = plot_phase_space_grid(dataset, [0.3, 0.5, 0.7], :tauT, :A)

```

## 7. Saving and Loading Data

To avoid re-running expensive simulations, you can save your generated datasets. The package determines the correct format based on the file extension.

```julia
# Save to HDF5 format (Recommended for large datasets)
save_dataset("simulation_data.h5", dataset)

# Save to CSV (Easier for inspection)
save_dataset("simulation_data.csv", dataset)

# Save to native Julia Serialized format
save_dataset("simulation_data.jls", dataset)

# Load the data back (format is auto-detected)
loaded_data = load_dataset("simulation_data.h5")

```
