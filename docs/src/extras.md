# Extras
Sometimes I can find techniques that see usefull for my work, I will save them  here


## Dynamical Systems Analysis

Beyond PCA, AtractorsQGP.jl provides tools to analyze the system from the perspective of chaos theory and dynamical systems.

### Potencial Way for upgrades
While I am doing my research sometimes I find out about interesting algorithms:

#### Lyapunov Exponent

Measures the rate of separation of infinitesimally close trajectories. A negative LLE strongly
indicates the presence of an attracting manifold.

```julia
u0 = [2.0, 5.0] # [T0 in fm⁻¹, A0]
lle = run_LLE(model_brsss, u0, (0.22, 5.0); perturbation=1e-6)
println("Local Lyapunov Exponent: ", lle)

```

#### Intrinsic Dimension

Estimates the intrinsic geometric dimension of the data cloud at a given time
using the participation ratio of the covariance matrix eigenvalues.

```julia
_, X_tau = get_tau_slice(dataset, 0.5)
dim = estimate_dimension(X_tau)

```




## youtube


### Dimensionality

- [The Curse of Dimensionality youtube ](https://www.youtube.com/watch?v=9Tf-_mJhOkU)

How distances Increse in higher dimensions?
How to indentify which features are more significant/important/relevant


### LLE
- [Localy Linear Embeding Lecture](https://www.youtube.com/watch?v=scMntW3s-Wk&t=59s)

Shows how PCA and PCA-kernel got a little problems with holding the structure of data

> How does it work?
> [!IMPORTANT]
> $\epsilon(W) = \sum_i = |\vec{X}_i - \sum_j W_{ij}\vec{Y}_j|^2$
Computing a set of weights that can be used for reconstructing a point






