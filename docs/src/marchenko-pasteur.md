# Marchenko Pasteur distribution

# 11 November

Today I am going to check how the Marchenko-Pasteur distribudion [Wikipedia link](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution)
work.

Main idea is that Matricies(macierze) $\mathcal{X}$ with Covariance Matrix
$$C_{\text{THEORETICAL}} = I \text{only ones on the diag} = \frac{1}{T}X^{T} X$$

And only because of maybe some random events and `statistical fluctuations` as it was called in someplace.
Some variables look corelalated, but they are not. Which provides us information about

$$C_{\text{real}} =\ I$$
Few $\lambda$ may be >1 or $\lambda <1$

## Theorem of Marchenko-Pasteur

They proved that with dataset with dimension $D$ and $N$ number of observations
and their proportion converges to the constant:
$$q = \frac{D}{N} <= 1$$

....

## Code

This function counts the marchenko $q$ and shows the plateu

```julia


function marchenko(
        dane::AbstractMatrix{<:Real};
        tau::Real = 0.2,
        tau0::Real = 0.2,
        K::Integer = 160,
        T_scale::Real = 193.0,
        K_sweep = [6:4:250...],
        D::Int = 3,
        atol::Real = 1.0e-5
    )


```
