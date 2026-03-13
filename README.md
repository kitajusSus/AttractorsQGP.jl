
# Thesis
> [!IMPORTANT]
> You can find docs in here [docs link ](https://kitajussus.github.io/Atractors-in-QGP/dev/)

```bash
git clone {repo html}
cd Atractors-in-QGP
julia --project=.


```




```julia
using AtractorsQGP
dane = load_dataset("datasets/data_testSmall.jls")
plot_lle_dim(dane, 3,2,0.35)
plot_lle_dim(dane, 3,2,0.40)
plot_lle_dim(dane, 6,2,0.40)
plot_lle_dim(dane, 6,2,0.22)
plot_lle_dim(dane, 7,2,0.22)
using CairoMakie

for i in 1:20
           fig = plot_lle_dim(dane, i, 2, 0.22)
           display(fig)
            println("dla $i, daj enter")
           readline()
end

```

# Hydrodynamic Attractors in Phase Space
Wszystkie Artykuły znajdują się w [HR Articles]

