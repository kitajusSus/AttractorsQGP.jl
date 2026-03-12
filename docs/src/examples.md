# Examples of usage




## Local Linear Embeding

To see  and `fine-tune` parameter `k`   for your dataset I propose this way of
repl code.



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




