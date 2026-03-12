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






## Testing the `@normalize_minmax` function


``` julia



julia> testy = run_main(MISModel(), n_points=100, )
run_main(model::AbstractHydroModel; n_points, tspan, T_range, A_range, saveat, seed) @ AtractorsQGP ~/github/Atractors-in-QGP/src/AtractorsQGP.jl:59
julia> testy = run_main(MISModel(), n_points=100, tspan=(0.25, 0.26), seed  = 5)
julia> save_dataset("datasets/testy_100.h5", tests.dataset)
julia> save_dataset("datasets/testy_100.h5", testy.dataset)
"datasets/testy_100.h5"

julia> test = load_dataset("datasets/testy_100.h5")
200×3 Matrix{Float64}:
julia> norm_test=normalize_minmax(test)

julia> using CairoMakie

julia> set_publication_theme()

julia> scatter!()
julia> scatter!(norm_test[1], norm_test[2])
julia> fig = Figure()
julia> fig = Figure(size=(1000, 1000))

julia> scatter!(norm_test[1], norm_test[2])
julia> ax1 = Axis(fig,
               title=L"TESTY DLA tests_100.h5 \tau",
               xlabel=L"X",
               ylabel=L"Y ",
           )
julia> scatter!(ax1, norm_test[1], norm_test[2])
julia> # norm_test[1] zwraca całą macierz znormalizowanych danych
       # norm_test[1][:, 1] zwraca pierwszą kolumnę tej macierzy

       scatter!(ax1, norm_test[1][:, 1], norm_test[1][:, 2])
Scatter{Tuple{Vector{Point{2, Float64}}}}

julia> fig

julia> linkaxes!(ax1)

julia> fig

julia> fig
julia> fig = Figure(size=(1000, 1000))
julia> ax1 = Axis(fig[1, 1],
           title=L"TESTY DLA tests_100.h5 \tau",
           xlabel=L"X",
           ylabel=L"Y"
       )
julia> scatter!(ax1, norm_test[1][:, 2], norm_test[1][:, 3])
Scatter{Tuple{Vector{Point{2, Float64}}}}
julia> fig
julia> ax1 = Axis(fig[1, 1],
           title=L"\text{TESTY DLA tests_100.h5}w \tau",
           xlabel=L"X",
           ylabel=L"Y"
       )
julia> fig
julia> fig = Figure(size=(1000, 1000))
julia> ax1 = Axis(fig[1, 1],
           title=L"\text{TESTY DLA tests_100.h5}w \tau",
           xlabel=L"X",
           ylabel=L"Y"
       )
julia> fig
julia> scatter!(ax1, norm_test[1][:, 2], norm_test[1][:, 3])
Scatter{Tuple{Vector{Point{2, Float64}}}}
julia> fig
julia> scatter!(ax1, norm_test[1][:, 1], norm_test[1][:, 3])
Scatter{Tuple{Vector{Point{2, Float64}}}}
julia> fig
julia> scatter!(ax1, norm_test[1][:, 1], norm_test[1][:, 2])
Scatter{Tuple{Vector{Point{2, Float64}}}}
julia> fig
julia> scatter!(ax1, norm_test[1][:, 1], norm_test[1][:, 1])
Scatter{Tuple{Vector{Point{2, Float64}}}}
julia> fig
