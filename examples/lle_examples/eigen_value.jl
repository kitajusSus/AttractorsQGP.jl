using LinearAlgebra
include("../ncbj_lle.jl")
using GLMakie


function ex_eigen_value(; N=2000, K=40)
    X, labels = swissroll_dane(N)
    
    W = ncbj4_lle_basic(X, K)

    M = (I - W)' * (I - W)
    F = eigen(Symmetric(M))
    wartosci_wlasne = F.values
    indeksy = 1:10
    wartosci_do_wykresu = wartosci_wlasne[indeksy]

    fig = Figure()
    ax = Axis(fig[1, 1], title = "Wartości własne", xlabel = "Indeks", ylabel = "Wartosc")
    scatterlines!(ax, indeksy, wartosci_do_wykresu, markersize=10)
    display(fig)
    return W, F.values

end



ex_eigen_value()
