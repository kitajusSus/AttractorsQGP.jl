
import Pkg
Pkg.activate(joinpath(@__DIR__, "../..")) 

using AtractorsQGP





using GLMakie 
using LinearAlgebra
const dx = 1e-3

# skopiowane z src/examples/ncbj_lle.jl



"""
    ncbj2_sąsiedzi!(macierz_punktow::Matrix{Float32}, dla_jakiego_punktu::Int, nn::Int)
- macierz punktów x_i najlepiej zrobiony z funkcji typu `ncbj1_macierz_wszyskich_punktów!()`
- nn - liczba sąsiadów do rozpatrywania

```julia
return 𝛈, indeksy_𝛈 , x_i
```
return - 𝛈 - macierz sąsiadów dla punktu i
indeksy_𝛈 - indeksy sąsiadów w macierzy punktów
x_i - punkt dla którego szukamy sąsiadów
"""
function ncbj2_sasiedzi(macierz_punktow::AbstractMatrix{T}, dla_jakiego_punktu::Int, nn::Int) where {T<:AbstractFloat}
    N = size(macierz_punktow, 2)
    x_i = @view macierz_punktow[:, dla_jakiego_punktu]
    distanse = zeros(T,N)

    for j in 1:N
       @views  distanse[j] = norm(macierz_punktow[:, j] - x_i)
    end

    indeksy_𝛈 = sortperm(distanse)[2 : nn + 1]
    𝛈 = zeros(T, size(macierz_punktow, 1), nn)

    for k in 1:nn
        𝛈[:, k] = macierz_punktow[:, indeksy_𝛈[k]]
    end
# trzeba tutaj dodać element by wychdoziły w tym samym typie
    # rzeczy jak \eta i x_i, bo inaczej będzie problem z typami w dalszych obliczeniach
    return 𝛈, indeksy_𝛈 , collect(x_i)
end

"""
    lle3_svd_wagi_dla_x_i(sasiedzi::AbstractMatrix{<:Real}, x_i::AbstractVector{<:Real}, d::Int)

dla punktu `x_i` na podstawie jego sąsiadów, wykorzystując rozkład SVD. Metoda eliminuje konieczność 
regularyzacji tzw Tichonowa. (ta z C_{c} + \eta I)

- `sasiedzi`: Macierz sąsiadów dla punktu x_i (zwracana np. przez ncbj2_sasiedzi)
- `x_i`: Rozpatrywany punkt w przestrzeni fazowej
- `d`: Przewidywany wymiar docelowej rozmaitości (atraktora)

- wektor wag `w` o długości równej liczbie sąsiadów, sumujący się do 1.
"""
function ncbj3_svd_wagi_dla_x_i(sasiedzi::AbstractMatrix{<:Real}, x_i::AbstractVector{<:Real}, d::Int)
    T = promote_type(eltype(sasiedzi), eltype(x_i))
    S = Matrix{T}(sasiedzi)
    x = Vector{T}(x_i)
    
    𝛈 = size(S, 2)
    Z = S .- x
    
    # Artykuł definiuje macierz otoczenia X_i jako macierz K x D,
    # dlatego transponujemy nasze Z
    X_i = Z'
    
    # Wykonujemy rozkład SVD. Używamy full=true, aby macierz U miała
    # pełny wymiar K x K, co jest wymagane do wyizolowania podprzestrzeni szumu.
    F = svd(X_i, full=true)
    U = F.U
    
    # U_2 to podmacierz zawierająca kolumny od d+1 do K, opisująca szum
    U_2 = U[:, d+1:𝛈]
    
    # Wektor jedynek o długości równej liczbie sąsiadów
    ones_vec = ones(T, 𝛈)
    
    #  w = (U_2 * U_2' * 1) / (1' * U_2 * U_2' * 1).
    🥕 = U_2' * ones_vec
    licznik = U_2 * 🥕
    mianownik = dot(ones_vec, licznik)
    
    w = licznik ./ mianownik
    
    return w
end



function ncbj4_lle_basic(macierz_punktow::AbstractMatrix{T}, nn::Int; dx=dx) where {T<:AbstractFloat}
    N = size(macierz_punktow, 2)
    W = zeros(T, N, N)

    for i in 1:N
        sasiedzi, indeksy, x_i = ncbj2_sasiedzi(macierz_punktow, i, nn)
        w = ncbj3_calculate_wagi_dla_x_i(sasiedzi, x_i; dx=dx)
        @inbounds for k in 1:nn
            W[i, indeksy[k]] = w[k]
        end
    end

    return W
end

function ncbj4_lle_svd(macierz_punktow::AbstractMatrix{T}, nn::Int) where {T<:AbstractFloat}
    N = size(macierz_punktow, 2)
    W = zeros(T, N, N)

    for i in 1:N
        sasiedzi, indeksy, x_i = ncbj2_sasiedzi(macierz_punktow, i, nn)
        w = ncbj3_svd_wagi_dla_x_i(sasiedzi, x_i, 2) # zakładamy docelowy wymiar 2
        @inbounds for k in 1:nn
            W[i, indeksy[k]] = w[k]
        end
    end

    return W
end


function ncbj5_nowy_manifold(W)
    N = size(W, 1)
    M = (I - W)' * (I - W)
    F = eigen(Symmetric(M))
    Y = F.vectors[:, 2:3]'.*sqrt(N)
    
    return Y
end

function plot_examples_lle(X, Y, labels)
    fig = Figure(size = (1200, 600))
    
    ax_3d = Axis3(fig[1, 1], title = "Oryginalna rozmaitość X (3D)", azimuth = 0.22 * π)
    scatter!(ax_3d, X[1, :], X[2, :], X[3, :], color = labels, colormap = :jet, markersize = 8)

    ax_2d = Axis(fig[1, 2], title = "Zredukowana przestrzeń Y (2D)")
    scatter!(ax_2d, Y[1, :], Y[2, :], color = labels, colormap = :jet, markersize = 8)
    
    return fig
end






function helix_dane(N)
    t = LinRange(0f0, 4f0 * π, N)
    x = sin.(t) .+ 0.1f0 .* randn(Float32, N)
    y = cos.(t) .+ 0.1f0 .* randn(Float32, N)
    z = t .+ 0.1f0 .* randn(Float32, N)
    X = [x y z]'
    return X, t
end

function ex_helix(; N=2000, K=12)
    X, labels = helix_dane(N)
    wagi = ncbj4_lle_basic(X, K)
    Y = ncbj5_nowy_manifold(wagi)
    fig = plot_examples_lle(X, Y, labels)
    display(fig)
    println("Naciśnij Enter, aby zakończyć...")
    readline()
    return fig
end



@time ex_helix()
