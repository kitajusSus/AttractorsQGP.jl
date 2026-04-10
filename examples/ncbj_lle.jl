
using LinearAlgebra
const dx = 1e-3

"""
    ncbj1_macierz_wszyskich_punktów(x::Vector{Float32})

Input:  wektor punktór `x` [1:10...] lub inne sposoby na uzycie wektorów


Output:

macierz punktów x_i [x,y]


```julia

function ncbj1_macierz_wszyskich_punktów!(x::Vector{Number})
    y = 1f-3 ./ x
    return hcat(x, y)'
end
```
"""
function ncbj1_macierz_wszystkich_punktow(x::AbstractVector{T}) where {T<:Number}
    y = 1 ./ x
    return hcat(x, y)'
end


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



"""
    ncbj3_calculate_wagi_dla_x_i(sasiedzi::Matrix{Float32}, x_i::Vector{Float32}; dx::Float32 = 1e-3)
"""
function ncbj3_calculate_wagi_dla_x_i(sasiedzi::AbstractMatrix{<:Real}, x_i::AbstractVector{<:Real}; dx=1e-3)
    # podmienia typy na wspólny w zależności od tego jaki jest typ elementów macierzy sasiedzi
    T = promote_type(eltype(sasiedzi), eltype(x_i), typeof(dx))
    S = Matrix{T}(sasiedzi)
    x = Vector{T}(x_i)
    nn = size(S, 2)
    Z = S .- x
    C = Z' * Z
    C += (T(dx) * tr(C)) * I
    w = C \ ones(T, nn)
    return w ./ sum(w)
end


# ncbj4_lle(macierz_punktow::Matrix{Float32}, nn::Int; dx::Float32 = 1e-3)
# dodałem stały typ T<:AbstractFloat, żeby można było używać zarówno Float32 jak i Float64
# i z górki
#  zmieniłem typ na AbstractMatrix by nie było probelmu z transponowaniem macierzy
function ncbj4_lle(macierz_punktow::AbstractMatrix{T}, nn::Int; dx=dx) where {T<:AbstractFloat}
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




using GLMakie 

# tutaj rysune i oblicza 
function ncbj_plot_examples_lle(X, labels, K, N=5000)
    W = ncbj4_lle(X, K)
    M = (I - W)' * (I - W)
    F = eigen(Symmetric(M))
    Y = F.vectors[:, 2:3]'
    
    fig = Figure(size = (1200, 600))
    
    ax_3d = Axis3(fig[1, 1], title = "Oryginalna rozmaitość X (3D)", azimuth = 0.22 * π)
    scatter!(ax_3d, X[1, :], X[2, :], X[3, :], color = labels, colormap = :jet, markersize = 8)

    ax_2d = Axis(fig[1, 2], title = "Zredukowana przestrzeń Y (2D)")
    scatter!(ax_2d, Y[1, :], Y[2, :], color = labels, colormap = :jet, markersize = 8)
    
    return fig
end

function examples_lle_data(dataset = "matlab", N = 5000, thickness = 0.5)
    if dataset == "matlab"
        τ = (1.5 * π) .* (1.0 .+ 2.0 .* rand(Float32, N))
        h = 21 .* rand(Float32, N)
        r = τ .+ (thickness .* randn(Float32, N))
        X = [r .* cos.(τ)  h  r .* sin.(τ)]'
        return X, τ
    elseif dataset == "sruba"            
        t = LinRange(0, 4π, N)
        x = sin.(t) .+ 0.1 .* randn(Float32, N)
        y = cos.(t) .+ 0.1 .* randn(Float32, N)
        z = t .+ 0.1 .* randn(Float32, N)
        X = [x y z]'
        return X, t
    elseif dataset == "scurve" 
        𝛉 = LinRange(-1.5π, 1.5π, N)
        x = sin.(𝛉) .+ 0.1 .* randn(Float32, N)
        y = LinRange(0, 5, N) .+ 0.01 .* randn(Float32, N) 
        z = sign.(𝛉) .+ 0.1 .* randn(Float32, N)
        X = [x y z]'
        return X, 𝛉
    else 
        return nothing, nothing
    end
end
#
# function  matlab(N = 5000)
#     τ = (1.5 * π) .* (1.0 .+ 2.0 .* rand(Float32, N))
#     h = 21 .* rand(Float32, N)
#     X = [τ .* cos.(τ)  h  τ .* sin.(τ)]'
#     return X
# end
# function sruba(N = 5000)
#     t = LinRange(0, 4π, N)
#     x = sin.(t) .+ 0.1 .* randn(Float32, N)
#     y = cos.(t) .+ 0.1 .* randn(Float32, N)
#     z = t .+ 0.1 .* randn(Float32, N)
#     X = [x y z]'
#     return X
# end
#
#
# function scurve(N = 5000)
#     𝛉 = LinRange(-1.5π, 1.5π, N)
#     x = sin.(𝛉) .+ 0.1 .* randn(Float32, N)
#     y = LinRange(0,5,N) .+ 0.01 .* randn(Float32, N) 
#     z = 𝛉 ./ abs.(𝛉) .+ 0.1 .* randn(Float32, N)
#     X = [x y z]'
#     return X
# end
