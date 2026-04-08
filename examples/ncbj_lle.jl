
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
regularyzacji Tichonowa. (ta z C_{c} + \eta I)

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


