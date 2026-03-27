
using LinearAlgebra
const dx = 1e-3

"""
    ncbj1_macierz_wszyskich_punktów(x::Vector{Float32})

Input:  wektor punktór `x` [1:10...] lub inne sposoby na uzycie wektorów


Output:

macierz punktów x_i [x,y]


```julia

function ncbj1_macierz_wszyskich_punktów!(x::Vector{Float64})
    y = 1f-3 ./ x
    return hcat(x, y)'
end
```
"""
function ncbj1_macierz_wszyskich_punktów!(x::AbstractVector{Float64})
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
function ncbj2_sąsiedzi!(macierz_punktow, dla_jakiego_punktu::Int, nn::Int)
    N = size(macierz_punktow, 2)
    x_i = macierz_punktow[:, dla_jakiego_punktu]
    distanse = zeros(Float32, N)

    for j in 1:N
        distanse[j] = norm(macierz_punktow[:, j] - x_i)
    end

    indeksy_𝛈 = sortperm(distanse)[2 : nn + 1]
    𝛈 = zeros(Float32, size(macierz_punktow, 1), nn)

    for k in 1:nn
        𝛈[:, k] = macierz_punktow[:, indeksy_𝛈[k]]
    end

    return 𝛈, indeksy_𝛈 , x_i
end




function ncbj3_calculate_wagi_dla_x_i!(𝛈, x_i::AbstractVector{Float64}, nn::Int)
    Z = 𝛈 .- x_i
    C = Z' * Z
    C += I * dx * tr(C)

    w_surowe = C \ ones(Float32, nn)

    return w_surowe ./ sum(w_surowe)
end



#  zmieniłem typ na AbstractMatrix by nie było probelmu z transponowaniem macierzy
function ncbj4_lle!(macierz_punktow::AbstractMatrix{<: AbstractFloat}, nn::Int)
    N = size(macierz_punktow, 2)
    W = zeros(Float32, N, N)

    for i in 1:N
        𝛈, indeksy_sasiadow, x_i = ncbj2_sąsiedzi!(macierz_punktow, i, nn)
        w = ncbj3_calculate_wagi_dla_x_i!(𝛈, x_i, nn)

        for k in 1:nn
            W[i, indeksy_sasiadow[k]] = w[k]
        end
    end

    return W
end
