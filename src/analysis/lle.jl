using LinearAlgebra

"""
Locally Linear Embedding (LLE)
X: matrix of input data with dimensions (D, N), `D` dimension of features,  `N` number of samples
k: numer of nearest neighbors to consider for each point
d: target dimension of the embedding space
"""
function lle(X::AbstractMatrix{T}; k::Int=10, d::Int=2) where T <: AbstractFloat
    D_dim, N = size(X)
    W = zeros(T, N, N)

    for i in 1:N
        # KROK 1: Wyznaczanie k najbliższych sąsiadów
        # euclidian space distances between point i and all other `nearest` points
        distances = vec(sum((X .- X[:, i]).^2, dims=1))
        # sortperm returns vector of indexes  ; we take from  2  to k+1 (because the first one is the point itself))
        neighbors = sortperm(distances)[2:k+1]

        # KROK 2: linear reconstruction
        # Z to różnica między sąsiadami a punktem analizowanym
        Z = X[:, neighbors] .- X[:, i]

        # Lokalna macierz kowariancji C
        C = Z' * Z

        # Regularyzacja macierzy kowariancji
        C += I(k) * 1e-3 * tr(C)

        # Rozwiązywanie układu równań liniowych w celu znalezienia wag
        w = C \ ones(T, k)

        # Przeskalowanie wag, aby sumowały się do 1
        w ./= sum(w)

        # Zapis wag do globalnej macierzy W
        W[i, neighbors] = w
    end

    # KROK 3: Mapowanie do nowej przestrzeni
    # Konstrukcja macierzy M
    M = (I(N) - W)' * (I(N) - W)

    # Rozwiązywanie problemu wartości własnych dla symetrycznej macierzy M
    eigvals, eigvecs = eigen(Symmetric(M))

    # Zwracamy wektory własne odpowiadające d najmniejszym niezerowym wartościom własnym
    return eigvecs[:, 2:d+1]
end
