using GLMakie 
using LinearAlgebra
const dx = 1e-3
include("../ncbj_lle.jl")
# skopiowane z examples/ncbj_lle.jl
include("plots_lle.jl")

# funkcja generująca dane dla swissrolla, zwraca macierz punktów i etykiety (tutaj τ)
function swissroll_dane(N)
    τ = 1.5π .+ 3π .* rand(Float32, N)
    h = 21f0 .* rand(Float32, N)
    r = τ .+ (0.2f0 .* randn(Float32, N))
    X = [r .* cos.(τ) h r .* sin.(τ)]'
    return X, τ
end

function ex_swissroll(; N=2000, K=12)

    X, labels = swissroll_dane(N)
    # println("Wygenerowane dane Swiss Roll (X) \n  ", X )
    wagi = ncbj4_lle_basic(X, K)
    # println("Macierz wag W:\n", wagi)
    Y = ncbj5_nowy_manifold(wagi)
    # println("Nowe Wektory Y (2D):\n", Y)
    fig = plot_examples_lle(X, Y, labels)
    display(fig)
    print("Press Enter to continue...")
    readline()
    return fig
end



ex_swissroll()
