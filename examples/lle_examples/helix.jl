using GLMakie 
using LinearAlgebra
const dx = 1e-3
include("../ncbj_lle.jl")
# skopiowane z examples/ncbj_lle.jl
include("plots_lle.jl")

function helix_dane(N)
    t = LinRange(0f0, 4f0 * π, N)
    x = sin.(t) .+ 0.1f0 .* randn(Float32, N)
    y = cos.(t) .+ 0.1f0 .* randn(Float32, N)
    z = t .+ 0.1f0 .* randn(Float32, N)
    X = [x y z]'
    return X, t
end

function ex_helix(; N=2000, K=20)
    X, labels = helix_dane(N)
    wagi = ncbj4_lle_basic(X, K)
    Y = ncbj5_nowy_manifold(wagi)
    fig = plot_examples_lle(X, Y, labels)
    display(fig)
    # println("Naciśnij Enter, aby zakończyć...")
    # readline()
    return fig
end



@time ex_helix()
