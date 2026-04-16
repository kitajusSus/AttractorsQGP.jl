
import Pkg
Pkg.activate(joinpath(@__DIR__, "../../..")) 

using AtractorsQGP



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
    return plot_examples_lle(X, Y, labels)
end



ex_helix()
