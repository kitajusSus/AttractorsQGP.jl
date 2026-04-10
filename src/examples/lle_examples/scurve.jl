function scurve_dane(N)
    θ = LinRange(-1.5f0 * π, 1.5f0 * π, N)
    x = sin.(θ) .+ 0.1f0 .* randn(Float32, N)
    y = LinRange(0f0, 5f0, N) .+ 0.01f0 .* randn(Float32, N) 
    z = sign.(θ) .+ 0.1f0 .* randn(Float32, N)
    X = [x y z]'
    return X, θ
end

function ex_scurve(; N=2000, K=12)
    X, labels = scurve_dane(N)
    wagi = ncbj4_lle_basic(X, K)
    Y = ncbj5_nowy_manifold(wagi)
    return plot_examples_lle(X, Y, labels)
end
