function swissroll_dane(N)
    τ = 1.5π .+ 3π .* rand(Float32, N)
    h = 21f0 .* rand(Float32, N)
    r = τ .+ (0.2f0 .* randn(Float32, N))
    X = [r .* cos.(τ) h r .* sin.(τ)]'
    return X, τ
end

function ex_swissroll(; N=2000, K=12)
    X, labels = swissroll_dane(N)
    wagi = ncbj4_lle_basic(X, K)
    Y = ncbj5_nowy_manifold(wagi)
    return plot_examples_lle(X, Y, labels)
end
