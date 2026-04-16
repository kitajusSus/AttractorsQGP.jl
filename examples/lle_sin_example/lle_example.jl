using LinearAlgebra
using GLMakie
using Random
using AtractorsQGP


function compute_lle(X::Matrix{Float64}, k::Int, d::Int)
    D, N = size(X)
    dist2 = sum(X.^2, dims=1)' .+ sum(X.^2, dims=1) .- 2 .* (X' * X)
    W = zeros(N, N)
    for i in 1:N
        p_idx = sortperm(dist2[:, i])
        n_idx = p_idx[2:k+1]
        Z = X[:, n_idx] .- X[:, i]
        C = Z' * Z
        C_reg = C + 1e-3 * tr(C) * I
        w = C_reg \ ones(k)
        w ./= sum(w)

        W[i, n_idx] = w
    end
    I_N = Matrix{Float64}(I, N, N)
    M = (I_N - W)' * (I_N - W)
    eig_decomp = eigen(Symmetric(M))

    idx = sortperm(eig_decomp.values)
    Y = eig_decomp.vectors[:, idx[2:d+1]]'

    return Y
end




set_publication_theme()
Random.seed!(4)
N_points = 150
t = range(0, 2pi, length=N_points)
#
X_orig = vcat(t', sin.(t)' .+ 0.2 .* randn(1, N_points))
k_neighbors = 4
d_target = 2

Y_lle = compute_lle(X_orig, k_neighbors, d_target)

Y_lle_norm = (Y_lle .- minimum(Y_lle)) ./ (maximum(Y_lle) - minimum(Y_lle)) .* 2pi

X_target = vcat(Y_lle_norm, zeros(1, N_points))

fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1], title = "LLE: przyklad sinusa $k_neighbors sasiadow, docelowa wymiarowosc $d_target",
          xlabel = "x", ylabel = "sin(x)")

alpha = Observable(0.0)

points = @lift begin
    current_X = (1.0 - $alpha) .* X_orig .+ $alpha .* X_target
    [Point2f(current_X[1, i], current_X[2, i]) for i in 1:N_points]
end

scatter!(ax, points, color=t, colormap=:magma, markersize=12)
limits!(ax, -1, 7, -2, 2)

framerate = 30
frames = 0:0.01:1.0

record(fig, "lle_sinus_animation.mp4", frames; framerate = framerate) do frame
    alpha[] = frame
end
