using CairoMakie
using LinearAlgebra
using Statistics
using Random
using Printf
using LaTeXStrings
using NearestNeighbors
using AttractorsQGP

CairoMakie.activate!()

function marchenko(
        dane::AbstractMatrix{<:Real};
        tau::Real = 0.2,
        tau0::Real = 0.2,
        K::Integer = 160,
        T_scale::Real = 193.0,
        K_sweep = [6:4:250...],
        D::Int = 3,
        atol::Real = 1.0e-5
    )
    set_publication_theme()
    colors = AttractorsQGP.LPCA_COLORS

    idx_0 = findall(isapprox.(dane[:, 1], tau0; atol = atol))
    idx_t = findall(isapprox.(dane[:, 1], tau; atol = atol))

    X0 = copy(dane[idx_0, 2:(D + 1)])
    Xt = copy(dane[idx_t, 2:(D + 1)])
    X0[:, 1] ./= T_scale
    Xt[:, 1] ./= T_scale

    scales0 = [std(X0[:, j]) for j in 1:D]
    Zt = Xt ./ reshape(scales0, 1, D)
    N_pts = size(Zt, 1)

    # 3. Lokalne PCA dla wybranego K
    tree = KDTree(transpose(Zt))
    knn_idxs, _ = knn(tree, transpose(Zt), K, true)

    dpr_vals = zeros(Float64, N_pts)
    evals_sum = zeros(Float64, D)
    for i in 1:N_pts
        Y = Zt[knn_idxs[i], :]
        Y .-= mean(Y, dims = 1)
        C = (transpose(Y) * Y) ./ (K - 1)
        ev = sort(eigvals(Symmetric(C)), rev = true)
        ev = max.(ev, 1.0e-35)
        evals_sum .+= ev
        dpr_vals[i] = (sum(ev)^2) / sum(ev .^ 2)
    end

    mean_evals = evals_sum ./ N_pts
    mean_dpr = mean(dpr_vals)

    q = D / K
    sigma2_loc = sum(mean_evals) / D
    lambda_minus = q < 1.0 ? sigma2_loc * (1.0 - sqrt(q))^2 : 0.0
    lambda_plus = sigma2_loc * (1.0 + sqrt(q))^2

    k_list = sort(unique(filter(k -> 2 <= k < N_pts, collect(K_sweep))))
    knn_sweep, _ = knn(tree, transpose(Zt), maximum(k_list), true)

    sweep_mean = zeros(Float64, length(k_list))
    sweep_std = zeros(Float64, length(k_list))

    for (k_idx, k_val) in enumerate(k_list)
        buf = zeros(Float64, N_pts)
        for i in 1:N_pts
            Y = Zt[knn_sweep[i][1:k_val], :]
            Y .-= mean(Y, dims = 1)
            C = (transpose(Y) * Y) ./ (k_val - 1)
            ev = sort(eigvals(Symmetric(C)), rev = true)
            ev = max.(ev, 1.0e-35)
            buf[i] = (sum(ev)^2) / sum(ev .^ 2)
        end
        sweep_mean[k_idx] = mean(buf)
        sweep_std[k_idx] = std(buf)
    end

    # 6. Wykres 1x3
    fig = Figure(size = (1300, 400), figure_padding = (30, 25, 20, 20))

    # (a) Histogram z poprawnym N / N_max
    ax1 = Axis(
        fig[1, 1],
        title = L"\text{(a) } d_{\mathrm{PR}} \ (K=%$(K))",
        xlabel = L"d_{\mathrm{PR}}",
        ylabel = L"N / N_{\max}",
        limits = (0.95, 3.05, 0.0, 1.05)
    )
    edges = range(1.0, 3.0, length = 41)
    centers = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
    counts = [count(x -> edges[i] <= x < edges[i + 1], dpr_vals) for i in 1:length(centers)]
    norm_counts = counts ./ max(maximum(counts), 1)
    barplot!(ax1, centers, norm_counts, width = step(edges) * 0.95, color = (colors[2], 0.6), strokewidth = 0.5, strokecolor = :black)
    vlines!(ax1, [mean_dpr], color = colors[1], linestyle = :dash, linewidth = 2.5, label = L"\langle d_{\mathrm{PR}} \rangle = %$(round(mean_dpr, digits=2))")
    axislegend(ax1, position = :rt, labelsize = 12)

    # (b) Skan plateau
    ax2 = Axis(
        fig[1, 2],
        # xscale = log10,
        title = L"\text{(b) Skan plateau } \langle d_{\mathrm{PR}}(K) \rangle",
        xlabel = L"K \text{ (liczba sąsiadów)}",
        ylabel = L"\langle d_{\mathrm{PR}} \rangle",
        limits = (minimum(k_list), maximum(k_list), 0.95, D + 0.05)
    )
    band!(ax2, k_list, sweep_mean .- sweep_std, sweep_mean .+ sweep_std, color = (colors[4], 0.2))
    lines!(ax2, k_list, sweep_mean, color = colors[4], linewidth = 2.8, label = L"\langle d_{\mathrm{PR}} \rangle \pm 1\sigma")
    vlines!(ax2, [K], color = colors[1], linestyle = :dash, linewidth = 2.0, label = L"K = %$(K)")
    vlines!(ax2, [12 * D], color = :black, linestyle = :dot, linewidth = 1.8, label = L"K_{\mathrm{RMT}} = 12D")
    axislegend(ax2, position = :rb, labelsize = 12)

    # (c) Spektrum vs pasmo MP przeskalowane do wariancji lokalnej
    ax3 = Axis(
        fig[1, 3],
        yscale = log10,
        title = L"\text{(c) Spektrum } \langle \lambda_j \rangle \text{ vs pasmo MP}",
        xlabel = L"\text{Indeks modów } j",
        ylabel = L"\langle \lambda_j \rangle",
        xticks = (1:D, [latexstring("\\lambda_{$j}") for j in 1:D])
    )
    scatter!(ax3, 1:D, mean_evals, color = colors[1], markersize = 14, label = L"\text{Średnie } \langle \lambda_j \rangle")
    lines!(ax3, 1:D, mean_evals, color = colors[1], linewidth = 2.0)
    band!(ax3, [0.5, Float64(D) + 0.5], [max(lambda_minus, 1.0e-35), max(lambda_minus, 1.0e-35)], [lambda_plus, lambda_plus], color = (:gray70, 0.35), label = L"\text{Pasmo szumu MP}")
    hlines!(ax3, [lambda_plus], color = :crimson, linestyle = :dash, linewidth = 1.5)
    if lambda_minus > 0
        hlines!(ax3, [lambda_minus], color = :navy, linestyle = :dash, linewidth = 1.5)
    end
    axislegend(ax3, position = :rt, labelsize = 12)

    display(fig)
    return (mean_dpr = mean_dpr, mean_evals = mean_evals, mp_band = (lambda_minus, lambda_plus), fig = fig)
end
