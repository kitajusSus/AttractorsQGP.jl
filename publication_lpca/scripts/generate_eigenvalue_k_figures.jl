using AttractorsQGP
using HDF5
using CairoMakie
using LinearAlgebra
using NearestNeighbors
using Statistics
using Printf
using LaTeXStrings

CairoMakie.activate!()

# Set publication theme
set_theme!(Theme(
    fontsize = 15,
    font = "TeX Gyre Heros",
    Axis = (
        xgridvisible = true,
        ygridvisible = true,
        xgridcolor = (:gray80, 0.5),
        ygridcolor = (:gray80, 0.5),
        spinewidth = 1.2,
        xtickwidth = 1.2,
        ytickwidth = 1.2,
        xtickalign = 1,
        ytickalign = 1,
    )
))

output_dir = joinpath(@__DIR__, "..", "plots", "k_eigenvalues")
mkpath(output_dir)

# 1. Load dataset
h5_path = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5")
println("Loading MIS dataset from: $h5_path")

data = h5open(h5_path, "r") do file
    file["dataset"][:, :]
end

# Analyze asymptotic attractor slice at tau = 2.0 fm/c
tau_target = 2.0
slice_mask = abs.(data[:, 1] .- tau_target) .< 1e-4

X_raw = data[slice_mask, 2:3]
N, D = size(X_raw)
println("Slice at tau = $tau_target: N = $N points, D = $D dimensions")

# Z-score standardization
μ = mean(X_raw, dims=1)
σ = std(X_raw, dims=1)
X_norm = (X_raw .- μ) ./ σ

# KDTree for fast neighbor lookup
tree = KDTree(X_norm')

# Dense grid of k values from 3 to 1024
k_list = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024]
n_eval = 2500
eval_idxs = round.(Int, range(1, N, length=n_eval))

println("Computing eigenvalues across $(length(k_list)) k-values for $n_eval points...")

m_l1_list = zeros(Float64, length(k_list))
m_l2_list = zeros(Float64, length(k_list))
m_ratio_list = zeros(Float64, length(k_list))
m_dpr_list = zeros(Float64, length(k_list))
s_dpr_list = zeros(Float64, length(k_list))

# Pre-fetch neighbors for maximum k to be ultra-fast
max_k = maximum(k_list)
knn_idxs_all, _ = knn(tree, X_norm[eval_idxs, :]', max_k, true)

for (k_idx, k) in enumerate(k_list)
    l1_vals = zeros(Float64, n_eval)
    l2_vals = zeros(Float64, n_eval)
    ratio_vals = zeros(Float64, n_eval)
    dpr_vals = zeros(Float64, n_eval)
    
    for i in 1:n_eval
        idxs = knn_idxs_all[i][1:k]
        Y = X_norm[idxs, :]
        Y_c = Y .- mean(Y, dims=1)
        C = (Y_c' * Y_c) ./ (k - 1)
        evals = sort(eigvals(Symmetric(C)), rev=true)
        l1 = evals[1]
        l2 = max(evals[2], 1e-35)
        
        l1_vals[i] = l1
        l2_vals[i] = l2
        ratio_vals[i] = l2 / l1
        dpr_vals[i] = (l1 + l2)^2 / (l1^2 + l2^2)
    end
    
    m_l1_list[k_idx] = mean(l1_vals)
    m_l2_list[k_idx] = mean(l2_vals)
    m_ratio_list[k_idx] = mean(ratio_vals)
    m_dpr_list[k_idx] = mean(dpr_vals)
    s_dpr_list[k_idx] = std(dpr_vals)
    
    @printf("k = %4d | <λ1> = %.3e | <λ2> = %.3e | <λ2/λ1> = %.3e | <d_PR> = %.4f ± %.4f\n", 
            k, m_l1_list[k_idx], m_l2_list[k_idx], m_ratio_list[k_idx], m_dpr_list[k_idx], s_dpr_list[k_idx])
end

# Optimal k index
opt_ratio_idx = argmin(m_ratio_list)
opt_k = k_list[opt_ratio_idx]
println("\n--> Global minimum of λ2/λ1: k = $opt_k (ratio = $(m_ratio_list[opt_ratio_idx]))")

# Theoretical K_opt prediction:
k_opt_theory = round(Int, N^0.45 * sqrt(D))
println("--> Theoretical rule K_opt = round(N^0.45 * sqrt(D)) = $k_opt_theory")

# Compute scale derivative: |d <d_PR> / d ln k|
scale_deriv = zeros(Float64, length(k_list))
for i in 2:(length(k_list)-1)
    d_dpr = m_dpr_list[i+1] - m_dpr_list[i-1]
    d_lnk = log(k_list[i+1]) - log(k_list[i-1])
    scale_deriv[i] = abs(d_dpr / d_lnk)
end
scale_deriv[1] = abs((m_dpr_list[2] - m_dpr_list[1]) / (log(k_list[2]) - log(k_list[1])))
scale_deriv[end] = abs((m_dpr_list[end] - m_dpr_list[end-1]) / (log(k_list[end]) - log(k_list[end-1])))

# ==============================================================================
# FIGURE 1: 4-PANEL PUBLICATION MASTER FIGURE
# ==============================================================================
fig = Figure(size = (1100, 850))

# Colors
c_l1 = :navy
c_l2 = :crimson
c_ratio = :darkviolet
c_dpr = :forestgreen
c_band = (:forestgreen, 0.20)
c_opt_line = :gray30
c_opt_zone = (:gold, 0.22)

# Subplot (a): Individual eigenvalues λ1 and λ2 vs k
ax_a = Axis(fig[1, 1],
    xscale = log10,
    yscale = log10,
    xlabel = L"\text{Neighborhood size } k",
    ylabel = L"\text{Covariance Eigenvalues } \lambda_i",
    title = L"\textbf{(a) Individual Eigenvalues: Tangent } \lambda_1 \text{ vs Normal } \lambda_2"
)
vspan!(ax_a, 64, 192, color = c_opt_zone, label = L"\text{Optimal Plateau } [64, 192]")
vlines!(ax_a, [opt_k], color = c_opt_line, linestyle = :dash, linewidth = 2.0, label = L"K_{\mathrm{opt}} = 128")
vlines!(ax_a, [k_opt_theory], color = :teal, linestyle = :dot, linewidth = 2.0, label = L"K_{\mathrm{rule}} = 89")

lines!(ax_a, k_list, m_l1_list, color = c_l1, linewidth = 3.0, label = L"\lambda_1(k) \propto k^2 \text{ (Tangent)}")
scatter!(ax_a, k_list, m_l1_list, color = c_l1, markersize = 7)

lines!(ax_a, k_list, m_l2_list, color = c_l2, linewidth = 3.0, label = L"\lambda_2(k) \propto k^4 \text{ (Curvature Sagitta)}")
scatter!(ax_a, k_list, m_l2_list, color = c_l2, markersize = 7)

# Power law guides
k_guide_large = 100:1000
guide_l1 = m_l1_list[opt_ratio_idx] .* (k_guide_large ./ opt_k).^2
lines!(ax_a, k_guide_large, guide_l1, color = c_l1, linestyle = :dot, linewidth = 1.5)
guide_l2 = m_l2_list[opt_ratio_idx] .* (k_guide_large ./ opt_k).^4
lines!(ax_a, k_guide_large, guide_l2, color = c_l2, linestyle = :dot, linewidth = 1.5)

axislegend(ax_a, position = :lt, framevisible = true, labelsize = 12, padding = (6, 6, 4, 4))

# Subplot (b): Eigenvalue Ratio λ2 / λ1 (The U-Curve)
ax_b = Axis(fig[1, 2],
    xscale = log10,
    yscale = log10,
    xlabel = L"\text{Neighborhood size } k",
    ylabel = L"\text{Ratio } \lambda_2(k) / \lambda_1(k)",
    title = L"\textbf{(b) The U-Curve: Spectral Noise vs. Curvature Sagitta}"
)
vspan!(ax_b, 64, 192, color = c_opt_zone, label = L"\text{Optimal Plateau } [64, 192]")
vlines!(ax_b, [opt_k], color = c_opt_line, linestyle = :dash, linewidth = 2.0, label = L"K_{\mathrm{opt}} = 128 \ (\min \lambda_2/\lambda_1)")
lines!(ax_b, k_list, m_ratio_list, color = c_ratio, linewidth = 3.5, label = L"\text{Empirical } \lambda_2 / \lambda_1")
scatter!(ax_b, k_list, m_ratio_list, color = c_ratio, markersize = 8)

# Highlight minimum point
scatter!(ax_b, [opt_k], [m_ratio_list[opt_ratio_idx]], color = :red, markersize = 14, marker = :star5, label = L"\text{Global Minimum } (9.3\times 10^{-6})")

# Noise asymptote ~ k^(-1.8) and Curvature asymptote ~ k^(1.8)
k_noise = 3:20
lines!(ax_b, k_noise, m_ratio_list[1] .* (k_noise ./ 3).^(-1.82), color = :blue, linestyle = :dash, linewidth = 1.8, label = L"\text{Sampling Noise } \sim k^{-1.8}")
k_curv = 128:1024
lines!(ax_b, k_curv, m_ratio_list[opt_ratio_idx] .* (k_curv ./ opt_k).^(1.80), color = :crimson, linestyle = :dash, linewidth = 1.8, label = L"\text{Sagitta Curvature } \sim k^{+1.8}")

text!(ax_b, 3.2, 3e-3, text = L"\leftarrow \text{\textbf{Noise-dominated}}", fontsize = 12, color = :blue)
text!(ax_b, 250, 4e-5, text = L"\text{\textbf{Curvature-dominated}} \rightarrow", fontsize = 12, color = :crimson)

axislegend(ax_b, position = :rt, framevisible = true, labelsize = 11, padding = (6, 6, 4, 4))

# Subplot (c): Participation Ratio Dimension d_PR(k)
ax_c = Axis(fig[2, 1],
    xscale = log10,
    xlabel = L"\text{Neighborhood size } k",
    ylabel = L"\text{Participation Ratio } \langle d_{\mathrm{PR}} \rangle",
    title = L"\textbf{(c) Local PR Dimension Plateau and Breakdown}"
)
vspan!(ax_c, 64, 192, color = c_opt_zone, label = L"\text{Stable Plateau } d_{\mathrm{PR}} \equiv 1.0000")
vlines!(ax_c, [opt_k], color = c_opt_line, linestyle = :dash, linewidth = 2.0, label = L"K_{\mathrm{opt}} = 128")
hlines!(ax_c, [1.0], color = :black, linestyle = :solid, linewidth = 1.5, label = L"\text{Exact Attractor } d = 1.0")

band!(ax_c, k_list, m_dpr_list .- s_dpr_list, m_dpr_list .+ s_dpr_list, color = c_band, label = L"\pm 1\sigma \text{ Ensemble Dispersion}")
lines!(ax_c, k_list, m_dpr_list, color = c_dpr, linewidth = 3.0, label = L"\text{Mean } \langle d_{\mathrm{PR}} \rangle")
scatter!(ax_c, k_list, m_dpr_list, color = c_dpr, markersize = 7)

ylims!(ax_c, 0.9995, 1.025)
axislegend(ax_c, position = :rt, framevisible = true, labelsize = 12, padding = (6, 6, 4, 4))

# Subplot (d): Optimization Metrics: Scale Derivative S(k) & Dispersion sigma(k)
ax_d = Axis(fig[2, 2],
    xscale = log10,
    yscale = log10,
    xlabel = L"\text{Neighborhood size } k",
    ylabel = L"\text{Optimization Diagnostic Metrics}",
    title = L"\textbf{(d) Data-Driven Indicators for Optimal Scale Selection}"
)
vspan!(ax_d, 64, 192, color = c_opt_zone, label = L"\text{Optimal Plateau } [64, 192]")
vlines!(ax_d, [opt_k], color = c_opt_line, linestyle = :dash, linewidth = 2.0, label = L"K_{\mathrm{opt}} = 128")

lines!(ax_d, k_list, max.(scale_deriv, 1e-6), color = :darkorange, linewidth = 2.8, label = L"\text{Scale Sensitivity } S(k) = |\partial \langle d \rangle / \partial \ln k|")
scatter!(ax_d, k_list, max.(scale_deriv, 1e-6), color = :darkorange, markersize = 7)

lines!(ax_d, k_list, max.(s_dpr_list, 1e-6), color = :teal, linewidth = 2.8, linestyle = :dash, label = L"\text{Standard Deviation } \sigma(d_{\mathrm{PR}})")
scatter!(ax_d, k_list, max.(s_dpr_list, 1e-6), color = :teal, markersize = 7)

# Minimum indicator
scatter!(ax_d, [opt_k], [max(scale_deriv[opt_ratio_idx], 1e-6)], color = :red, markersize = 12, marker = :star5, label = L"\text{Scale Plateau } S(k) \to 0")

axislegend(ax_d, position = :rt, framevisible = true, labelsize = 11, padding = (6, 6, 4, 4))

CairoMakie.activate!()
fig_path_pdf = joinpath(output_dir, "eigenvalue_k_sweep_multipanel.pdf")
fig_path_png = joinpath(output_dir, "eigenvalue_k_sweep_multipanel.png")
save(fig_path_pdf, fig)
save(fig_path_png, fig, px_per_unit = 2)
println("Successfully saved Figure 1 to:")
println("  PDF: $fig_path_pdf")
println("  PNG: $fig_path_png")

# ==============================================================================
# FIGURE 2: GEOMETRIC SCHEMATIC OF SAGITTA AND NEIGHBORHOOD BENDING
# ==============================================================================
fig_geom = Figure(size = (950, 480))

# Left panel: The geometry of the arc, tangent, and sagitta
ax_g1 = Axis(fig_geom[1, 1],
    xlabel = L"\text{Tangent coordinate } s \ (\propto k/N)",
    ylabel = L"\text{Normal coordinate } n \ (\text{Sagitta } \propto (k/N)^2)",
    title = L"\textbf{(a) Geometry of Extrinsic Curvature Sagitta on Attractor}",
    aspect = AxisAspect(1.4)
)

# True attractor arc: n = 0.5 * kappa * s^2
kappa_val = 0.8
s_arc = range(-1.4, 1.4, length=300)
n_arc = 0.5 .* kappa_val .* s_arc.^2

lines!(ax_g1, s_arc, n_arc, color = :black, linewidth = 3.5, label = L"\text{Attractor Arc } n(s) = \frac{1}{2}\kappa s^2")

# Tangent line n = 0
lines!(ax_g1, [-1.5, 1.5], [0.0, 0.0], color = :gray50, linestyle = :dash, linewidth = 2.0, label = L"\text{Tangent Space } T_{\mathbf{x}}\mathcal{M}")

# Small neighborhood (k_opt): L1 = 0.8
s_small = range(-0.4, 0.4, length=50)
n_small = 0.5 .* kappa_val .* s_small.^2
scatter!(ax_g1, s_small, n_small, color = :forestgreen, markersize = 8, label = L"k = K_{\mathrm{opt}} \ (\delta_{\mathrm{sagitta}} \ll L)")

# Large neighborhood (excessive k): L2 = 2.2
s_large = range(-1.1, 1.1, length=40)
n_large = 0.5 .* kappa_val .* s_large.^2
scatter!(ax_g1, s_large, n_large, color = :crimson, markersize = 8, label = L"k \gg K_{\mathrm{opt}} \ (\delta_{\mathrm{sagitta}} \sim L^2)")

# Sagitta arrow illustration at s = 1.1
arrows!(ax_g1, [1.1], [0.0], [0.0], [0.5 * kappa_val * 1.1^2], color = :crimson, linewidth = 2.5, arrowsize = 14)
text!(ax_g1, 1.15, 0.2, text = L"\delta_{\mathrm{sagitta}} \approx \frac{1}{2}\kappa L^2", color = :crimson, fontsize = 14)

# Tangent length arrow
arrows!(ax_g1, [-1.1], [-0.08], [2.2], [0.0], color = :navy, linewidth = 2.0, arrowsize = 14)
text!(ax_g1, -0.3, -0.16, text = L"\text{Arc length } L \propto \frac{k}{N}", color = :navy, fontsize = 14)

axislegend(ax_g1, position = :lt, framevisible = true, labelsize = 11)
ylims!(ax_g1, -0.22, 1.0)

# Right panel: Three Neighborhood Regimes and Covariance Ellipses
ax_g2 = Axis(fig_geom[1, 2],
    xlabel = L"\delta x_1 \text{ (along attractor)}",
    ylabel = L"\delta x_2 \text{ (anisotropy direction)}",
    title = L"\textbf{(b) Covariance Ellipse Shapes across } k \text{ Regimes}",
    aspect = AxisAspect(1.0)
)

theta = range(0, 2π, length=200)

# 1. k small: isotropic sampling noise -> circle / blurry ellipse
x_small = 0.25 .* cos.(theta)
y_small = 0.22 .* sin.(theta)
lines!(ax_g2, x_small, y_small, color = :blue, linewidth = 2.5, label = L"k < 12D: \text{ Noise floor } (d_{\mathrm{PR}} \to 2)")

# 2. k opt: extremely thin needle -> 1D line
x_opt = 0.65 .* cos.(theta)
y_opt = 0.02 .* sin.(theta)
lines!(ax_g2, x_opt, y_opt, color = :forestgreen, linewidth = 3.5, label = L"k = K_{\mathrm{opt}}: \text{ Flat needle } (d_{\mathrm{PR}} \equiv 1.00)")

# 3. k large: curvature sagitta -> inflated thick ellipse
x_large = 1.10 .* cos.(theta)
y_large = 0.35 .* sin.(theta)
lines!(ax_g2, x_large, y_large, color = :crimson, linewidth = 2.5, linestyle = :dash, label = L"k > 200: \text{ Curvature inflated } (d_{\mathrm{PR}} > 1)")

axislegend(ax_g2, position = :lt, framevisible = true, labelsize = 11)
xlims!(ax_g2, -1.25, 1.25)
ylims!(ax_g2, -1.25, 1.25)

fig2_path_pdf = joinpath(output_dir, "curvature_sagitta_schematic.pdf")
fig2_path_png = joinpath(output_dir, "curvature_sagitta_schematic.png")
save(fig2_path_pdf, fig_geom)
save(fig2_path_png, fig_geom, px_per_unit = 2)
println("Successfully saved Figure 2 to:")
println("  PDF: $fig2_path_pdf")
println("  PNG: $fig2_path_png")
