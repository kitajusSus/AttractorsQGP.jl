using HDF5
using CairoMakie
using LinearAlgebra
using NearestNeighbors
using Statistics
using Printf
using LaTeXStrings

# Load theme directly to avoid full package precompilation overhead
include(joinpath(@__DIR__, "..", "..", "src", "analysis", "plots", "theme.jl"))

# Explicitly activate CairoMakie and apply the user's publication theme
CairoMakie.activate!()
set_publication_theme(fontsize = 22)

output_dir = joinpath(@__DIR__, "..", "plots", "k_eigenvalues")
mkpath(output_dir)

# 1. Load dataset
h5_path = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5")
println("Loading MIS dataset from: $h5_path")

data = h5open(h5_path, "r") do file
    file["dataset"][:, :]
end

k_list = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024]
n_eval = 2500

# ------------------------------------------------------------------------------
# A. High-statistics analysis at tau = 2.0 fm/c (Asymptotic Attractor)
# ------------------------------------------------------------------------------
tau_att = 2.0
slice_mask_att = abs.(data[:, 1] .- tau_att) .< 1e-4
X_raw_att = data[slice_mask_att, 2:3]
N_att, D = size(X_raw_att)

μ_att = mean(X_raw_att, dims=1)
σ_att = std(X_raw_att, dims=1)
X_norm_att = (X_raw_att .- μ_att) ./ σ_att

tree_att = KDTree(transpose(X_norm_att))
eval_idxs_att = round.(Int, range(1, N_att, length=n_eval))

max_k = maximum(k_list)
knn_all_att, _ = knn(tree_att, transpose(X_norm_att[eval_idxs_att, :]), max_k, true)

mean_ratio_12 = zeros(Float64, length(k_list))
std_ratio_12  = zeros(Float64, length(k_list))
mean_ratio_21 = zeros(Float64, length(k_list))
std_ratio_21  = zeros(Float64, length(k_list))

for (k_idx, k) in enumerate(k_list)
    r21_vals = zeros(Float64, n_eval)
    for i in 1:n_eval
        idxs = knn_all_att[i][1:k]
        Y = X_norm_att[idxs, :]
        Y_c = Y .- mean(Y, dims=1)
        C = (transpose(Y_c) * Y_c) ./ (k - 1)
        evals = sort(eigvals(Symmetric(C)), rev=true)
        l1 = evals[1]
        l2 = max(evals[2], 1e-35)
        r21_vals[i] = l2 / l1
    end
    mean_ratio_21[k_idx] = mean(r21_vals)
    std_ratio_21[k_idx]  = std(r21_vals)
    mean_ratio_12[k_idx] = 1.0 / mean_ratio_21[k_idx]
    std_ratio_12[k_idx]  = std_ratio_21[k_idx] / (mean_ratio_21[k_idx]^2)
end

peak_idx = argmax(mean_ratio_12)
k_opt_peak = k_list[peak_idx]
peak_val = mean_ratio_12[peak_idx]

@printf("Peak Attractor Resolution λ1/λ2 = %.2e at K = %d\n", peak_val, k_opt_peak)

# ------------------------------------------------------------------------------
# B. Multi-tau evolution: How the peak emerges across hydrodynamization
# ------------------------------------------------------------------------------
tau_slices = [0.25, 0.45, 0.65, 1.00, 2.00]
tau_curves = Dict{Float64, Vector{Float64}}()

for tau_val in tau_slices
    mask_tau = abs.(data[:, 1] .- tau_val) .< 1e-4
    X_tau = data[mask_tau, 2:3]
    μ_t = mean(X_tau, dims=1)
    σ_t = std(X_tau, dims=1)
    Xn_t = (X_tau .- μ_t) ./ σ_t
    
    tree_t = KDTree(transpose(Xn_t))
    knn_t, _ = knn(tree_t, transpose(Xn_t[eval_idxs_att, :]), max_k, true)
    
    r12_tau = zeros(Float64, length(k_list))
    for (k_idx, k) in enumerate(k_list)
        tmp = zeros(Float64, n_eval)
        for i in 1:n_eval
            sub_ids = knn_t[i][1:k]
            Y = Xn_t[sub_ids, :]
            Y_c = Y .- mean(Y, dims=1)
            C = (transpose(Y_c) * Y_c) ./ (k - 1)
            ev = sort(eigvals(Symmetric(C)), rev=true)
            tmp[i] = max(ev[2], 1e-35) / ev[1]
        end
        r12_tau[k_idx] = 1.0 / mean(tmp)
    end
    tau_curves[tau_val] = r12_tau
    println("Processed tau = $tau_val | max ratio = ", round(maximum(r12_tau), sigdigits=3))
end

# ------------------------------------------------------------------------------
# RYSUNEK 1: STANDALONE ELEGANCKI WYKRES λ1 / λ2 vs K (ATTRACTOR RESOLUTION PEAK)
# ------------------------------------------------------------------------------
fig_single = Figure(size = (850, 600))
ax1 = Axis(fig_single[1, 1],
    xscale = log10,
    yscale = log10,
    xlabel = L"\text{Liczba najbliższych sąsiadów } K",
    ylabel = L"\text{Stosunek wartości własnych } \lambda_1 / \lambda_2",
    xautolimitmargin = (0.04, 0.04),
    yautolimitmargin = (0.06, 0.08)
)

# Subtle shading of optimal plateau [64, 192]
vspan!(ax1, 64, 192, color = RGBAf(0.85, 0.72, 0.20, 0.18), label = L"\text{Optymalne plato } [64, 192]")
vlines!(ax1, [k_opt_peak], color = RGBAf(0.2, 0.2, 0.2, 0.8), linestyle = :dash, linewidth = 2.0, label = L"K_{\mathrm{opt}} = 128 \ (\text{Maksimum rozdzielczości})")

# Guide asymptotes:
# Left: Noise regime ~ K^(+1.82)
k_noise = 3:24
noise_guide = mean_ratio_12[1] .* (k_noise ./ 3).^(1.82)
lines!(ax1, k_noise, noise_guide, color = RGBAf(0.15, 0.45, 0.85, 0.75), linestyle = :dash, linewidth = 2.0, label = L"\text{Tłumienie szumu } \sim K^{+1.8}")

# Right: Curvature sagitta regime ~ K^(-1.80)
k_curv = 128:1024
curv_guide = peak_val .* (k_curv ./ k_opt_peak).^(-1.80)
lines!(ax1, k_curv, curv_guide, color = RGBAf(0.85, 0.20, 0.20, 0.75), linestyle = :dash, linewidth = 2.0, label = L"\text{Wzrost strzałki ugięcia } \sim K^{-1.8}")

# Empirical curve with subtle dispersion band
band!(ax1, k_list, max.(mean_ratio_12 .- std_ratio_12, 10.0), mean_ratio_12 .+ std_ratio_12, color = RGBAf(0.10, 0.25, 0.55, 0.15), label = L"\pm 1\sigma \text{ Dyspersja statystyczna}")
lines!(ax1, k_list, mean_ratio_12, color = RGBf(0.08, 0.22, 0.58), linewidth = 3.5, label = L"\text{Dane empiryczne } \langle \lambda_1 / \lambda_2 \rangle")
scatter!(ax1, k_list, mean_ratio_12, color = RGBf(0.08, 0.22, 0.58), markersize = 9)

# Subtle peak star
scatter!(ax1, [k_opt_peak], [peak_val], color = RGBf(0.80, 0.10, 0.10), markersize = 16, marker = :star5, label = L"\text{Maksimum: } \lambda_1/\lambda_2 \approx 1.1 \times 10^5")

# Text annotations pointing subtly to the regimes
text!(ax1, 3.5, 2e3, text = "← Reżim szumu (odpychanie MP)", fontsize = 13, color = RGBf(0.15, 0.40, 0.80))
text!(ax1, 200.0, 4e3, text = "Reżim krzywizny → (strzałka ugięcia)", fontsize = 13, color = RGBf(0.80, 0.15, 0.15))

# Elegant legend
axislegend(ax1, position = :rb, framevisible = true, framewidth = 1.0, framecolor = :black, backgroundcolor = RGBAf(1.0, 1.0, 1.0, 0.92), labelsize = 14)

pdf_single = joinpath(output_dir, "ratio_lambda1_lambda2_vs_K_subtle.pdf")
png_single = joinpath(output_dir, "ratio_lambda1_lambda2_vs_K_subtle.png")
save(pdf_single, fig_single)
save(png_single, fig_single, px_per_unit = 2)
println("Saved subtle single plot:")
println("  PDF: $pdf_single")
println("  PNG: $png_single")

# ------------------------------------------------------------------------------
# RYSUNEK 2: DUAL-PANEL: (a) λ1/λ2 vs K na atraktorze, (b) Ewolucja w czasie tau
# ------------------------------------------------------------------------------
fig_dual = Figure(size = (1250, 560))

# Left panel: The single attractor resolution peak
ax_d1 = Axis(fig_dual[1, 1],
    xscale = log10,
    yscale = log10,
    xlabel = L"\text{Liczba najbliższych sąsiadów } K",
    ylabel = L"\text{Stosunek } \lambda_1 / \lambda_2 \text{ na atraktorze}",
    title = "(a) Szczyt rozdzielczości atraktora (τ = 2.0 fm/c)",
    xautolimitmargin = (0.04, 0.04),
    yautolimitmargin = (0.06, 0.08)
)
vspan!(ax_d1, 64, 192, color = RGBAf(0.85, 0.72, 0.20, 0.18), label = L"\text{Plato } [64, 192]")
vlines!(ax_d1, [k_opt_peak], color = RGBAf(0.2, 0.2, 0.2, 0.8), linestyle = :dash, linewidth = 2.0, label = L"K_{\mathrm{opt}} = 128")

lines!(ax_d1, k_noise, noise_guide, color = RGBAf(0.15, 0.45, 0.85, 0.75), linestyle = :dash, linewidth = 1.8, label = L"\text{Szum } \sim K^{+1.8}")
lines!(ax_d1, k_curv, curv_guide, color = RGBAf(0.85, 0.20, 0.20, 0.75), linestyle = :dash, linewidth = 1.8, label = L"\text{Krzywizna } \sim K^{-1.8}")

band!(ax_d1, k_list, max.(mean_ratio_12 .- std_ratio_12, 10.0), mean_ratio_12 .+ std_ratio_12, color = RGBAf(0.10, 0.25, 0.55, 0.15))
lines!(ax_d1, k_list, mean_ratio_12, color = RGBf(0.08, 0.22, 0.58), linewidth = 3.2, label = L"\lambda_1 / \lambda_2 \ (\tau = 2.0\,\mathrm{fm}/c)")
scatter!(ax_d1, k_list, mean_ratio_12, color = RGBf(0.08, 0.22, 0.58), markersize = 8)
scatter!(ax_d1, [k_opt_peak], [peak_val], color = RGBf(0.80, 0.10, 0.10), markersize = 14, marker = :star5)

axislegend(ax_d1, position = :rb, framevisible = true, labelsize = 13)

# Right panel: Multi-tau emergence of the peak
ax_d2 = Axis(fig_dual[1, 2],
    xscale = log10,
    yscale = log10,
    xlabel = L"\text{Liczba najbliższych sąsiadów } K",
    ylabel = L"\text{Stosunek } \lambda_1 / \lambda_2",
    title = "(b) Wyłanianie się atraktora w toku ewolucji τ",
    xautolimitmargin = (0.04, 0.04),
    yautolimitmargin = (0.06, 0.08)
)
vspan!(ax_d2, 64, 192, color = RGBAf(0.85, 0.72, 0.20, 0.18))
vlines!(ax_d2, [k_opt_peak], color = RGBAf(0.2, 0.2, 0.2, 0.8), linestyle = :dash, linewidth = 2.0)

palette_tau = [
    RGBf(0.10, 0.55, 0.85), # 0.25
    RGBf(0.15, 0.65, 0.60), # 0.45
    RGBf(0.85, 0.60, 0.10), # 0.65
    RGBf(0.85, 0.35, 0.15), # 1.00
    RGBf(0.65, 0.10, 0.25), # 2.00
]

for (idx, tau_val) in enumerate(tau_slices)
    lines!(ax_d2, k_list, tau_curves[tau_val], color = palette_tau[idx], linewidth = 2.8, label = L"\tau = %$tau_val\,\mathrm{fm}/c")
    scatter!(ax_d2, k_list, tau_curves[tau_val], color = palette_tau[idx], markersize = 6)
end

hlines!(ax_d2, [1.0], color = :gray50, linestyle = :dot, linewidth = 1.2)
axislegend(ax_d2, position = :lt, framevisible = true, labelsize = 13)

pdf_dual = joinpath(output_dir, "ratio_lambda1_lambda2_dual_panel_subtle.pdf")
png_dual = joinpath(output_dir, "ratio_lambda1_lambda2_dual_panel_subtle.png")
save(pdf_dual, fig_dual)
save(png_dual, fig_dual, px_per_unit = 2)
println("Saved subtle dual panel plot:")
println("  PDF: $pdf_dual")
println("  PNG: $png_dual")

# ------------------------------------------------------------------------------
# RYSUNEK 3: TAKŻE WERSJA ODWROTNA: λ2 / λ1 (DOLINA U-CURVE)
# ------------------------------------------------------------------------------
fig_inv = Figure(size = (850, 600))
ax_inv = Axis(fig_inv[1, 1],
    xscale = log10,
    yscale = log10,
    xlabel = L"\text{Liczba najbliższych sąsiadów } K",
    ylabel = L"\text{Względna grubość poprzeczna } \lambda_2 / \lambda_1",
    xautolimitmargin = (0.04, 0.04),
    yautolimitmargin = (0.06, 0.08)
)
vspan!(ax_inv, 64, 192, color = RGBAf(0.85, 0.72, 0.20, 0.18), label = L"\text{Optymalne plato } [64, 192]")
vlines!(ax_inv, [k_opt_peak], color = RGBAf(0.2, 0.2, 0.2, 0.8), linestyle = :dash, linewidth = 2.0, label = L"K_{\mathrm{opt}} = 128 \ (\text{Minimum błędu})")

# Inverted asymptotes
inv_noise_guide = mean_ratio_21[1] .* (k_noise ./ 3).^(-1.82)
lines!(ax_inv, k_noise, inv_noise_guide, color = RGBAf(0.15, 0.45, 0.85, 0.75), linestyle = :dash, linewidth = 2.0, label = L"\text{Szum próbkowania } \sim K^{-1.8}")
inv_curv_guide = (1.0 / peak_val) .* (k_curv ./ k_opt_peak).^(1.80)
lines!(ax_inv, k_curv, inv_curv_guide, color = RGBAf(0.85, 0.20, 0.20, 0.75), linestyle = :dash, linewidth = 2.0, label = L"\text{Strzałka ugięcia (krzywizna) } \sim K^{+1.8}")

lines!(ax_inv, k_list, mean_ratio_21, color = RGBf(0.55, 0.15, 0.65), linewidth = 3.5, label = L"\text{Krzywa U-curve } \langle \lambda_2 / \lambda_1 \rangle")
scatter!(ax_inv, k_list, mean_ratio_21, color = RGBf(0.55, 0.15, 0.65), markersize = 9)
scatter!(ax_inv, [k_opt_peak], [1.0 / peak_val], color = RGBf(0.80, 0.10, 0.10), markersize = 16, marker = :star5, label = L"\text{Minimum: } \lambda_2/\lambda_1 \approx 9.1 \times 10^{-6}")

axislegend(ax_inv, position = :rt, framevisible = true, framewidth = 1.0, framecolor = :black, backgroundcolor = RGBAf(1.0, 1.0, 1.0, 0.92), labelsize = 14)

pdf_inv = joinpath(output_dir, "ratio_lambda2_lambda1_u_curve_subtle.pdf")
png_inv = joinpath(output_dir, "ratio_lambda2_lambda1_u_curve_subtle.png")
save(pdf_inv, fig_inv)
save(png_inv, fig_inv, px_per_unit = 2)
println("Saved subtle inverted U-curve plot:")
println("  PDF: $pdf_inv")
println("  PNG: $png_inv")
