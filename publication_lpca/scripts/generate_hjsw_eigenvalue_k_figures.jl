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

output_dir = joinpath(@__DIR__, "..", "plots", "k_eigenvalues_hjsw")
mkpath(output_dir)

h5_path = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5")
println("Loading HJSW dataset from: $h5_path")

data = h5open(h5_path, "r") do file
    file["dataset"][:, :]
end

total_rows, total_cols = size(data)
println("Dataset loaded: $total_rows rows, $total_cols columns")

# Time slices:
# tau = 0.20 (Stage 1: 3D volume)
# tau = 0.40 (Stage 2: 2D sheet collapse)
# tau = 2.00 (Stage 3: 1D attractor)
taus_to_test = [0.20, 0.40, 2.00]

k_list = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024]
n_eval = 2000

results_by_tau = Dict{Float64, Any}()

for tau_val in taus_to_test
    slice_mask = abs.(data[:, 1] .- tau_val) .< 1e-4
    X_raw = data[slice_mask, 2:4] # columns: T, A, B (D = 3)
    N, D = size(X_raw)
    println("\n=== Processing tau = $tau_val: N = $N points, D = $D dimensions ===")
    
    # Z-score standardization
    μ = mean(X_raw, dims=1)
    σ = std(X_raw, dims=1)
    X_norm = (X_raw .- μ) ./ σ
    
    tree = KDTree(X_norm')
    eval_idxs = round.(Int, range(1, N, length=n_eval))
    
    max_k = maximum(k_list)
    knn_idxs_all, _ = knn(tree, X_norm[eval_idxs, :]', max_k, true)
    
    m_l1 = zeros(Float64, length(k_list))
    m_l2 = zeros(Float64, length(k_list))
    m_l3 = zeros(Float64, length(k_list))
    m_dpr = zeros(Float64, length(k_list))
    s_dpr = zeros(Float64, length(k_list))
    
    for (k_idx, k) in enumerate(k_list)
        l1_v = zeros(Float64, n_eval)
        l2_v = zeros(Float64, n_eval)
        l3_v = zeros(Float64, n_eval)
        dpr_v = zeros(Float64, n_eval)
        
        for i in 1:n_eval
            idxs = knn_idxs_all[i][1:k]
            Y = X_norm[idxs, :]
            Y_c = Y .- mean(Y, dims=1)
            C = (Y_c' * Y_c) ./ (k - 1)
            evals = sort(eigvals(Symmetric(C)), rev=true)
            
            l1 = evals[1]
            l2 = max(evals[2], 1e-35)
            l3 = max(evals[3], 1e-35)
            
            l1_v[i] = l1
            l2_v[i] = l2
            l3_v[i] = l3
            dpr_v[i] = (l1 + l2 + l3)^2 / (l1^2 + l2^2 + l3^2)
        end
        
        m_l1[k_idx] = mean(l1_v)
        m_l2[k_idx] = mean(l2_v)
        m_l3[k_idx] = mean(l3_v)
        m_dpr[k_idx] = mean(dpr_v)
        s_dpr[k_idx] = std(dpr_v)
        
        @printf("k = %4d | λ1 = %.2e | λ2 = %.2e | λ3 = %.2e | d_PR = %.3f ± %.3f\n",
                k, m_l1[k_idx], m_l2[k_idx], m_l3[k_idx], m_dpr[k_idx], s_dpr[k_idx])
    end
    
    results_by_tau[tau_val] = (
        k_list = k_list,
        l1 = m_l1,
        l2 = m_l2,
        l3 = m_l3,
        dpr = m_dpr,
        s_dpr = s_dpr,
        N = N,
        D = D
    )
end

# ==============================================================================
# FIGURE 1: HJSW 3-STAGE EVOLUTION AND EIGENVALUE SPECTRA
# ==============================================================================
fig1 = Figure(size = (1150, 850))

# Panel (a): All 3 eigenvalues at tau = 2.0 (Attractor) showing 1D collapse
res_att = results_by_tau[2.00]
ax_a = Axis(fig1[1, 1],
    xscale = log10,
    yscale = log10,
    xlabel = L"\text{Neighborhood size } k",
    ylabel = L"\text{Covariance Eigenvalues } \lambda_i",
    title = L"\textbf{(a) Late Attractor } (\tau = 2.0\,\mathrm{fm}/c)\text{: } 1\text{D Curve Spectrum}"
)
vspan!(ax_a, 64, 256, color = (:gold, 0.22), label = L"\text{Optimal Plateau } [64, 256]")
lines!(ax_a, k_list, res_att.l1, color = :navy, linewidth = 3.0, label = L"\lambda_1 \propto k^2 \text{ (Tangent)}")
scatter!(ax_a, k_list, res_att.l1, color = :navy, markersize = 7)

lines!(ax_a, k_list, res_att.l2, color = :crimson, linewidth = 3.0, label = L"\lambda_2 \propto k^4 \text{ (Curvature Sagitta 1)}")
scatter!(ax_a, k_list, res_att.l2, color = :crimson, markersize = 7)

lines!(ax_a, k_list, res_att.l3, color = :darkmagenta, linewidth = 3.0, label = L"\lambda_3 \propto k^4 \text{ (Curvature Sagitta 2)}")
scatter!(ax_a, k_list, res_att.l3, color = :darkmagenta, markersize = 7)

vlines!(ax_a, [128], color = :gray30, linestyle = :dash, linewidth = 2.0, label = L"K_{\mathrm{opt}} = 128")
axislegend(ax_a, position = :lt, framevisible = true, labelsize = 11)

# Panel (b): Transverse Ratios λ2/λ1 and λ3/λ1 at tau = 2.0 (Double U-Curve)
ax_b = Axis(fig1[1, 2],
    xscale = log10,
    yscale = log10,
    xlabel = L"\text{Neighborhood size } k",
    ylabel = L"\text{Eigenvalue Ratios } \lambda_i / \lambda_1",
    title = L"\textbf{(b) Transverse U-Curves in 3D Ambient Space}"
)
vspan!(ax_b, 64, 256, color = (:gold, 0.22), label = L"\text{Optimal Valley } [64, 256]")
ratio_21 = res_att.l2 ./ res_att.l1
ratio_31 = res_att.l3 ./ res_att.l1

lines!(ax_b, k_list, ratio_21, color = :crimson, linewidth = 3.0, label = L"\lambda_2 / \lambda_1 \text{ (Slow mode sagitta)}")
scatter!(ax_b, k_list, ratio_21, color = :crimson, markersize = 7)

lines!(ax_b, k_list, ratio_31, color = :darkmagenta, linewidth = 3.0, label = L"\lambda_3 / \lambda_1 \text{ (Fast mode noise/sagitta)}")
scatter!(ax_b, k_list, ratio_31, color = :darkmagenta, markersize = 7)

vlines!(ax_b, [128], color = :gray30, linestyle = :dash, linewidth = 2.0, label = L"K_{\mathrm{opt}} = 128")
axislegend(ax_b, position = :rt, framevisible = true, labelsize = 11)

# Panel (c): Participation Ratio across the 3 hydrodynamic stages
ax_c = Axis(fig1[2, 1],
    xscale = log10,
    xlabel = L"\text{Neighborhood size } k",
    ylabel = L"\text{Participation Ratio } \langle d_{\mathrm{PR}} \rangle",
    title = L"\textbf{(c) Resolving } 3 \to 2 \to 1 \text{ Collapse Without Prior Knowledge}"
)
vspan!(ax_c, 64, 256, color = (:gold, 0.22), label = L"\text{Optimal Plateau}")

colors_tau = [:dodgerblue, :darkorange, :forestgreen]
labels_tau = [
    L"\tau = 0.20\,\mathrm{fm}/c \text{ (Stage 1: 3D volume, } d \approx 2.8\text{)}",
    L"\tau = 0.40\,\mathrm{fm}/c \text{ (Stage 2: 2D sheet, } d \approx 1.9\text{)}",
    L"\tau = 2.00\,\mathrm{fm}/c \text{ (Stage 3: 1D attractor, } d \equiv 1.00\text{)}"
]

for (idx, tau_val) in enumerate(taus_to_test)
    res = results_by_tau[tau_val]
    band!(ax_c, res.k_list, res.dpr .- res.s_dpr, res.dpr .+ res.s_dpr, color = (colors_tau[idx], 0.18))
    lines!(ax_c, res.k_list, res.dpr, color = colors_tau[idx], linewidth = 3.0, label = labels_tau[idx])
    scatter!(ax_c, res.k_list, res.dpr, color = colors_tau[idx], markersize = 6)
end

hlines!(ax_c, [1.0, 2.0, 3.0], color = :gray60, linestyle = :dot, linewidth = 1.5)
ylims!(ax_c, 0.95, 3.1)
axislegend(ax_c, position = :rc, framevisible = true, labelsize = 11)

# Panel (d): Scale Sensitivity S(k) for all three regimes
ax_d = Axis(fig1[2, 2],
    xscale = log10,
    yscale = log10,
    xlabel = L"\text{Neighborhood size } k",
    ylabel = L"\text{Scale Sensitivity } S(k) = |\partial \langle d \rangle / \partial \ln k|",
    title = L"\textbf{(d) Data-Driven Scale Invariance Indicator } S(k) \to 0"
)
vspan!(ax_d, 64, 256, color = (:gold, 0.22), label = L"\text{Optimal Window } [64, 256]")

for (idx, tau_val) in enumerate(taus_to_test)
    res = results_by_tau[tau_val]
    # compute S(k)
    s_k = zeros(Float64, length(res.k_list))
    for i in 2:(length(res.k_list)-1)
        dd = res.dpr[i+1] - res.dpr[i-1]
        dl = log(res.k_list[i+1]) - log(res.k_list[i-1])
        s_k[i] = abs(dd / dl)
    end
    s_k[1] = abs((res.dpr[2] - res.dpr[1]) / (log(res.k_list[2]) - log(res.k_list[1])))
    s_k[end] = abs((res.dpr[end] - res.dpr[end-1]) / (log(res.k_list[end]) - log(res.k_list[end-1])))
    
    lines!(ax_d, res.k_list, max.(s_k, 1e-5), color = colors_tau[idx], linewidth = 2.8, label = L"\tau = %$tau_val\,\mathrm{fm}/c")
    scatter!(ax_d, res.k_list, max.(s_k, 1e-5), color = colors_tau[idx], markersize = 6)
end

axislegend(ax_d, position = :rt, framevisible = true, labelsize = 11)

pdf_path = joinpath(output_dir, "hjsw_eigenvalues_k_sweep_multipanel.pdf")
png_path = joinpath(output_dir, "hjsw_eigenvalues_k_sweep_multipanel.png")
save(pdf_path, fig1)
save(png_path, fig1, px_per_unit = 2)
println("Successfully saved HJSW Figure to:")
println("  PDF: $pdf_path")
println("  PNG: $png_path")
