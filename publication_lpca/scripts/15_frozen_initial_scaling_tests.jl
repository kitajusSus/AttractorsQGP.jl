"""
    Script 15: Evaluation of Frozen Initial-Scale Normalization and k-Plateau Selection
    
    Styled using lpca_style and publication typography.

    Implements and evaluates:
    1. Variant 1 (Frozen Scale):
       T_tilde(t) = T(t) / x_T(t_0), A_tilde(t) = A(t) / x_A(t_0)
       where x_T(t_0) = std(T(t_0)), x_A(t_0) = std(A(t_0)).
    2. Variant 2 (Cooling-Compensated Scale):
       T_tilde(t) = T(t) / x_T(t), A_tilde(t) = A(t) / x_A(t_0)
       where x_T(t) = x_T(t_0) * (<T(t)> / <T(t_0)>).
    3. Variant 3 (Logarithmic Temperature):
       y_tilde(t) = ln(T(t)/<T(t_0)>) / x_lnT(t_0), A_tilde(t) = A(t) / x_A(t_0)
       where x_lnT(t_0) = std(ln(T(t_0))).
    4. Reference (Per-Slice Z-Score):
       Standardization recomputed independently at each time slice.

    Generates four publication-grade PDF figures in plots/frozen_scaling/:
    - fig1_phase_space_evolution.pdf
    - fig2_eigenvalues_and_ratio_vs_tau.pdf
    - fig3_k_plateau_analysis.pdf
    - fig4_dimension_trajectories.pdf
"""

using HDF5
using CairoMakie
CairoMakie.activate!()

using AttractorsQGP
CairoMakie.activate!()

using LinearAlgebra
using NearestNeighbors
using Statistics
using Printf
using LaTeXStrings

# Set publication theme based on lpca_style
set_publication_theme(; cmap = lpca_style, fontsize = 14)

# Paths setup
dataset_path = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5")
output_dir = joinpath(@__DIR__, "..", "plots", "frozen_scaling")
results_dir = joinpath(@__DIR__, "..", "results")
mkpath(output_dir)
mkpath(results_dir)

println("Loading Conformal MIS dataset from: ", dataset_path)
data = h5open(dataset_path, "r") do f
    f["dataset"][:, :]
end

N_total, n_cols = size(data)
all_taus = sort(unique(data[:, 1]))
tau0 = all_taus[1]
println("Dataset loaded: $N_total rows across $(length(all_taus)) tau slices (tau_0 = $tau0, tau_max = $(all_taus[end])).")

# -----------------------------------------------------------------------------
# 1. Compute Frozen Reference Scales at tau_0
# -----------------------------------------------------------------------------
mask0 = abs.(data[:, 1] .- tau0) .< 1e-4
X0_full = data[mask0, :]
T0 = X0_full[:, 2]
A0 = X0_full[:, 3]
N_slice = length(T0)

mean_T0 = mean(T0)
x_T0 = std(T0)
x_A0 = std(A0)
x_lnT0 = std(log.(T0))

println("\n--- Reference Initial Scales at tau_0 = $tau0 fm/c (N = $N_slice) ---")
@printf("  <T_0>     = %.4f fm^-1\n", mean_T0)
@printf("  x_T(t_0)  = %.4f fm^-1 (std T_0)\n", x_T0)
@printf("  x_A(t_0)  = %.4f (std A_0)\n", x_A0)
@printf("  x_lnT(t0) = %.4f (std ln T_0)\n", x_lnT0)

# Colors and labels for the methods derived strictly from lpca_style palette
const LPCA_HEX = AttractorsQGP.LPCA_COLORS

methods_info = [
    (key = :var1, name = "Variant 1 (Frozen Scale)", color = LPCA_HEX[1], linestyle = :solid),
    (key = :var2, name = "Variant 2 (Cooling-Compensated)", color = LPCA_HEX[2], linestyle = :dash),
    (key = :var3, name = "Variant 3 (Log-Temperature)", color = LPCA_HEX[3], linestyle = :dot),
    (key = :zscore, name = "Reference (Per-Slice Z-score)", color = LPCA_HEX[5], linestyle = :dashdot)
]

# Helper to transform a raw slice (T, A) into scaled matrix Z according to method
function transform_slice(T::AbstractVector{<:Real}, A::AbstractVector{<:Real}, method::Symbol)
    if method === :var1
        return hcat(T ./ x_T0, A ./ x_A0)
    elseif method === :var2
        x_Tt = x_T0 * (mean(T) / mean_T0)
        return hcat(T ./ x_Tt, A ./ x_A0)
    elseif method === :var3
        return hcat(log.(T ./ mean_T0) ./ x_lnT0, A ./ x_A0)
    elseif method === :zscore
        sT = std(T)
        sA = std(A)
        return hcat((T .- mean(T)) ./ (sT > 0 ? sT : 1.0), (A .- mean(A)) ./ (sA > 0 ? sA : 1.0))
    else
        error("Unknown method: $method")
    end
end

# -----------------------------------------------------------------------------
# Figure 1: Phase Space Evolution (4 Methods x 4 Proper Times) using lpca_style
# -----------------------------------------------------------------------------
println("\nGenerating Figure 1: Phase Space Scatter Evolution (lpca_style)...")
selected_taus = [0.20, 0.45, 0.85, 2.00]
n_times = length(selected_taus)
n_meth = length(methods_info)

fig1 = Figure(size = (1200, 1050), figure_padding = (25, 25, 20, 20))

# Subsampling for scatter to keep PDF light and crisp
n_scatter = min(2500, N_slice)
scatter_sub = round.(Int, range(1, N_slice, length = n_scatter))

for (m_idx, minfo) in enumerate(methods_info)
    for (t_idx, tau_val) in enumerate(selected_taus)
        m_slice = abs.(data[:, 1] .- tau_val) .< 1e-4
        X_slice = data[m_slice, :]
        T_curr = X_slice[:, 2]
        A_curr = X_slice[:, 3]
        
        Z = transform_slice(T_curr, A_curr, minfo.key)
        
        ax = Axis(
            fig1[m_idx, t_idx],
            xlabel = m_idx == n_meth ? L"\tilde{x}_1" : "",
            ylabel = t_idx == 1 ? (minfo.key == :var1 ? L"\mathbf{Var\ 1:}\ \tilde{\mathcal{A}}" :
                                   minfo.key == :var2 ? L"\mathbf{Var\ 2:}\ \tilde{\mathcal{A}}" :
                                   minfo.key == :var3 ? L"\mathbf{Var\ 3:}\ \tilde{\mathcal{A}}" :
                                   L"\mathbf{Z\text{-}score:}\ \tilde{\mathcal{A}}") : "",
            title = m_idx == 1 ? latexstring("\\tau = $(@sprintf("%.2f", tau_val))\\ \\mathrm{fm}/c") : "",
            titlesize = 15
        )
        
        # Color by initial anisotropy A0 using lpca_style colormap
        scatter!(
            ax,
            Z[scatter_sub, 1], Z[scatter_sub, 2],
            color = A0[scatter_sub],
            colormap = lpca_style,
            colorrange = (minimum(A0), maximum(A0)),
            markersize = 4.0,
            alpha = 0.75
        )
        
        # Standardized limits for direct comparison
        if minfo.key === :var1
            xlims!(ax, 0.0, 4.5)
            ylims!(ax, -0.2, 3.8)
        elseif minfo.key === :var2
            xlims!(ax, 0.0, 4.5)
            ylims!(ax, -0.2, 3.8)
        elseif minfo.key === :var3
            xlims!(ax, -3.5, 2.5)
            ylims!(ax, -0.2, 3.8)
        elseif minfo.key === :zscore
            xlims!(ax, -2.5, 2.5)
            ylims!(ax, -2.5, 2.5)
        end
    end
end

Colorbar(
    fig1[1:n_meth, n_times + 1],
    colormap = lpca_style,
    limits = (minimum(A0), maximum(A0)),
    label = L"\text{Initial Anisotropy }\mathcal{A}_0",
    labelsize = 15,
    width = 18
)

save(joinpath(output_dir, "fig1_phase_space_evolution.pdf"), fig1)
println("Saved: ", joinpath(output_dir, "fig1_phase_space_evolution.pdf"))

# -----------------------------------------------------------------------------
# Figure 2: Local Eigenvalues lambda_1, lambda_2 and Anisotropy Ratio vs Tau
# -----------------------------------------------------------------------------
println("\nGenerating Figure 2: Eigenvalues and Ratio vs Tau (lpca_style)...")

tau_grid_eval = [
    0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
    0.70, 0.80, 0.85, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20,
    2.50, 2.70, 3.00
]
n_grid = length(tau_grid_eval)
k_eval_fixed = 16

eigen_results = Dict{Symbol, NamedTuple}()

for minfo in methods_info
    m_key = minfo.key
    l1_means = zeros(Float64, n_grid)
    l2_means = zeros(Float64, n_grid)
    ratio_means = zeros(Float64, n_grid)
    dpr_means = zeros(Float64, n_grid)
    
    for (t_idx, tau_val) in enumerate(tau_grid_eval)
        m_slice = abs.(data[:, 1] .- tau_val) .< 1e-4
        X_slice = data[m_slice, :]
        T_curr = X_slice[:, 2]
        A_curr = X_slice[:, 3]
        
        Z = transform_slice(T_curr, A_curr, m_key)
        tree = KDTree(permutedims(Z))
        idxs, _ = knn(tree, permutedims(Z), k_eval_fixed, true)
        
        sum_l1 = 0.0
        sum_l2 = 0.0
        sum_ratio = 0.0
        sum_dpr = 0.0
        
        for i in 1:N_slice
            nbrs = idxs[i]
            Y = Z[nbrs, :]
            Yc = Y .- mean(Y, dims=1)
            C = (permutedims(Yc) * Yc) ./ (k_eval_fixed - 1)
            evals = sort(eigvals(Symmetric(C)), rev=true)
            l1 = max(evals[1], 1e-35)
            l2 = max(evals[2], 1e-35)
            
            sum_l1 += l1
            sum_l2 += l2
            sum_ratio += (l2 / l1)
            sum_dpr += ((l1 + l2)^2) / (l1^2 + l2^2)
        end
        
        l1_means[t_idx] = sum_l1 / N_slice
        l2_means[t_idx] = sum_l2 / N_slice
        ratio_means[t_idx] = sum_ratio / N_slice
        dpr_means[t_idx] = sum_dpr / N_slice
    end
    
    eigen_results[m_key] = (
        l1 = l1_means,
        l2 = l2_means,
        ratio = ratio_means,
        dpr = dpr_means
    )
end

fig2 = Figure(size = (1200, 390), figure_padding = (25, 25, 20, 20))

ax2a = Axis(
    fig2[1, 1],
    yscale = log10,
    xlabel = L"\tau\ [\mathrm{fm}/c]",
    ylabel = L"\langle \lambda_1 \rangle\ \text{(Tangent Variance)}",
    title = L"\text{(a) Longitudinal Variance }\lambda_1(\tau)"
)

ax2b = Axis(
    fig2[1, 2],
    yscale = log10,
    xlabel = L"\tau\ [\mathrm{fm}/c]",
    ylabel = L"\langle \lambda_2 \rangle\ \text{(Transverse Variance)}",
    title = L"\text{(b) Transverse Variance }\lambda_2(\tau)"
)

ax2c = Axis(
    fig2[1, 3],
    yscale = log10,
    xlabel = L"\tau\ [\mathrm{fm}/c]",
    ylabel = L"\langle \lambda_2 / \lambda_1 \rangle\ \text{(Spectral Ratio)}",
    title = L"\text{(c) Eigenvalue Ratio }\lambda_2 / \lambda_1"
)

for minfo in methods_info
    m_key = minfo.key
    res = eigen_results[m_key]
    lines!(ax2a, tau_grid_eval, res.l1, label = minfo.name, color = minfo.color, linestyle = minfo.linestyle, linewidth = 2.4)
    lines!(ax2b, tau_grid_eval, res.l2, label = minfo.name, color = minfo.color, linestyle = minfo.linestyle, linewidth = 2.4)
    lines!(ax2c, tau_grid_eval, res.ratio, label = minfo.name, color = minfo.color, linestyle = minfo.linestyle, linewidth = 2.4)
end

# Theoretical scaling overlay for Variant 1 in panel (a): (tau_0 / tau)^(2/3)
l1_theory_var1 = eigen_results[:var1].l1[1] .* (tau0 ./ tau_grid_eval).^(2/3)
lines!(ax2a, tau_grid_eval, l1_theory_var1, color = :black, linestyle = :dot, linewidth = 2.0, label = L"\text{Bjorken }\propto \tau^{-2/3}")

axislegend(ax2a, position = :lb, framevisible = false, labelsize = 11)
axislegend(ax2b, position = :rt, framevisible = false, labelsize = 11)
axislegend(ax2c, position = :rt, framevisible = false, labelsize = 11)

save(joinpath(output_dir, "fig2_eigenvalues_and_ratio_vs_tau.pdf"), fig2)
println("Saved: ", joinpath(output_dir, "fig2_eigenvalues_and_ratio_vs_tau.pdf"))

# -----------------------------------------------------------------------------
# Figure 3: k-Sweep and Automatic Plateau Detection
# -----------------------------------------------------------------------------
println("\nGenerating Figure 3: k-Sweep and Plateau Detection (lpca_style)...")
k_sweep_list = 3:1:50
n_k = length(k_sweep_list)
plateau_tau_targets = [0.20, 0.55, 2.00]

fig3 = Figure(size = (1200, 700), figure_padding = (25, 25, 20, 20))

plateau_results = Dict{Tuple{Symbol, Float64}, NamedTuple}()

for (col_idx, tau_val) in enumerate(plateau_tau_targets)
    m_slice = abs.(data[:, 1] .- tau_val) .< 1e-4
    X_slice = data[m_slice, :]
    T_curr = X_slice[:, 2]
    A_curr = X_slice[:, 3]
    
    ax_top = Axis(
        fig3[1, col_idx],
        xlabel = "",
        ylabel = col_idx == 1 ? L"\langle d_{\mathrm{PR}}(k) \rangle" : "",
        title = latexstring("\\tau = $(@sprintf("%.2f", tau_val))\\ \\mathrm{fm}/c"),
        titlesize = 15
    )
    ylims!(ax_top, 0.95, 2.05)
    
    ax_bot = Axis(
        fig3[2, col_idx],
        xlabel = L"k\ \text{(Neighbors)}",
        ylabel = col_idx == 1 ? L"\Delta(k) = |\Delta \langle d_{\mathrm{PR}} \rangle|" : "",
        yscale = log10,
        title = L"\text{Finite Step Difference }\Delta(k)"
    )
    ylims!(ax_bot, 1e-4, 3e-1)
    
    for minfo in methods_info
        m_key = minfo.key
        Z = transform_slice(T_curr, A_curr, m_key)
        tree = KDTree(permutedims(Z))
        max_k = maximum(k_sweep_list)
        knn_idxs, _ = knn(tree, permutedims(Z), max_k, true)
        
        dpr_k = zeros(Float64, n_k)
        for (k_idx, k_val) in enumerate(k_sweep_list)
            sum_dpr = 0.0
            for i in 1:N_slice
                nbrs = knn_idxs[i][1:k_val]
                Y = Z[nbrs, :]
                Yc = Y .- mean(Y, dims=1)
                C = (permutedims(Yc) * Yc) ./ (k_val - 1)
                evals = sort(eigvals(Symmetric(C)), rev=true)
                l1 = max(evals[1], 1e-35)
                l2 = max(evals[2], 1e-35)
                sum_dpr += ((l1 + l2)^2) / (l1^2 + l2^2)
            end
            dpr_k[k_idx] = sum_dpr / N_slice
        end
        
        # Finite difference Delta(k) = |d_PR(k+1) - d_PR(k)|
        diff_dpr = abs.(diff(dpr_k))
        k_diff_axis = collect(k_sweep_list[1:end-1])
        
        # Automatic plateau detection: local minimum of Delta(k) for k >= 7
        k_search_range = 7:(length(diff_dpr))
        local_min_idx = argmin(diff_dpr[k_search_range])
        best_k_idx = k_search_range[local_min_idx]
        k_star = k_diff_axis[best_k_idx]
        dpr_star = dpr_k[best_k_idx]
        delta_star = diff_dpr[best_k_idx]
        
        plateau_results[(m_key, tau_val)] = (
            k_vals = collect(k_sweep_list),
            dpr_vals = dpr_k,
            diff_vals = diff_dpr,
            k_star = k_star,
            dpr_star = dpr_star
        )
        
        # Plot curves
        lines!(ax_top, collect(k_sweep_list), dpr_k, label = minfo.name, color = minfo.color, linestyle = minfo.linestyle, linewidth = 2.2)
        scatter!(ax_top, [k_star], [dpr_star], color = minfo.color, markersize = 10, marker = :diamond)
        vlines!(ax_top, [k_star], color = (minfo.color, 0.45), linestyle = :dot, linewidth = 1.3)
        
        lines!(ax_bot, k_diff_axis, diff_dpr, label = minfo.name, color = minfo.color, linestyle = minfo.linestyle, linewidth = 2.0)
        scatter!(ax_bot, [k_star], [delta_star], color = minfo.color, markersize = 10, marker = :diamond)
        
        @printf("  tau = %.2f | %-30s | k* = %2d | d_PR(k*) = %.4f | Delta(k*) = %.2e\n",
                tau_val, minfo.name, k_star, dpr_star, delta_star)
    end
    
    if col_idx == 1
        axislegend(ax_top, position = :rb, framevisible = false, labelsize = 10)
    end
end

save(joinpath(output_dir, "fig3_k_plateau_analysis.pdf"), fig3)
println("Saved: ", joinpath(output_dir, "fig3_k_plateau_analysis.pdf"))

# -----------------------------------------------------------------------------
# Figure 4: Dimension Trajectories Across Time d_PR(tau)
# -----------------------------------------------------------------------------
println("\nGenerating Figure 4: Dimension Trajectories Across Tau (lpca_style)...")

fig4 = Figure(size = (1100, 480), figure_padding = (25, 25, 20, 20))

ax4a = Axis(
    fig4[1, 1],
    xlabel = L"\tau\ [\mathrm{fm}/c]",
    ylabel = L"\langle d_{\mathrm{PR}}(\tau) \rangle",
    title = L"\text{(a) Fixed Neighborhood }k = 16"
)
ylims!(ax4a, 0.95, 2.00)

ax4b = Axis(
    fig4[1, 2],
    xlabel = L"\tau\ [\mathrm{fm}/c]",
    ylabel = L"\langle d_{\mathrm{PR}}(\tau) \rangle",
    title = L"\text{(b) Dynamic Plateau Neighborhood }k^*(\tau)"
)
ylims!(ax4b, 0.95, 2.00)

k_fixed_val = 16
summary_csv_path = joinpath(results_dir, "frozen_scaling_summary.csv")

# Subsampling for fast sweep of k* across all tau slices
n_eval_kstar = min(3000, N_slice)
eval_sub_kstar = round.(Int, range(1, N_slice, length = n_eval_kstar))

open(summary_csv_path, "w") do f_csv
    write(f_csv, "tau,method,dpr_fixed_k16,k_star,dpr_k_star,lambda1,lambda2,ratio\n")
    
    for minfo in methods_info
        m_key = minfo.key
        dpr_fixed_traj = zeros(Float64, n_grid)
        dpr_star_traj = zeros(Float64, n_grid)
        k_star_traj = zeros(Int, n_grid)
        
        for (t_idx, tau_val) in enumerate(tau_grid_eval)
            m_slice = abs.(data[:, 1] .- tau_val) .< 1e-4
            X_slice = data[m_slice, :]
            T_curr = X_slice[:, 2]
            A_curr = X_slice[:, 3]
            
            Z = transform_slice(T_curr, A_curr, m_key)
            tree = KDTree(permutedims(Z))
            
            # Sweep k in 3..35 on representative subset for fast robust k*
            k_sweep_sub = 3:1:35
            max_k_sub = maximum(k_sweep_sub)
            knn_idxs_sub, _ = knn(tree, permutedims(Z[eval_sub_kstar, :]), max_k_sub, true)
            
            dpr_sweep = zeros(Float64, length(k_sweep_sub))
            for (k_idx, k_val) in enumerate(k_sweep_sub)
                s_d = 0.0
                for i in 1:n_eval_kstar
                    nbrs = knn_idxs_sub[i][1:k_val]
                    Y = Z[nbrs, :]
                    Yc = Y .- mean(Y, dims=1)
                    C = (permutedims(Yc) * Yc) ./ (k_val - 1)
                    evals = sort(eigvals(Symmetric(C)), rev=true)
                    l1 = max(evals[1], 1e-35)
                    l2 = max(evals[2], 1e-35)
                    s_d += ((l1 + l2)^2) / (l1^2 + l2^2)
                end
                dpr_sweep[k_idx] = s_d / n_eval_kstar
            end
            
            # Fixed k = 16 (from full eigen_results)
            dpr_fixed_traj[t_idx] = eigen_results[m_key].dpr[t_idx]
            
            # Plateau k*
            diff_sweep = abs.(diff(dpr_sweep))
            search_rng = 7:(length(diff_sweep))
            k_star_idx = search_rng[argmin(diff_sweep[search_rng])]
            k_star = collect(k_sweep_sub)[k_star_idx]
            dpr_star_traj[t_idx] = dpr_sweep[k_star_idx]
            k_star_traj[t_idx] = k_star
            
            # Write to CSV
            res_eig = eigen_results[m_key]
            @printf(f_csv, "%.3f,%s,%.4f,%d,%.4f,%.4e,%.4e,%.4e\n",
                    tau_val, string(m_key), dpr_fixed_traj[t_idx], k_star, dpr_star_traj[t_idx],
                    res_eig.l1[t_idx], res_eig.l2[t_idx], res_eig.ratio[t_idx])
        end
        
        lines!(ax4a, tau_grid_eval, dpr_fixed_traj, label = minfo.name, color = minfo.color, linestyle = minfo.linestyle, linewidth = 2.4)
        scatter!(ax4a, tau_grid_eval, dpr_fixed_traj, color = minfo.color, markersize = 6)
        
        lines!(ax4b, tau_grid_eval, dpr_star_traj, label = minfo.name, color = minfo.color, linestyle = minfo.linestyle, linewidth = 2.4)
        scatter!(ax4b, tau_grid_eval, dpr_star_traj, color = minfo.color, markersize = 6)
    end
end

axislegend(ax4a, position = :rt, framevisible = false, labelsize = 11)
axislegend(ax4b, position = :rt, framevisible = false, labelsize = 11)

save(joinpath(output_dir, "fig4_dimension_trajectories.pdf"), fig4)
println("Saved: ", joinpath(output_dir, "fig4_dimension_trajectories.pdf"))
println("Saved summary CSV: ", summary_csv_path)

println("\n=== All computations and figure exports completed successfully! ===")
