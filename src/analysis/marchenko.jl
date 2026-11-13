using LinearAlgebra
using Statistics
using LaTeXStrings

function _marchenko_pastur_band(mean_evals::AbstractVector{<:Real}, k::Integer)
    d = length(mean_evals)
    q = d / k
    sigma2_loc = sum(mean_evals) / d
    lambda_minus = q < 1.0 ? sigma2_loc * (1.0 - sqrt(q))^2 : 0.0
    lambda_plus = sigma2_loc * (1.0 + sqrt(q))^2
    return (lambda_minus, lambda_plus)
end

function _compute_scales_at_tau0(
        dane::AbstractMatrix{<:Real},
        tau0::Real,
        feature_cols::AbstractVector{<:Integer};
        atol::Real = 1.0e-5
    )
    _, X0 = get_tau_slice(dane, tau0; atol = atol, feature_cols = feature_cols, nearest = true)
    scales = vec(std(X0, dims = 1))
    scales[scales .== 0.0] .= 1.0
    return scales
end

function marchenko(
        dane::AbstractMatrix{<:Real};
        tau::Real = 0.2,
        tau0::Real = 0.2,
        K::Integer = 160,
        T_unit::Real = 193.0,
        K_sweep = [6:4:250...],
        D::Int = 3,
        atol::Real = 1.0e-5,
        show_spectrum::Bool = false
    )
    set_publication_theme()
    colors = LPCA_COLORS
    feature_cols = collect(2:(D + 1))

    scales = _compute_scales_at_tau0(dane, tau0, feature_cols; atol = atol)
    slice_res = lpca_frozen_slice(dane, tau, scales; k = K, feature_cols = feature_cols)
    mp_band = _marchenko_pastur_band(slice_res.mean_evals, slice_res.k)

    n_pts = length(slice_res.d_pr)
    k_list = sort(unique(filter(k -> 2 <= k < n_pts, collect(K_sweep))))
    sweep_res = k_sweep_frozen(dane, tau, scales; k_range = k_list, feature_cols = feature_cols)

    fig_width = show_spectrum ? 1850 : 1300
    fig = Figure(size = (fig_width, 400), figure_padding = (30, 25, 20, 20))

    ax1 = Axis(
        fig[1, 1],
        title = L"\text{(a) } d_{\mathrm{PR}} \ (K=%$(K)), \tau = %$(tau)",
        xlabel = L"d_{\mathrm{PR}}",
        ylabel = L"N / N_{\max}",
        limits = (0.95, D + 0.05, 0.0, 1.05)
    )
    centers, scaled, bin_width = _bin_pr_dimensions(slice_res.d_pr; n_bins = 40, range_limits = (1.0, D))
    barplot!(
        ax1, centers, scaled;
        width = bin_width * 0.95,
        color = centers,
        colormap = lpca_style,
        colorrange = (1.0, D),
        strokewidth = 0.5,
        strokecolor = (:black, 0.4)
    )
    vlines!(
        ax1, [slice_res.mean_dpr],
        color = colors[1], linestyle = :dash, linewidth = 2.5,
        label = L"\langle d_{\mathrm{PR}} \rangle = %$(round(slice_res.mean_dpr, digits=2))"
    )
    axislegend(ax1, position = :rt, labelsize = 12)

    ax2 = Axis(
        fig[1, 2],
        title = L"\text{(b) Scan plateau } \langle d_{\mathrm{PR}}(K) \rangle",
        xlabel = L"K \text{ (Number of KNN)}",
        ylabel = L"\langle d_{\mathrm{PR}} \rangle",
        limits = (minimum(k_list), maximum(k_list), 0.95, D + 0.05)
    )
    band!(
        ax2, sweep_res.k_values,
        sweep_res.mean_dpr .- sweep_res.std_dpr,
        sweep_res.mean_dpr .+ sweep_res.std_dpr,
        color = (colors[4], 0.2)
    )
    lines!(ax2, sweep_res.k_values, sweep_res.mean_dpr, color = colors[4], linewidth = 2.8, label = L"\langle d_{\mathrm{PR}} \rangle \pm 1\sigma")
    vlines!(ax2, [K], color = colors[1], linestyle = :dash, linewidth = 2.0, label = L"K = %$(K)")
    axislegend(ax2, position = :rb, labelsize = 12)

    if show_spectrum
        lambda_minus, lambda_plus = mp_band
        ax3 = Axis(
            fig[1, 3],
            yscale = log10,
            title = L"\text{(c) Spectrum } \langle \lambda_j \rangle \text{ vs range Marchenko Pasteur}",
            xlabel = L"\text{Index od modes } j",
            ylabel = L"\langle \lambda_j \rangle",
            xticks = (1:D, [latexstring("\\lambda_{$j}") for j in 1:D])
        )
        scatterlines!(ax3, 1:D, max.(slice_res.mean_evals, 1.0e-35), color = colors[1], markersize = 14, linewidth = 2.0, label = L"\text{Średnie } \langle \lambda_j \rangle")
        band!(
            ax3, [0.5, Float64(D) + 0.5],
            fill(max(lambda_minus, 1.0e-35), 2),
            fill(lambda_plus, 2),
            color = (:gray70, 0.35), label = L"\text{Pasmo szumu MP}"
        )
        hlines!(ax3, [lambda_plus], color = :crimson, linestyle = :dash, linewidth = 1.5)
        lambda_minus > 0 && hlines!(ax3, [lambda_minus], color = :navy, linestyle = :dash, linewidth = 1.5)
        axislegend(ax3, position = :rt, labelsize = 12)
    end

    # display(fig)
    return (
        mean_dpr = slice_res.mean_dpr,
        mean_evals = slice_res.mean_evals,
        mp_band = mp_band,
        fig = fig,
    )
end

marchenko(dane::AbstractArray{<:Real, 3}, args...; kwargs...) = marchenko(to_2d_local_plots(dane), args...; kwargs...)
