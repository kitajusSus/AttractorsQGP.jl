function plot_lle_spectrum_statistics(dataset; τ, k_values = 5:5:50, ile_λ = 4)
    k_values = flatten_k_values(k_values)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    spectra, _ = lle_spectrum_over_k(dataset; tau = τ, k_values = k_values)
    μ, σ = spectrum_statistics(spectra)
    x = 1:ile_λ
    k_min, k_max = extrema(k_values)

    fig = Figure(size = (1400, 700))
    ax = Axis(fig[1, 1], title = L"\text{LLE analiza wartości własnych dla K z zakresu (%$k_min - %$k_max) (średnia } \pm \sigma \text{) } \tau=%$τ", xlabel = L"\text{Indeks wartości własnej } \lambda_{i}", ylabel = L"\text{Wartość } \lambda_{i}")
    scatterlines!(ax, x, μ[1:ile_λ], linewidth = 2, color = (palette[5], 0.3), markercolor = palette[5], markersize = 15, label = L"\text{Średnia } i\text{-ta } \lambda_{i}")
    band!(ax, x, μ[1:ile_λ] .- σ[1:ile_λ], μ[1:ile_λ] .+ σ[1:ile_λ], color = (palette[3], 0.2))
    axislegend(ax, position = :lt)
    return fig
end

function plot_lle_spectrum_statistics_grid(dataset, taus; k_values = 5:5:50, ile_λ = 4, atol = 1.0e-8)
    k_values = flatten_k_values(k_values)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    n = length(taus)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    spectra1, _ = lle_spectrum_over_k(dataset; tau = taus[1], k_values = k_values, atol = atol)
    μ1, σ1 = spectrum_statistics(spectra1)
    y_min = minimum(μ1[1:ile_λ] .- σ1[1:ile_λ])
    y_max = maximum(μ1[1:ile_λ] .+ σ1[1:ile_λ])
    y_pad = (y_max - y_min) * 0.07
    lims = (0.9, ile_λ + 0.05, y_min - y_pad, y_max + y_pad)

    fig = Figure(size = (450 * ncols + 40, 350 * nrows + 120))
    k_min, k_max = extrema(k_values)
    Label(fig[1, 2:(ncols + 1)], L"\text{LLE analiza wartości własnych dla K z zakresu (%$k_min - %$k_max) (średnia } \pm \sigma \text{)}", fontsize = 36, font = :bold, padding = (0, 0, 10, 0))
    Label(fig[2:(nrows + 1), 1], L"\text{Wartość } \lambda_{i}", rotation = pi / 2, font = :bold, fontsize = 32)
    Label(fig[nrows + 2, 2:(ncols + 1)], L"\text{Indeks wartości własnej } \lambda_{i}", font = :bold, fontsize = 32)

    x = 1:ile_λ
    for (i, tau) in enumerate(taus)
        row = (i - 1) ÷ ncols + 2
        col = (i - 1) % ncols + 2
        μ, σ = i == 1 ? (μ1, σ1) : spectrum_statistics(lle_spectrum_over_k(dataset; tau = tau, k_values = k_values, atol = atol)[1])
        ax = Axis(fig[row, col], title = L"\tau = %$tau\,\mathrm{fm}/c", limits = lims, xticks = 1:ile_λ)
        scatterlines!(ax, x, μ[1:ile_λ], linewidth = 2, color = (palette[5], 0.3), markercolor = palette[5], markersize = 15, label = L"\text{Średnia } i\text{-ta } \lambda_{i}")
        band!(ax, x, μ[1:ile_λ] .- σ[1:ile_λ], μ[1:ile_λ] .+ σ[1:ile_λ], color = (palette[3], 0.2))
        axislegend(ax, position = :lt)
        row_idx = (i - 1) ÷ ncols + 1
        col_idx = (i - 1) % ncols + 1
        !((row_idx == nrows) || (i + ncols > n)) && hidexdecorations!(ax, grid = false)
        !(col_idx == 1) && hideydecorations!(ax, grid = false)
    end
    return fig
end

@views function plot_lle_σ_spectrum(dataset, taus; k_values = 5:5:50, ile_λ = 3, atol = 1.0e-8)
    k_values = flatten_k_values(k_values)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    n_taus = length(taus)
    sigma_matrix = fill(NaN, n_taus, ile_λ)
    for (i, tau) in enumerate(taus)
        spectra, _ = lle_spectrum_over_k(dataset; tau = tau, k_values = k_values, atol = atol)
        _, σ = spectrum_statistics(spectra)
        N = min(ile_λ, length(σ))
        sigma_matrix[i, 1:N] .= σ[1:N] .* 10^(8)
    end
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title = L"\text{Zmiana } \sigma_i \text{ w funkcji } \tau \text{ dla } i \le %$ile_λ", xlabel = L"\tau \ [\mathrm{fm/c}]", ylabel = L"\sigma_i \cdot 10^{8}", xticks = 0.2:0.05:maximum(taus))
    for j in 1:ile_λ
        lines!(ax, taus, sigma_matrix[:, j], linewidth = 3, color = palette[(j + 1) * 3], label = L"\sigma_{%$j}")
        scatter!(ax, taus, sigma_matrix[:, j], color = palette[j + 3], markersize = 5)
    end
    axislegend(ax, position = :rt)
    return fig
end

function plot_lle_dim(dataset::AbstractMatrix{<:Real}, k::Int, d::Int, tau::Real)
    set_publication_theme()
    palette = Makie.theme(:Palette, :color)[]
    embedding = run_lle_per_time(dataset; k = k, d = d).lle_results[tau]
    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1], title = L"\text{LLE: } k=%$k, d=%$d, \tau=%$tau")
    if d == 1
        ax.xlabel = L"\text{LLE1}"
        ax.ylabel = L"\text{Wartość stała}"
        scatter!(ax, embedding[:, 1], zeros(size(embedding, 1)); markersize = 4.5, color = (palette[1], 0.75))
    else
        ax.xlabel = L"\text{LLE1}"
        ax.ylabel = L"\text{LLE2}"
        scatter!(ax, embedding[:, 1], embedding[:, 2]; markersize = 4.5, color = (palette[1], 0.75))
    end
    return fig
end

function plot_lle_dim!(ax::Axis, dataset::AbstractMatrix{<:Real}, k::Int, d::Int, tau::Real)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    embedding = run_lle_per_time(dataset; k = k, d = d).lle_results[tau]
    ax.title = L"\text{LLE: } k=%$k, d=%$d, \tau=%$tau"
    if d == 1
        ax.xlabel = L"\text{Odwzorowanie 1}"
        ax.ylabel = L"\text{Wartość stała}"
        scatter!(ax, embedding[:, 1], zeros(size(embedding, 1)); markersize = 4.5, color = (palette[1], 0.75))
    else
        ax.xlabel = L"\text{Odwzorowanie LLE1}"
        ax.ylabel = L"\text{Odwzorowanie LLE2}"
        scatter!(ax, embedding[:, 1], embedding[:, 2]; markersize = 4.5, color = (palette[1], 0.75))
    end
    return nothing
end

function plot_simulation_lle(dataset::AbstractMatrix{<:Real}, k::Int, d::Int, tau_zakres)
    set_publication_theme()
    kolumny = 2
    wiersze = ceil(Int, length(tau_zakres) / kolumny)
    bok = max(kolumny, wiersze)
    fig = Figure(size = (bok * 400, bok * 400))
    for (i, tau) in enumerate(tau_zakres)
        row = (i - 1) ÷ kolumny + 1
        col = (i - 1) % kolumny + 1
        ax = Axis(fig[row, col], aspect = 1)
        plot_lle_dim!(ax, dataset, k, d, tau)
    end
    return fig
end

function plot_lle_embedding(dataset::AbstractMatrix{<:Real}; labels = nothing, title = L"\text{Projekcja LLE}", osie = nothing)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    dim = size(dataset, 2)
    fig = Figure(size = (750, 600))
    color_param = isnothing(labels) ? (palette[1], 0.75) : labels
    cmap_param = isnothing(labels) ? :viridis : :jet
    if dim >= 3
        ax = Axis3(fig[1, 1], title = title)
        scatter!(ax, dataset[:, 1], dataset[:, 2], dataset[:, 3], color = color_param, colormap = cmap_param, markersize = 6)
    elseif dim == 2
        ax = Axis(fig[1, 1], title = title, xlabel = L"\text{LLE1}", ylabel = L"\text{LLE2}")
        scatter!(ax, dataset[:, 1], dataset[:, 2], color = color_param, colormap = cmap_param, markersize = 6)
    else
        ax = Axis(fig[1, 1], title = title, xlabel = L"\text{LLE1}")
        scatter!(ax, dataset[:, 1], zeros(size(dataset, 1)), color = color_param, colormap = cmap_param, markersize = 6)
    end
    return fig
end

function plot_lle_grid(lle_results::Dict, taus; labels = nothing)
    dim = size(first(values(lle_results)), 2)
    return dim >= 3 ? plot_lle_grid_3d(lle_results, taus; labels = labels) : plot_lle_grid_2d(lle_results, taus; labels = labels)
end

function plot_lle_grid_2d(lle_results::Dict, taus; labels = nothing)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    color_param = isnothing(labels) ? (palette[1], 0.75) : labels
    cmap_param = isnothing(labels) ? :viridis : :jet
    n = length(taus)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size = (400 * ncols, 380 * nrows))
    for (i, t) in enumerate(taus)
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1
        embedding = lle_results[t]
        ax = Axis(fig[row, col], title = L"\tau = %$(round(t, digits=2))", xlabel = L"\text{LLE1}", ylabel = L"\text{LLE2}")
        scatter!(ax, embedding[:, 1], embedding[:, 2], color = color_param, colormap = cmap_param, markersize = 5)
    end
    return fig
end

function plot_lle_grid_3d(lle_results::Dict, taus; labels = nothing)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    color_param = isnothing(labels) ? (palette[1], 0.75) : labels
    cmap_param = isnothing(labels) ? :viridis : :jet
    n = length(taus)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size = (400 * ncols, 380 * nrows))
    for (i, t) in enumerate(taus)
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1
        embedding = lle_results[t]
        ax = Axis3(fig[row, col], title = L"\tau = %$(round(t, digits=2))", xlabel = L"\text{LLE1}", ylabel = L"\text{LLE2}", zlabel = L"\text{LLE3}")
        scatter!(ax, embedding[:, 1], embedding[:, 2], embedding[:, 3], color = color_param, colormap = cmap_param, markersize = 5)
    end
    return fig
end

function plot_lle_results_for_taus(
        dataset::AbstractMatrix{<:Real},
        target_taus::AbstractVector{<:Real};
        k::Integer = 20,
        d::Integer = 2,
        atol::Real = 1.0e-3,
        feature_cols::AbstractVector{<:Integer} = collect(2:size(dataset, 2))
    )
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    results = run_lle_for_selected_taus(dataset, target_taus; k = k, d = d, atol = atol, feature_cols = feature_cols)
    lle_res = results.lle_results
    n = length(target_taus)
    ncols = min(3, n)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size = (max(1400, 450 * ncols), max(700, 450 * nrows)))
    for (i, t) in enumerate(target_taus)
        !haskey(lle_res, t) && continue
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1
        embedding = lle_res[t]
        dim = size(embedding, 2)
        ax_title = L"\tau = %$(round(t, digits=2))"
        if dim >= 3
            ax = Axis3(fig[row, col], title = ax_title, xlabel = L"\text{LLE1}", ylabel = L"\text{LLE2}", zlabel = L"\text{LLE3}")
            scatter!(ax, embedding[:, 1], embedding[:, 2], embedding[:, 3], color = (palette[1], 0.75), markersize = 6)
        elseif dim == 2
            ax = Axis(fig[row, col], title = ax_title, xlabel = L"\text{LLE1}", ylabel = L"\text{LLE2}")
            scatter!(ax, embedding[:, 1], embedding[:, 2], color = (palette[1], 0.75), markersize = 6)
        else
            ax = Axis(fig[row, col], title = ax_title, xlabel = L"\text{LLE1}")
            scatter!(ax, embedding[:, 1], zeros(size(embedding, 1)), color = (palette[1], 0.75), markersize = 6)
        end
    end
    return fig
end

function plot_lle_spectrum_analysis(dataset::AbstractMatrix{<:Real}, tau::Real; k_values = 5:5:50, atol::Real = 1.0e-8)
    k_values = flatten_k_values(k_values)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    spectra, ks = lle_spectrum_over_k(dataset; tau = tau, k_values = k_values, atol = atol)
    fig = Figure(size = (1400, 700))
    ax = Axis(fig[1, 1], title = L"\text{Widmo LLE dla różnych } k \text{, } \tau = %$(tau)", xlabel = L"\text{Indeks wartości własnej}", ylabel = L"\lambda")
    for (i, (s, k)) in enumerate(zip(spectra, ks))
        col = palette[mod1(i, length(palette))]
        lines!(ax, 1:length(s), s, label = L"k = %$k", linewidth = 2.5, color = col)
    end
    axislegend(ax, position = :lt, nbanks = 2)
    return fig
end

function plot_lle_spectrum_scan_analysis(dataset::AbstractMatrix{<:Real}, taus::AbstractVector{<:Real}; k_values = 5:5:50)
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    results = scan_lle_spectrum(dataset; taus = taus, k_values = k_values)
    fig = Figure(size = (1400, 700))
    ax = Axis(fig[1, 1], title = L"\text{Statystyki widma LLE w czasie}", xlabel = L"\text{Indeks wartości własnej}", ylabel = L"\lambda")
    for (i, t) in enumerate(taus)
        !haskey(results, t) && continue
        res = results[t]
        μ, σ = res.mean, res.std
        col = palette[mod1(i, length(palette))]
        lines!(ax, 1:length(μ), μ, label = L"\tau = %$(t)", linewidth = 3, color = col)
        band!(ax, 1:length(μ), μ .- σ, μ .+ σ, color = (col, 0.2))
    end
    axislegend(ax, position = :rt, nbanks = 2)
    return fig
end

plot_lle_dim(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_lle_dim(to_2d_local_plots(dataset), args...; kwargs...)
plot_lle_dim!(ax::Axis, dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_lle_dim!(ax, to_2d_local_plots(dataset), args...; kwargs...)
plot_simulation_lle(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_simulation_lle(to_2d_local_plots(dataset), args...; kwargs...)
plot_lle_embedding(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_lle_embedding(to_2d_local_plots(dataset), args...; kwargs...)
plot_lle_results_for_taus(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_lle_results_for_taus(to_2d_local_plots(dataset), args...; kwargs...)
plot_lle_spectrum_analysis(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_lle_spectrum_analysis(to_2d_local_plots(dataset), args...; kwargs...)
plot_lle_spectrum_scan_analysis(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_lle_spectrum_scan_analysis(to_2d_local_plots(dataset), args...; kwargs...)
