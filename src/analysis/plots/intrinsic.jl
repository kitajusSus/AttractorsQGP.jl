function plot_pinn_deff_evolution(results::AbstractMatrix{<:Real})
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    fig = Figure(size = (850, 600))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Ewolucja wymiaru efektywnego } d_{\mathrm{eff}} \text{ w czasie } \tau",
        xlabel = L"\tau\,[\mathrm{fm}/c]",
        ylabel = L"d_{\mathrm{eff}}",
        limits = (minimum(results[:, 1]), maximum(results[:, 1]), 0.95, maximum(results[:, 2]) * 1.05)
    )
    lines!(ax, results[:, 1], results[:, 2], color = palette[1], linewidth = 3.0)
    return fig
end

function plot_lid_dimension(dataset::AbstractMatrix{<:Real})
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    fig = Figure(size = (1200, 700))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Ewolucja estymowanego wymiaru LID w funkcji czasu dla różnych } k",
        xlabel = L"\text{Czas } \tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{estymowany wymiar } d_{\mathrm{LID}}"
    )
    for (idx, current_k) in enumerate(1:10:100)
        taus, lid, _, _ = scan_intrinsic_dimensions(dataset, k = current_k)
        lines!(ax, taus, lid, color = palette[idx], linewidth = 2.5, label = latexstring("k = ", current_k))
    end
    return fig
end

function plot_twonn(dataset::AbstractMatrix{<:Real})
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    fig = Figure(size = (1200, 700))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Przykładowy wykres dla TWONN - Two Nearest Neighbors}",
        xlabel = L"\text{Czas } \tau\,[\mathrm{fm}/c]",
        ylabel = L"\text{estymowany wymiar } d_{\mathrm{TWONN}}"
    )
    taus, _, twonn, _ = scan_intrinsic_dimensions(dataset)
    lines!(ax, taus, twonn, color = palette[3], linewidth = 2.5, label = L"\text{TWONN}")
    axislegend(ax, position = :rt)
    return fig
end

plot_pinn_deff_evolution(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_pinn_deff_evolution(to_2d_local_plots(dataset), args...; kwargs...)
plot_lid_dimension(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_lid_dimension(to_2d_local_plots(dataset), args...; kwargs...)
plot_twonn(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_twonn(to_2d_local_plots(dataset), args...; kwargs...)
