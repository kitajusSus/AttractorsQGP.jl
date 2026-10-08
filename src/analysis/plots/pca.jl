function plot_pca_evr_over_time(
        dataset::AbstractMatrix{<:Real};
        n_components::Integer = 2,
        method::Symbol = :minmax,
        gamma::Real = 1.0,
        feature_cols::AbstractVector = collect(2:size(dataset, 2)),
        plot_title::Union{String, LaTeXString} = L"\text{Explained Variance Ratio (EVR) w funkcji czasu}",
        x_label::Union{String, LaTeXString} = L"\tau\,[\mathrm{fm}/c]",
        y_label::Union{String, LaTeXString} = L"\text{EVR}",
        tau_min::Union{Real, Nothing} = nothing
    )
    set_publication_theme()
    result = run_pca_per_time(dataset; n_components = n_components, method = method, gamma = gamma, feature_cols = feature_cols)
    taus = result.taus
    evr = result.explained_variance_ratio
    t_min = isnothing(tau_min) ? minimum(taus) : Float64(tau_min)

    fig = Figure(size = (850, 600))
    ax = Axis(fig[1, 1], title = plot_title, xlabel = x_label, ylabel = y_label, limits = (t_min, maximum(taus), 0, 1.05))
    hlines!(ax, [1.0], color = :gray45, linestyle = :dash, label = L"100\%")
    palette = Makie.theme(:Palette).color[]
    kolory = [palette[3], palette[7], palette[3], palette[7], palette[2], palette[8]]

    for comp in 1:n_components
        vals = evr[:, comp]
        mask = .!isnan.(vals)
        if any(mask)
            lines!(ax, taus[mask], vals[mask], linewidth = 3.0, color = kolory[mod1(comp, length(kolory))], label = L"\text{PC}%$(comp)")
        end
    end
    axislegend(ax, position = :rb)
    return fig
end

function plot_pca_bar_variance(
        dataset::AbstractMatrix{<:Real};
        tau::Real = 1.14,
        method::Symbol = :minmax,
        gamma::Real = 1.0,
        feature_cols::Union{AbstractVector{<:Integer}, Nothing} = nothing
    )
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    cols = isnothing(feature_cols) ? (2:size(dataset, 2)) : feature_cols
    _, Xtau = get_tau_slice(dataset, tau; atol = 1.0e-8, feature_cols = cols)

    pca_result = method === :minmax ? run_pca(Xtau; n_components = 2) : (method === :kernel ? run_pca_kernel(Xtau; n_components = 2, gamma = gamma) : error("Nieznana metoda PCA"))
    evr = pca_result.explained_variance_ratio_full
    n_comp = length(evr)
    cumulative_evr = cumsum(evr)

    fig = Figure(size = (850, 550))
    ax = Axis(fig[1, 1], title = L"\text{Analiza Wariancji — PCA z normalizacją min-max}", xlabel = L"\text{Główna Składowa}", ylabel = L"\text{EVR [\%]}", xticks = (1:n_comp, [L"\mathrm{PC}_%$(i)" for i in 1:n_comp]), limits = (0.4, n_comp + 0.6, 0, 110))
    evr_percent = evr .* 100
    barplot!(ax, 1:n_comp, evr_percent, color = palette[1], width = 0.5)

    for i in 1:n_comp
        text!(ax, i, evr_percent[i] + 3, text = string(round(evr_percent[i], digits = 1), "%"), align = (:center, :bottom), fontsize = 18, font = "Libertinus Serif")
    end
    cum_percent = cumulative_evr .* 100
    lines!(ax, 1:n_comp, cum_percent, color = :gray35, linestyle = :dash, linewidth = 2.5)
    scatter!(ax, 1:n_comp, cum_percent, color = palette[2], markersize = 12, label = L"\text{Suma EVR}")
    axislegend(ax, position = :rc)
    return fig
end

function plot_pca_summary(
        dataset::AbstractMatrix{<:Real};
        tau::Union{Nothing, Real} = nothing,
        tau_tol::Real = 1.0e-8,
        tau_mode::Symbol = :nearest,
        n_components::Integer = 2,
        method::Symbol = :minmax,
        gamma::Real = 1.0,
    )
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    data = Matrix{Float64}(dataset)
    subtitle = L"\text{Wszystkie } \tau"
    if tau !== nothing
        τ = Float64(tau)
        τcol = data[:, 1]
        d = abs.(τcol .- τ)
        strict_mask = d .<= tau_tol
        if any(strict_mask)
            data = data[strict_mask, :]
            subtitle = L"\tau=%$(τ) \pm %$(tau_tol)"
        else
            i = argmin(d)
            near_mask = τcol .== τcol[i]
            data = data[near_mask, :]
            subtitle = L"\text{najbliższe } \tau=%$(τcol[i])"
        end
    end
    features = data[:, 2:3]
    @assert size(features, 1) > 1 "Need at least two samples in selected tau slice."
    pca_result = method === :minmax ? run_pca(features; n_components = n_components) : run_pca_kernel(features; n_components = n_components, gamma = gamma)
    transformed = pca_result.transformed
    evr = pca_result.explained_variance_ratio
    n_show = min(size(transformed, 2), 2)

    fig = Figure(size = (980, 420))
    ax_proj = Axis(fig[1, 1], xlabel = L"\text{PC1}", ylabel = L"\text{PC2}", title = L"\text{Projekcja PCA } (%$subtitle)")
    if n_show >= 2
        scatter!(ax_proj, transformed[:, 1], transformed[:, 2]; markersize = 4.5, color = (palette[1], 0.75))
    elseif n_show == 1
        scatter!(ax_proj, transformed[:, 1], zeros(size(transformed, 1)); markersize = 4.5, color = (palette[1], 0.75))
    end
    ax_evr = Axis(fig[1, 2], xlabel = L"\text{Główna składowa}", ylabel = L"\text{EVR}", limits = (0.5, max(length(evr), 1) + 0.5, 0, 1), title = L"\text{Współczynnik wariancji wyjaśnionej}")
    !isempty(evr) && barplot!(ax_evr, 1:length(evr), evr; color = palette[2])
    return fig
end

function animate_pca_evolution(
        dataset::AbstractMatrix{<:Real};
        filename::String = "pca_evolution.gif",
        fps::Integer = 15,
        n_components::Integer = 2,
        method::Symbol = :minmax,
        gamma::Real = 1.0,
        tau_tol::Real = 1.0e-8
    )
    set_publication_theme()
    palette = Makie.theme(:Palette).color[]
    taus = sort(unique(dataset[:, 1]))
    fig = Figure(size = (800, 600))
    title_obs = Observable(L"\text{Projekcja PCA } (\tau = %$(taus[1]))")
    ax = Axis(fig[1, 1], xlabel = L"\text{PC1}", ylabel = L"\text{PC2}", title = title_obs)
    pts_obs = Observable(Point2f[])
    scatter!(ax, pts_obs; markersize = 6.0, color = (palette[1], 0.75))

    record(fig, filename, taus; framerate = fps) do t
        val = round(t, digits = 3)
        title_obs[] = L"\text{Projekcja PCA } (\tau = %$(val))"
        mask = abs.(dataset[:, 1] .- t) .<= tau_tol
        data_slice = dataset[mask, :]
        features = data_slice[:, 2:3]
        if size(features, 1) > 1
            pca_result = method === :minmax ? run_pca(features; n_components = n_components) : run_pca_kernel(features; n_components = n_components, gamma = gamma)
            transformed = pca_result.transformed
            pts_obs[] = size(transformed, 2) >= 2 ? [Point2f(transformed[i, 1], transformed[i, 2]) for i in 1:size(transformed, 1)] : [Point2f(transformed[i, 1], 0.0) for i in 1:size(transformed, 1)]
            reset_limits!(ax)
        end
    end
    return filename
end

plot_pca_evr_over_time(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_pca_evr_over_time(to_2d_local_plots(dataset), args...; kwargs...)
plot_pca_bar_variance(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_pca_bar_variance(to_2d_local_plots(dataset), args...; kwargs...)
plot_pca_summary(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = plot_pca_summary(to_2d_local_plots(dataset), args...; kwargs...)
animate_pca_evolution(dataset::AbstractArray{<:Real, 3}, args...; kwargs...) = animate_pca_evolution(to_2d_local_plots(dataset), args...; kwargs...)
