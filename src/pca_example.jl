include("lib.jl")
include("pca.jl")
include("plt.jl")
using .modHydroSim
using .modPlots
using .modPCA
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using Random
using Distributions
function generate_matrix_example_plots()
    CairoMakie.activate!(type="png")
    modPlots.set_publication_theme()

    data = [
        1.0 2.0 8.0 6.0;
        2.0 4.0 6.0 4.0;
        3.0 6.0 4.0 8.0
    ]

    println("--- Analiza Przykładu Macierzowego ---")

    res, evr_raw, comps = modPCA.linear_pca(data, 4; mode=:center)

    evr = zeros(4)
    evr[1:length(evr_raw)] = evr_raw

    println("Wariancja wyjaśniona (EVR): ", round.(evr .* 100, digits=2), "%")

    fig_var = Figure(size=(1200, 600), fontsize=24)
    ax_var = Axis(fig_var[1, 1],
        title=L"\textbf{Analiza Wariancji}",
        xlabel=L"\text{Główna Składowa}",
        ylabel=L" EVR [\%]",
        xticks=(1:4, [L"PC_1", L"PC_2", L"PC_3", L"PC_4"]),
        limits=(0.5, 4.5, 0, 110),
        titlesize=38,
        xlabelsize=35,
        ylabelsize=35,
        xticklabelsize=24,
        yticklabelsize=24
    )

    barplot!(ax_var, 1:4, evr .* 100,
        color=ifelse.(evr .> 0.01, :dodgerblue, :gray80),
        strokewidth=2, strokecolor=:black
    )

    for i in 1:4
        val = evr[i] * 100
        text_str = i <= length(evr_raw) ? "$(round(val, digits=1))%" : "0.0%"
        text!(ax_var, i, val + 2, text=text_str,
            align=(:center, :bottom), fontsize=26, font=:bold)
    end

    save("plots/pca_matrix_example_variance.png", fig_var, px_per_unit=3)
    println("Wygenerowano wysokiej jakości: plots/pca_matrix_example_variance.png")


    fig_proj = Figure(size=(1600, 1300), fontsize=35)
    ax_proj = Axis(fig_proj[1, 1],
        title=L"\textbf{Przestrzeń Głównych Składowych}",
        xlabel=L"PC_1 \quad (\sim 80\% \text{ wariancji})",
        ylabel=L"PC_2 \quad (\sim 20\% \text{ wariancji})",
        aspect=DataAspect(),
        titlesize=36,
        xlabelsize=50,
        ylabelsize=50
    )

    scatter!(ax_proj, res[:, 1], res[:, 2],
        markersize=70, color=:white, strokecolor=:black, strokewidth=5, label=L"\text{Pomiary } t_i")

    text!(ax_proj, res[:, 1], res[:, 2],
        text=[L"t_1", L"t_2", L"t_3"],
        align=(:center, :center), fontsize=34, font=:bold)

    max_res = maximum(abs.(res))
    max_comps = maximum(abs.(comps))
    scale_factor = (max_res / max_comps) * 0.8

    vars = ["a", "b", "c", "d"]
    colors = [:firebrick, :firebrick, :firebrick, :forestgreen]

    for i in 1:4
        u, v = comps[i, 1] * scale_factor, comps[i, 2] * scale_factor

        arrows!(ax_proj, [0.0], [0.0], [u], [v],
            color=colors[i], linewidth=6, arrowsize=35)

        align_x = u > 0 ? :left : :right
        align_y = v > 0 ? :bottom : :top
        offset_x = u > 0 ? 0.05 : -0.05
        offset_y = v > 0 ? 0.05 : -0.05

        text!(ax_proj, u * 1.15, v * 1.15,
            text=L"%$((vars[i]))",
            color=colors[i], fontsize=48, font=:bold, align=(align_x, align_y))
    end

    hlines!(ax_proj, [0], color=(:gray, 0.5), linestyle=:dash, linewidth=2)
    vlines!(ax_proj, [0], color=(:gray, 0.5), linestyle=:dash, linewidth=2)

    elem_points = MarkerElement(color=:white, marker=:circle,
        strokecolor=:black, strokewidth=4, markersize=30)
    elem_corr = LineElement(color=:firebrick, linewidth=7)
    elem_indep = LineElement(color=:forestgreen, linewidth=7)

    axislegend(ax_proj,
        [elem_points, elem_corr, elem_indep],
        [L"\text{Pomiary } (t_1, t_2, t_3)", L"\text{Zmienne } a, b, c", L"\text{Zmienna } d"],
        position=:lt,
        framevisible=true,
        padding=(20, 20, 20, 20),
        labelsize=28,
        patchsize=(40, 20)
    )

    save("plots/pca_matrix_example_projection.png", fig_proj, px_per_unit=3)
    println("Wygenerowano wysokiej jakości: plots/pca_matrix_example_projection.png")
end

# generate_matrix_example_plots()


function kernel_pca_example()
    rng = Xoshiro(5)
    x = rand(rng, 100)

    X = sin.((1:100) .+ 0.5x)

    return X, x
end
