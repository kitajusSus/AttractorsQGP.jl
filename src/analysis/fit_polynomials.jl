using Polynomials
using CairoMakie
using Random
using Statistics
using Symbolics
using Latexify

function compute_polynomial_lle(data::AbstractMatrix{<:Real}, order::Int, tau::Float64)
    dane = get_tau_slice(data, tau)[2]
    x = dane[:, 1]
    y = dane[:, 2]

    poly_fit = Polynomials.fit(x, y, order)
    wspolczynniki = coeffs(poly_fit)

    @variables ξ₁ ξ₂

    wyrazenie_sym = 0.0
    for (i, c) in enumerate(wspolczynniki)
        stopien = i - 1
        wyrazenie_sym += round(c, sigdigits=4) * (ξ₁^stopien)
    end

    rownanie = ξ₂ ~ wyrazenie_sym
    kod_latex = latexify(rownanie)

    println("==================================================================")
    println("Odzyskane prawo fizyczne (Symbolics):")
    println(rownanie)
    println("\n latex:")
    println(kod_latex)

    set_publication_theme()
    fig = Figure(size = (1200, 850))

    ax = Axis(
        fig[1, 1],
        title = L"\text{Wyodrębniona dynamika atraktora } (\tau = %$tau\,\mathrm{fm}/c)",
        xlabel = L"\text{Odwzorowanie LLE1 } (\xi_1)",
        ylabel = L"\text{Odwzorowanie LLE2 } (\xi_2)"
    )

    scatter!(ax, x, y, color=(:midnightblue, 0.4), markersize=5)

    x_min, x_max = extrema(x)
    x_plot = range(x_min, x_max, length=600)
    y_plot = poly_fit.(x_plot)
    lines!(ax, x_plot, y_plot, color=:crimson, linewidth=4)

    # rownanie_czyste = replace(string(kod_latex), "\$" => "")
    Label(fig[2, 1], " ", fontsize=42, color=:crimson, tellwidth=false)

    display(fig)
    return fig
end
