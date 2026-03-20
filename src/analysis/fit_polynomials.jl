using Polynomials
using CairoMakie
using Random
using Statistics

function compute_polynomial_lle(data::AbstractMatrix{<:Real}, order::Int, tau::Float64)

    dane = get_tau_slice(data, tau)[2]

    x = dane[:, 1]
    y = dane[:, 2]
    poly_fit = Polynomials.fit(x, y, order)

    set_publication_theme()
    fig = Figure(size = (1200, 700))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Wielomian do układu } (\tau = %$tau)",
        xlabel = L"\text{Odwzorowanie LLE1 } (\xi_1)",
        ylabel = L"\text{Odwzorowanie LLE2 } (\xi_2)"
    )
    scatter!(ax, x, y)

    x_min, x_max = extrema(x)
    x_plot = range(x_min, x_max, length=600)
    y_plot = poly_fit.(x_plot)
    lines!(ax, x_plot, y_plot, color=:crimson)

    wspolczynniki = coeffs(poly_fit)
    a0 = round(wspolczynniki[1], digits=5)
    a1 = length(wspolczynniki) > 1 ? round(wspolczynniki[2], digits=5) : 0.0
    a2 = length(wspolczynniki) > 2 ? round(wspolczynniki[3], digits=5) : 0.0
    a3 = length(wspolczynniki) > 3 ? round(wspolczynniki[4], digits=6) : 0.0
    println("LLE2 ≈ $a3 ×(LLE1)^3 + $a2 ⋅ (LLE1)² + $a1 ⋅ (LLE1) + $a0")
    println("==================================================================")

    display(fig)
end
