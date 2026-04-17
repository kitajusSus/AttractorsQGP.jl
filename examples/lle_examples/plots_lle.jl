
function plot_examples_lle(X, Y, labels)
    fig = Figure(size = (1200, 600))
    
    ax_3d = Axis3(fig[1, 1], title = "Oryginalna rozmaitość X (3D)", azimuth = 0.22 * π)
    scatter!(ax_3d, X[1, :], X[2, :], X[3, :], color = labels, colormap = :jet, markersize = 8)

    ax_2d = Axis(fig[1, 2], title = "Zredukowana przestrzeń Y (2D)")
    scatter!(ax_2d, Y[1, :], Y[2, :], color = labels, colormap = :jet, markersize = 8)
    
    return fig
end

