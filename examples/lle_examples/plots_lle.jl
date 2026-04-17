
function plot_examples_lle(X, Y, labels)
    fig = Figure(size = (1200, 600))
    sizeX = size(X)
    ax_3d = Axis3(fig[1, 1], title = "Oryginalna rozmaitość X $sizeX", azimuth = 0.22 * π)
    scatter!(ax_3d, X[1, :], X[2, :], X[3, :], color = labels, colormap = :jet, markersize = 8)
    # println(Y)
    ax_2d = Axis3(fig[1, 2], title = "Zredukowana przestrzeń Y $(size(Y))", azimuth = 0.22 * π)
    scatter!(ax_2d, Y[1, :], Y[2, :], Y[3,:],color = labels, colormap = :jet, markersize = 8)
    
    return fig
end




function plot_examples_lle_3d(X, Y, labels)
    fig = Figure(size = (1200, 600))
    
    dim_X = size(X, 1)
    dim_Y = size(Y, 1)

    if dim_X >= 3
        ax_X = Axis3(fig[1, 1], title = "Przestrzeń X ($(dim_X)D)", azimuth = 0.22 * π)
        scatter!(ax_X, X[1, :], X[2, :], X[3, :], color = labels, colormap = :jet, markersize = 8)
    elseif dim_X == 2
        ax_X = Axis(fig[1, 1], title = "Przestrzeń X (2D)")
        scatter!(ax_X, X[1, :], X[2, :], color = labels, colormap = :jet, markersize = 8)
    else
        ax_X = Axis(fig[1, 1], title = "Przestrzeń X (1D)")
        # Dla 1D dodajemy wektor zer, by narysować punkty na płaskiej osi
        scatter!(ax_X, X[1, :], zeros(size(X, 2)), color = labels, colormap = :jet, markersize = 8)
    end

    if dim_Y >= 3
        ax_Y = Axis3(fig[1, 2], title = "Przestrzeń Y ($(dim_Y)D)")
        scatter!(ax_Y, Y[1, :], Y[2, :], Y[3, :], color = labels, colormap = :jet, markersize = 8)
    elseif dim_Y == 2
        ax_Y = Axis(fig[1, 2], title = "Przestrzeń Y (2D)")
        scatter!(ax_Y, Y[1, :], Y[2, :], color = labels, colormap = :jet, markersize = 8)
    else
        ax_Y = Axis(fig[1, 2], title = "Przestrzeń Y (1D)")
        scatter!(ax_Y, Y[1, :], zeros(size(Y, 2)), color = labels, colormap = :jet, markersize = 8)
    end
    
    return fig
end
