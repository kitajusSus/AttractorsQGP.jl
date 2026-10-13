"""
    Script 13: Sample Size Comparison for Local PCA across K
    Generates a 3x3 multipanel comparison:
    - Columns: N = 5,000, N = 10,000, N = 20,000 points
    - Rows: tau = 0.25 fm/c, tau = 0.50 fm/c, tau = 1.00 fm/c
    - Fixed K nearest neighbors (e.g. K = 4, K = 8)
    - Discrete local dimension dims() in {1, 2}
    - Styled consistently with publication_lpca script 12
"""

using AttractorsQGP
using CairoMakie
using LaTeXStrings
using Printf
using Random
using Statistics

function run_sample_size_comparison(;
    k_eval::Integer = 8,
    mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
    output_dir::AbstractString = joinpath(@__DIR__, "..", "plots", "sample_size_comparison")
)
    mkpath(output_dir)
    CairoMakie.activate!()
    set_publication_theme()

    println("Loading MIS dataset from: ", mis_dataset_path)
    ds = load_hydro_dataset(mis_dataset_path)

    taus = [0.25, 0.50, 1.00]
    sample_sizes = [5_000, 10_000, 20_000]
    tol_eval = 0.01
    colormap_choice = lpca_style

    c1 = AttractorsQGP._get_dim_color(1.0, 2.0, colormap_choice)
    c2 = AttractorsQGP._get_dim_color(2.0, 2.0, colormap_choice)

    # 3x3 figure: width 1500, height 1350
    fig = Figure(size = (1500, 1350), figure_padding = (65, 55, 45, 45))
    colgap!(fig.layout, 45)
    rowgap!(fig.layout, 55)

    for (row_idx, tau) in enumerate(taus)
        tau_str = @sprintf("%.2f", tau)
        for (col_idx, N) in enumerate(sample_sizes)
            println("Processing tau = $tau_str fm/c, N = $N, K = $k_eval...")
            
            slice_data = compute_pointwise_dimensions(
                ds, tau;
                feature_indices = [2, 3],
                k_neighbor = k_eval,
                tolerance = tol_eval,
                normalize_method = :none,
                return_normalized = false,
                max_points = N,
                seed = 42 + row_idx * 10 + col_idx
            )

            x_vals = slice_data.coordinates[:, 1]
            y_vals = slice_data.coordinates[:, 2]
            d_vals = slice_data.dimensions

            c1_count = count(isapprox.(d_vals, 1.0; atol = 0.1))
            c2_count = count(isapprox.(d_vals, 2.0; atol = 0.1))
            pct1 = round(c1_count / N * 100, digits = 1)
            pct2 = round(c2_count / N * 100, digits = 1)
            println("  -> tau=$tau_str, N=$N: d=1: $c1_count ($pct1%), d=2: $c2_count ($pct2%), mean=$(round(mean(d_vals), digits=3))")

            sub_title = L"\tau = %$(tau_str)\,\mathrm{fm}/c,\; N = %$(N/1000)\mathrm{k}"

            ax = Axis(
                fig[row_idx, col_idx],
                title = sub_title,
                titlesize = 21,
                xlabel = row_idx == 3 ? L"T\,[\mathrm{fm}^{-1}]" : "",
                ylabel = col_idx == 1 ? L"\mathcal{A}" : "",
                xlabelsize = 20,
                ylabelsize = 22,
                xticklabelsize = 15,
                yticklabelsize = 15,
                xticks = LinearTicks(4),
                yticks = LinearTicks(4),
                xautolimitmargin = (0.04, 0.04),
                yautolimitmargin = (0.05, 0.05)
            )

            # Plot d=2 points then d=1 points
            mask_d2 = isapprox.(d_vals, 2.0; atol = 0.1)
            mask_d1 = isapprox.(d_vals, 1.0; atol = 0.1)

            scatter!(ax, x_vals[mask_d2], y_vals[mask_d2]; color = (c2, 0.70), markersize = 5.5, strokewidth = 0.2, strokecolor = (:black, 0.2), label = L"d = 2")
            scatter!(ax, x_vals[mask_d1], y_vals[mask_d1]; color = (c1, 0.85), markersize = 5.5, strokewidth = 0.2, strokecolor = (:black, 0.2), label = L"d = 1")

            # In corner (row 1, col 3), add legend
            if row_idx == 1 && col_idx == 3
                axislegend(ax, position = :rt, labelsize = 16)
            end
        end
    end

    pdf_path = joinpath(output_dir, "mis_phase_space_k$(k_eval)_sample_size_comparison.pdf")
    png_path = joinpath(output_dir, "mis_phase_space_k$(k_eval)_sample_size_comparison.png")

    println("Saving figure to PDF: ", pdf_path)
    save(pdf_path, fig)
    println("Saving figure to PNG: ", png_path)
    save(png_path, fig; px_per_unit = 2)

    println("Finished K = $k_eval comparison successfully!")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    k_arg = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
    run_sample_size_comparison(k_eval = k_arg)
end
