"""
    Script 05: Phase Space Grids and Pointwise LPCA Dimension Maps
    Generates:
    1. Standard phase space grids across proper times for MIS and HJSW (PDF only).
    2. Pointwise colored phase space grids where each point is colored by its estimated local dimension:
       - MIS (d in {1, 2}): physical (T, A) and dimensionless (w, A).
       - HJSW (d in {1, 2, 3}): (A, B) projection resolving the two-stage collapse.
    3. Standalone 3D and 2D single-slice phase space plots colored by local dimension (PDF only).
"""

using AttractorsQGP
using CairoMakie
using LaTeXStrings
using Printf

if !isdefined(Main, :PublicationLPCA)
    include(joinpath(@__DIR__, "..", "src", "PublicationLPCA.jl"))
end
using .PublicationLPCA

function run_phase_space_grids_experiment(;
    mis_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "mis_matched_seed5.h5"),
    hjsw_dataset_path::AbstractString = joinpath(@__DIR__, "..", "..", "datasets", "hjsw_lpca", "hjsw_matched_seed5.h5"),
    output_directory::AbstractString = joinpath(@__DIR__, "..", "plots", "phase_space")
)
    mkpath(output_directory)
    CairoMakie.activate!()

    println("=== Starting 05_phase_space_grids Experiment ===")

    # Standard 9-slice grid corresponding to paper figures
    grid_times = Float64[0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 1.0, 2.5, 5.0]
    k_eval = 24
    tol_eval = 0.01

    # -------------------------------------------------------------
    # A. Conformal MIS: Standard and Colored Phase Space Grids
    # -------------------------------------------------------------
    println("\n[1/3] Processing Conformal MIS Phase Space...")
    mis_raw = load_hydro_dataset(mis_dataset_path)
    mis_variants = prepare_dataset_variants(mis_raw, :mis)

    # A1. MIS Standard 2D Grids (PDF only)
    println("  - Standard 2D grid for MIS (T, A)...")
    fig_mis_TA = plot_phase_space_grid(mis_raw, grid_times, :T, :A)
    save(joinpath(output_directory, "phase_space_grid_mis_coords_T_A.pdf"), fig_mis_TA)

    println("  - Standard 2D grid for MIS (w, A)...")
    fig_mis_wA = plot_phase_space_grid(mis_raw, grid_times, :tauT, :A)
    save(joinpath(output_directory, "phase_space_grid_mis_coords_w_A.pdf"), fig_mis_wA)

    # A2. MIS Colored 2D Grids by Local Dimension (PDF only)
    println("  - Colored 2D grid for MIS (T, A) by local dimension d in {1, 2}...")
    fig_mis_colored_TA = plot_colored_phase_space_grid_2d(
        mis_variants.physical,
        grid_times;
        feature_indices = [2, 3],
        x_label = L"T\,[\mathrm{fm}^{-1}]",
        y_label = L"\mathcal{A}",
        k_neighbor = k_eval,
        tolerance = tol_eval,
        normalize_method = :max
    )
    save(joinpath(output_directory, "phase_space_colored_grid_mis_coords_T_A.pdf"), fig_mis_colored_TA)

    println("  - Colored 2D grid for MIS (w, A) by local dimension d in {1, 2}...")
    fig_mis_colored_wA = plot_colored_phase_space_grid_2d(
        mis_variants.dimensionless,
        grid_times;
        feature_indices = [2, 3],
        x_label = L"w = \tau T",
        y_label = L"\mathcal{A}",
        k_neighbor = k_eval,
        tolerance = tol_eval,
        normalize_method = :max
    )
    save(joinpath(output_directory, "phase_space_colored_grid_mis_coords_w_A.pdf"), fig_mis_colored_wA)

    # A3. MIS Standalone Slices at Key Times (tau = 0.25, 0.65, 2.50)
    for tau_snap in [0.25, 0.65, 2.5]
        slice_data_TA = compute_pointwise_dimensions(
            mis_variants.physical,
            tau_snap;
            feature_indices = [2, 3],
            k_neighbor = k_eval,
            tolerance = tol_eval,
            normalize_method = :max
        )
        fig_snap_TA = plot_colored_phase_space_slice_2d(
            slice_data_TA;
            x_col_idx = 1,
            y_col_idx = 2,
            x_label = L"T\,[\mathrm{fm}^{-1}]",
            y_label = L"\mathcal{A}"
        )
        save(joinpath(output_directory, @sprintf("phase_space_colored_slice_mis_coords_T_A_tau_%.2f.pdf", tau_snap)), fig_snap_TA)

        slice_data_wA = compute_pointwise_dimensions(
            mis_variants.dimensionless,
            tau_snap;
            feature_indices = [2, 3],
            k_neighbor = k_eval,
            tolerance = tol_eval,
            normalize_method = :max
        )
        fig_snap_wA = plot_colored_phase_space_slice_2d(
            slice_data_wA;
            x_col_idx = 1,
            y_col_idx = 2,
            x_label = L"w = \tau T",
            y_label = L"\mathcal{A}"
        )
        save(joinpath(output_directory, @sprintf("phase_space_colored_slice_mis_coords_w_A_tau_%.2f.pdf", tau_snap)), fig_snap_wA)
    end

    # -------------------------------------------------------------
    # B. HJSW Model: Standard and Colored Phase Space Grids
    # -------------------------------------------------------------
    println("\n[2/3] Processing HJSW Phase Space...")
    hjsw_raw = load_hydro_dataset(hjsw_dataset_path)
    hjsw_variants = prepare_dataset_variants(hjsw_raw, :hjsw)

    # B1. HJSW Standard 3D Grids (PDF only)
    println("  - Standard 3D grid for HJSW (T, A, B)...")
    fig_hjsw_TAB = plot_phase_space_grid_3d(
        hjsw_raw,
        grid_times,
        (L"T\,[\mathrm{MeV}]", (x, _) -> x[2]),
        :A,
        :B;
        azimuth = 1.3,
        elevation = 0.15,
        n_points_scatter = 2000
    )
    save(joinpath(output_directory, "phase_space_grid_hjsw_coords_T_A_B.pdf"), fig_hjsw_TAB)

    println("  - Standard 3D grid for HJSW (w, A, B)...")
    fig_hjsw_wAB = plot_phase_space_grid_3d(
        hjsw_raw,
        grid_times,
        :tauT,
        :A,
        :B;
        azimuth = 1.3,
        elevation = 0.15,
        n_points_scatter = 2000
    )
    save(joinpath(output_directory, "phase_space_grid_hjsw_coords_w_A_B.pdf"), fig_hjsw_wAB)

    # B2. HJSW Colored (A, B) Projection Grid by Local Dimension d in {1, 2, 3}
    println("  - Colored (A, B) projection grid for HJSW by local dimension d in {1, 2, 3}...")
    fig_hjsw_colored_AB = plot_colored_phase_space_grid_hjsw_projections(
        hjsw_raw,
        grid_times;
        k_neighbor = k_eval,
        tolerance = tol_eval,
        normalize_method = :max
    )
    save(joinpath(output_directory, "phase_space_colored_grid_hjsw_projections_A_B.pdf"), fig_hjsw_colored_AB)

    # B3. HJSW Colored (w, A) Projection Grid by Local Dimension d in {1, 2, 3}
    println("  - Colored (w, A) projection grid for HJSW by local dimension d in {1, 2, 3}...")
    fig_hjsw_colored_wA = plot_colored_phase_space_grid_2d(
        hjsw_variants.dimensionless,
        grid_times;
        feature_indices = [2, 3], # tauT, A
        x_label = L"w = \tau T",
        y_label = L"\mathcal{A}",
        k_neighbor = k_eval,
        tolerance = tol_eval,
        normalize_method = :max
    )
    save(joinpath(output_directory, "phase_space_colored_grid_hjsw_coords_w_A.pdf"), fig_hjsw_colored_wA)

    # B4. HJSW Standalone 3D Slices at Key Times (tau = 0.25, 0.65, 2.50)
    println("\n[3/3] Generating HJSW Standalone 3D Slices colored by dimension...")
    for tau_snap in [0.25, 0.65, 2.5]
        slice_data_3d = compute_pointwise_dimensions(
            hjsw_raw,
            tau_snap;
            feature_indices = [2, 3, 4],
            k_neighbor = k_eval,
            tolerance = tol_eval,
            normalize_method = :max
        )
        fig_hjsw_3d = plot_colored_phase_space_slice_hjsw_3d(
            slice_data_3d;
            x_label = L"T\,[\mathrm{MeV}]",
            y_label = L"\mathcal{A}",
            z_label = L"\mathcal{B}"
        )
        save(joinpath(output_directory, @sprintf("phase_space_colored_slice_hjsw_coords_T_A_B_tau_%.2f.pdf", tau_snap)), fig_hjsw_3d)

        slice_data_3d_w = compute_pointwise_dimensions(
            hjsw_variants.dimensionless,
            tau_snap;
            feature_indices = [2, 3, 4],
            k_neighbor = k_eval,
            tolerance = tol_eval,
            normalize_method = :max
        )
        fig_hjsw_3d_w = plot_colored_phase_space_slice_hjsw_3d(
            slice_data_3d_w;
            x_label = L"w = \tau T",
            y_label = L"\mathcal{A}",
            z_label = L"\mathcal{B}"
        )
        save(joinpath(output_directory, @sprintf("phase_space_colored_slice_hjsw_coords_w_A_B_tau_%.2f.pdf", tau_snap)), fig_hjsw_3d_w)
    end

    println("=== 05_phase_space_grids Completed Successfully! ===")
    return (
        fig_mis_TA = fig_mis_TA,
        fig_mis_wA = fig_mis_wA,
        fig_mis_colored_TA = fig_mis_colored_TA,
        fig_mis_colored_wA = fig_mis_colored_wA,
        fig_hjsw_TAB = fig_hjsw_TAB,
        fig_hjsw_wAB = fig_hjsw_wAB,
        fig_hjsw_colored_AB = fig_hjsw_colored_AB,
        fig_hjsw_colored_wA = fig_hjsw_colored_wA
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_phase_space_grids_experiment()
end
