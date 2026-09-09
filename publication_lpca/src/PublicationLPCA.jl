module PublicationLPCA

using AttractorsQGP
using CairoMakie

# Activate CairoMakie for publication-quality headless vector (PDF) and raster (PNG) rendering
CairoMakie.activate!()

# Re-export all publication LPCA functions and structs directly from AttractorsQGP
for sym in [
    :load_hydro_dataset,
    :transform_to_dimensionless,
    :rescale_temperature,
    :create_mixed_scaled,
    :prepare_dataset_variants,
    :evaluate_k_pair,
    :evaluate_k_dependency,
    :plot_k_dependency_bands,
    :compare_normalization_methods,
    :plot_normalization_multipanel,
    :plot_normalization_direct_overlay,
    :test_coordinate_invariance,
    :plot_coordinate_invariance,
    :plot_coordinate_invariance_single,
    :analyze_dimension_distribution,
    :analyze_tolerance_sensitivity,
    :plot_dimension_distribution,
    :plot_dimension_mean_single,
    :plot_dimension_populations_single,
    :plot_tolerance_sensitivity,
    :PointwiseDimensionSlice,
    :compute_pointwise_dimensions,
    :compute_soft_weighted_dimension,
    :scan_soft_weighted_dimension,
    :plot_soft_weighted_dimension,
    :plot_colored_phase_space_slice_2d,
    :plot_colored_phase_space_grid_2d,
    :plot_colored_phase_space_grid_hjsw_projections,
    :plot_colored_phase_space_slice_hjsw_3d,
    :compute_parameterization_k_sweep,
    :plot_parameterization_focus_tripanel,
    :plot_parameterization_focus_2x2
]
    @eval export $sym
    @eval const $sym = AttractorsQGP.$sym
end

end

