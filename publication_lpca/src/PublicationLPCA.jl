module PublicationLPCA

using AttractorsQGP
using CairoMakie

CairoMakie.activate!()

for sym in [
        :load_hydro_dataset,
        :transform_to_dimensionless,
        :rescale_temperature,
        :create_mixed_scaled,
        :prepare_dataset_variants,
        :evaluate_k_pair,
        :evaluate_k_pair_pr,
        :evaluate_k_dependency,
        :evaluate_k_dependency_pr,
        :plot_k_dependency_bands,
        :plot_dims_k_dependency,
        :plot_pr_k_dependency,
        :plot_k_sweep_curve,
        :plot_d_vs_tau_multi_k,
        :plot_d_vs_tau_k_averaged,
        :compare_normalization_methods,
        :compare_normalization_methods_pr,
        :plot_normalization_multipanel,
        :plot_normalization_direct_overlay,
        :plot_pr_normalization_direct_overlay,
        :test_coordinate_invariance,
        :plot_coordinate_invariance,
        :plot_coordinate_invariance_single,
        :analyze_dimension_distribution,
        :analyze_tolerance_sensitivity,
        :sweep_k_dimensions_over_tau,
        :plot_dimension_distribution,
        :plot_dimension_mean_single,
        :plot_dimension_populations_single,
        :plot_tolerance_sensitivity,
        :PointwiseDimensionSlice,
        :compute_pointwise_dimensions,
        :compute_pointwise_pr,
        :compute_soft_weighted_dimension,
        :scan_soft_weighted_dimension,
        :plot_soft_weighted_dimension,
        :plot_colored_phase_space_slice_2d,
        :plot_colored_phase_space_grid_2d,
        :plot_colored_phase_space_grid_hjsw_projections,
        :plot_colored_phase_space_slice_hjsw_3d,
        :plot_colored_phase_space_grid_3d,
        :compute_parameterization_k_sweep,
        :plot_parameterization_focus_tripanel,
        :plot_parameterization_focus_2x2,
        :compute_local_pr_dimension,
        :scan_local_pr_dimension,
        :plot_local_pr_dimension,
        :plot_pr_phase_space_slice_2d,
        :plot_pr_phase_space_grid_2d,
        :plot_pr_phase_space_slice_3d,
        :plot_pr_phase_space_grid_3d,
    ]
    @eval export $sym
    @eval const $sym = AttractorsQGP.$sym
end

end
