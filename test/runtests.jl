using Test
# include("../src/AttractorsQGP.jl")
using AttractorsQGP
using StaticArrays
using Random
using Lux
using ComponentArrays
using GLMakie

@testset "AttractorsQGP refactor API" begin
    model = BRSSSModel()
    u0 = [800.0, 0.5]
    tspan = (0.22, 0.5)

    sol = solve_hydro(model, u0, tspan; saveat = 0.02)
    @test !isempty(sol.t)
    τ = 0.3
    T, A = 5.0, 0.4
    p = model.params
    du = AttractorsQGP.rhs([T, A], model, τ)
    expected_dT = (T / τ) * (-1 / 3 + A / 18)
    expected_term_T = τ * T * (A + (p.lambda1 / (12 * p.eta_over_s)) * A^2)
    expected_term_A2 = (2 / 9) * p.tau_pi * A^2
    expected_dA = (1 / (p.tau_pi * τ)) * (8 * p.eta_over_s - expected_term_T - expected_term_A2)
    @test du[1] ≈ expected_dT atol = 1.0e-12
    @test du[2] ≈ expected_dA atol = 1.0e-12

    mis_model = MISModel()
    mis_attr = attractor_data(mis_model; tau_max = 0.03, krok = 0.01, w_min = 1.0e-4)
    expected_mis_A0 = 6 * sqrt(mis_model.params.eta_over_s / mis_model.params.tau_pi)
    @test all(mis_attr[:, 2] .== 1.0)
    @test mis_attr[1, 3] ≈ expected_mis_A0 atol = 1.0e-12


    ics_default_a = generate_initial_conditions(5; T_range = (700.0, 900.0), A_range = (-1.0, 1.0))
    ics_default_b = generate_initial_conditions(5; T_range = (700.0, 900.0), A_range = (-1.0, 1.0))
    @test ics_default_a == ics_default_b

    ics = ics_default_a
    @test eltype(ics) == SVector{2, Float64}
    @test all(ic -> 700.0 * FM_PER_MEV <= ic[1] <= 900.0 * FM_PER_MEV, ics)

    ics_fm = generate_initial_conditions(5; T_range = (3.0, 4.0), temperature_unit = :fm, rng = MersenneTwister(123))
    @test all(ic -> 3.0 <= ic[1] <= 4.0, ics_fm)

    sols = generate_trajectories(model, ics, tspan; saveat = 0.02, parallel = :serial)
    @test length(sols) == 5
    @test eltype(sols) != Any

    dataset = build_dataset(sols)
    @test size(dataset, 3) == 3

    dataset_mev = build_dataset(sols; temperature_unit = :MeV)
    @test size(dataset_mev) == size(dataset)
    @test dataset_mev[1, 1, 2] ≈ dataset[1, 1, 2] * MEV_PER_FM
    @test_throws ArgumentError build_dataset(sols; temperature_unit = :Kelvin)

    split_input = [
        0.2 1000.0 1.0
        0.3  900.0 0.5
        0.2  850.0 0.2
        0.3  800.0 0.1
    ]
    split_ranges = AttractorsQGP._split_trajectories(split_input)
    @test split_ranges == [1:2, 3:4]

    bad_solutions = [
        (t = [0.2, 0.21], u = [SVector(1000.0, 0.5), SVector(NaN, 0.4)]),
        (t = [0.2, 0.21], u = [SVector(950.0, -0.2), SVector(NaN, NaN)]),
    ]
    cleaned = build_dataset(bad_solutions)
    @test size(cleaned, 1) == 2
    @test !all(isfinite, cleaned)

    # lle = run_LLE(model, u0, tspan; saveat=0.02)
    # @test isfinite(lle)

    dim = estimate_dimension(reshape(dataset, :, size(dataset, 3))[:, 2:3])
    @test dim > 0
    flow = run_evolution_pca_workflow(
        model;
        n_points = 6,
        tspan = (0.22, 0.3),
        saveat = 0.02,
        parallel = :serial,
        feature_cols = [2, 3],
        rng = MersenneTwister(7),
    )
    @test length(flow.initial_conditions) == 6
    @test length(flow.solutions) == 6
    @test size(flow.dataset, 3) == 3
    @test size(flow.pca_over_time.explained_variance_ratio, 2) == 2
    flow_default_a = run_evolution_pca_workflow(
        model;
        n_points = 4,
        tspan = (0.22, 0.28),
        saveat = 0.02,
        parallel = :serial,
    )
    flow_default_b = run_evolution_pca_workflow(
        model;
        n_points = 4,
        tspan = (0.22, 0.28),
        saveat = 0.02,
        parallel = :serial,
    )
    @test flow_default_a.dataset == flow_default_b.dataset


    # PCA per-time should honor selected feature columns
    pca_dataset = zeros(3, 2, 3)
    pca_dataset[1, 1, :] = [0.2, 1.0, 10.0]
    pca_dataset[1, 2, :] = [0.3, 10.0, 1.0]
    pca_dataset[2, 1, :] = [0.2, 2.0, 20.0]
    pca_dataset[2, 2, :] = [0.3, 20.0, 2.0]
    pca_dataset[3, 1, :] = [0.2, 3.0, 30.0]
    pca_dataset[3, 2, :] = [0.3, 30.0, 3.0]

    tau_only = sort(Float64.(pca_dataset[1, :, 1]))
    idx_tau = get_tau_slice(tau_only, 0.2)
    @test length(idx_tau) == 1

    _, x_default = get_tau_slice(pca_dataset, 0.2)
    _, x_selected = get_tau_slice(pca_dataset, 0.2; feature_cols = [3])
    @test size(x_default, 2) == 2
    @test size(x_selected, 2) == 1
    @test x_selected[:, 1] == [10.0, 20.0, 30.0]

    pca_time = run_pca_per_time(pca_dataset; n_components = 1, feature_cols = [2])
    @test size(pca_time.explained_variance_ratio) == (2, 1)
    @test all(isfinite, pca_time.explained_variance_ratio)
    
    pca_time_symbols = run_pca_per_time(pca_dataset; n_components = 1, feature_cols = [:T, :tauT])
    @test size(pca_time_symbols.explained_variance_ratio) == (2, 1)
    @test all(isfinite, pca_time_symbols.explained_variance_ratio)

    one_tau = run_pca_for_tau(pca_dataset, 0.2; n_components = 1, feature_cols = [2, 3])
    @test one_tau.n_points == 3
    @test length(one_tau.pca_result.explained_variance_ratio) == 1


    shifted = [1.0 1.0; 2.0 2.0; 3.0 3.0]
    shifted_plus = shifted .+ 100.0
    evr_a = run_pca(shifted; n_components = 2).explained_variance_ratio
    evr_b = run_pca(shifted_plus; n_components = 2).explained_variance_ratio
    @test evr_a ≈ evr_b atol = 1.0e-12

    # PCA per-time should honor selected feature columns (duplicate block from original code, rewritten for consistency)
    tau_only2 = sort(Float64.(pca_dataset[1, :, 1]))
    idx_tau2 = get_tau_slice(tau_only2, 0.2)
    @test length(idx_tau2) == 1

    _, x_default = get_tau_slice(pca_dataset, 0.2)
    _, x_selected = get_tau_slice(pca_dataset, 0.2; feature_cols = [3])
    @test size(x_default, 2) == 2
    @test size(x_selected, 2) == 1
    @test x_selected[:, 1] == [10.0, 20.0, 30.0]

    pca_time = run_pca_per_time(pca_dataset; n_components = 1, feature_cols = [2])
    @test size(pca_time.explained_variance_ratio) == (2, 1)
    @test all(isfinite, pca_time.explained_variance_ratio)
    one_tau = run_pca_for_tau(pca_dataset, 0.2; n_components = 1, feature_cols = [2, 3])
    @test one_tau.n_points == 3
    @test length(one_tau.pca_result.explained_variance_ratio) == 1


    shifted = [1.0 1.0; 2.0 2.0; 3.0 3.0]
    shifted_plus = shifted .+ 100.0
    evr_a = run_pca(shifted; n_components = 2).explained_variance_ratio
    evr_b = run_pca(shifted_plus; n_components = 2).explained_variance_ratio
    @test evr_a ≈ evr_b atol = 1.0e-12


    mktempdir() do tmp
        csv_path = joinpath(tmp, "dataset.csv")
        h5_path = joinpath(tmp, "dataset.h5")
        jls_path = joinpath(tmp, "dataset.jls")

        save_dataset(csv_path, dataset)
        save_dataset(h5_path, dataset)
        save_dataset(jls_path, dataset)

        expected_size = (size(dataset, 1) * size(dataset, 2), size(dataset, 3))
        @test size(load_dataset(csv_path)) == expected_size
        @test size(load_dataset(h5_path)) == expected_size
        @test size(load_dataset(jls_path)) == expected_size

        csv_lower = joinpath(tmp, "dataset_lower.csv")
        write(csv_lower, "tau,t,a
0.22,1.0,2.0
0.23,1.1,2.1
")
        lower_loaded = load_dataset(csv_lower)
        @test size(lower_loaded) == (2, 3)
    end

    @testset "get_limits and plot_phase_space_grid limits" begin
        # sample dataset
        data = [
            0.2  1.0  -1.0
            0.2  2.0   1.0
            0.3  1.5  -0.5
            0.3  2.5   0.5
        ]
        
        # Test single axis limits
        xlims = get_limits(data, :T; times=(0.2, 0.3))
        @test xlims[1] < 1.0
        @test xlims[2] > 2.5
        
        ylims = get_limits(data, :A; times=(0.2, 0.3))
        @test ylims[1] < -1.0
        @test ylims[2] > 1.0

        # Test both axes limits
        lims = get_limits(data, :T, :A; times=(0.2, 0.3))
        @test lims[1] == xlims[1]
        @test lims[2] == xlims[2]
        @test lims[3] == ylims[1]
        @test lims[4] == ylims[2]

        attractor = [
            0.2 1.0 2.0
            0.4 1.0 4.0
        ]
        same_temperature_data = [
            0.2 1.0 0.0
            0.2 1.0 0.0
            0.4 1.0 0.0
            0.4 1.0 0.0
        ]
        attr_02 = get_attractor_line_for_frame(
            same_temperature_data, attractor, 0.2, :tauT, :A; n_points = 1
        )
        attr_04 = get_attractor_line_for_frame(
            same_temperature_data, attractor, 0.4, :tauT, :A; n_points = 1
        )
        @test attr_02.x[1] ≈ 0.2
        @test attr_04.x[1] ≈ 0.4
        @test attr_02.y[1] ≈ 2.0
        @test attr_04.y[1] ≈ 4.0

        # Test plot_phase_space_grid with attractor / Attractor Bool flags
        fig_attr_true = plot_phase_space_grid(same_temperature_data, [0.2, 0.4], :tauT, :A; attractor = true)
        @test any(p -> p isa GLMakie.Lines, fig_attr_true.content[1].scene.plots)

        fig_attr_false = plot_phase_space_grid(same_temperature_data, [0.2, 0.4], :tauT, :A; attractor = false)
        @test !any(p -> p isa GLMakie.Lines, fig_attr_false.content[1].scene.plots)

        fig_attr_cap_true = plot_phase_space_grid(same_temperature_data, [0.2, 0.4], :tauT, :A; Attractor = true)
        @test any(p -> p isa GLMakie.Lines, fig_attr_cap_true.content[1].scene.plots)

        fig_attr_cap_false = plot_phase_space_grid(same_temperature_data, [0.2, 0.4], :tauT, :A; Attractor = false)
        @test !any(p -> p isa GLMakie.Lines, fig_attr_cap_false.content[1].scene.plots)
    end

    @testset "scan_dimension_from_data" begin
        mock_data = [
            0.2  1.0  -1.0  0.5
            0.2  2.0   1.0  0.6
            0.2  3.0   0.0  0.4
            0.2  1.5  -0.5  0.5
            0.3  1.5  -0.5  1.0
            0.3  2.5   0.5  2.0
            0.3  2.0   0.0  1.5
            0.3  1.8  -0.2  1.2
        ]
        scan_all = scan_dimension_from_data(mock_data)
        @test scan_all.taus == [0.2, 0.3]
        @test length(scan_all.dimensions) == 2
        @test all(d -> 1.0 <= d <= 3.0, scan_all.dimensions)

        scan_subset = scan_dimension_from_data(mock_data; feature_cols=[2, 3])
        @test scan_subset.taus == [0.2, 0.3]
        @test length(scan_subset.dimensions) == 2
        @test all(d -> 1.0 <= d <= 2.0, scan_subset.dimensions)
    end

    @testset "Data-driven Neural Jacobians" begin
        mock_data = [
            0.1 10.0 100.0
            0.5 50.0 500.0
        ]
        norm_data, mn, mx = normalize_generic(mock_data)
        @test size(norm_data) == (2, 3)
        @test all(v -> -1.0 <= v <= 1.0, norm_data)
        @test mn ≈ [0.1, 10.0, 100.0]
        @test mx ≈ [0.5, 50.0, 500.0]

        nn = build_generic_network(3, 2; hidden_layers=2, hidden_size=10)
        rng = Random.MersenneTwister(42)
        ps_raw, st = Lux.setup(rng, nn)
        ps = ComponentArray{Float64}(ps_raw)

        mn_in = [0.1, 10.0, 100.0]
        mx_in = [0.5, 50.0, 500.0]
        mn_out = [1.0, 10.0]
        mx_out = [2.0, 20.0]

        τ = 0.3
        x0 = [30.0, 300.0]

        J = compute_data_jacobian(nn, ps, st, τ, x0, mn_in, mx_in, mn_out, mx_out)
        @test size(J) == (2, 2)
        @test all(isfinite, J)

        # 3. Test plot_pinn_deff_evolution
        wyniki_mock = [
            0.2 1.5
            0.3 1.2
            0.4 1.05
        ]
        fig = plot_pinn_deff_evolution(wyniki_mock)
        @test fig isa GLMakie.Figure

        # 4. Test plot_lle_spectrum_statistics_grid
        mock_model = BRSSSModel()
        mock_ics = generate_initial_conditions(15; T_range=(700.0, 900.0), A_range=(-1.0, 1.0))
        mock_sols = generate_trajectories(mock_model, mock_ics, (0.22, 0.3); saveat=0.04, parallel=:serial)
        mock_dataset = build_dataset(mock_sols)
        fig_grid = plot_lle_spectrum_statistics_grid(mock_dataset, [0.22, 0.26]; k_values = 2:2:4, ile_λ = 2)
        @test fig_grid isa GLMakie.Figure

        fig_grid_nested = plot_lle_spectrum_statistics_grid(mock_dataset, [0.22, 0.26]; k_values = [2:2:4], ile_λ = 2)
        @test fig_grid_nested isa GLMakie.Figure

        @test AttractorsQGP.flatten_k_values([5:1:10]) == [5, 6, 7, 8, 9, 10]
        @test AttractorsQGP.flatten_k_values(5:1:10) == 5:1:10
        @test AttractorsQGP.flatten_k_values([5, 6, 7]) == [5, 6, 7]

        # 5. Test plot_local_pca and data regularization methods
        X_test = [1.0 200.0; 2.0 400.0; 3.0 600.0]
        @test size(apply_normalization(X_test, :max)) == (3, 2)
        @test size(apply_normalization(X_test, :minmax)) == (3, 2)
        @test size(apply_normalization(X_test, :zscore)) == (3, 2)
        @test size(apply_normalization(X_test, :none)) == (3, 2)

        fig_lpca = plot_local_pca(mock_dataset; tablica_k = [2, 3], n_slices = 2, normalize = :minmax)
        @test fig_lpca isa GLMakie.Figure

        fig_lpca_regs = plot_local_pca_regularizations(mock_dataset; tablica_k = [2, 3], n_slices = 2, methods = [:max, :minmax, :zscore, :none])
        @test fig_lpca_regs isa GLMakie.Figure

        fig_ps_lpca_2d = plot_phase_space_lpca_dims(mock_dataset; liczba_sąsiadów = 3, wybrane_czasy = 0.22, zmienna_x = :T, zmienna_y = :A)
        @test fig_ps_lpca_2d isa GLMakie.Figure

        fig_ps_lpca_3d = plot_phase_space_lpca_dims(mock_dataset; liczba_sąsiadów = 3, wybrane_czasy = [0.22, 0.26], zmienna_x = :T, zmienna_y = :A, zmienna_z = :tau)
        @test fig_ps_lpca_3d isa GLMakie.Figure
    end

    @testset "HJSW Model ewolucja i PCA" begin
        model = HJSWModel()
        # @test model.omega_R ≈ 9.8
        # @test model.omega_I ≈ 8.629
        @test model.omega_R ≈ 9.80005
        @test model.omega_I ≈ 2.87631
        @test model.C_eta ≈ 0.0795775

        ics = generate_initial_conditions(model, 5; T_range=(800.0, 1000.0), A_range=(-1.0, 1.0), B_range=(-0.5, 0.5))
        @test eltype(ics) == SVector{3, Float64}
        @test length(ics) == 5

        sols = generate_trajectories(model, ics, (0.22, 0.3); saveat=0.02, parallel=:serial)
        @test length(sols) == 5

        dataset = build_dataset(sols)
        @test size(dataset, 3) == 4 # [tau, T, A, B]
        @test all(isfinite, dataset)

        # check that PCA and dimension scanner work on this 4-column dataset:
        wynik_dim = scan_dimension_from_data(dataset)
        @test length(wynik_dim.taus) > 0
        @test all(d -> 1.0 <= d <= 3.0, wynik_dim.dimensions)

        wynik_pca = run_pca_per_time(dataset; n_components=2)
        @test length(wynik_pca.taus) > 0
        @test size(wynik_pca.explained_variance_ratio, 2) == 2

        # test run_main with HJSWModel
        hjsw_res = run_main(model, n_points = 5, tspan = (0.2, 0.3), T_range = (200.0, 1400.0), A_range = (-1.0, 8.0), seed = 5)
        @test length(hjsw_res.solutions) == 5
        @test size(hjsw_res.dataset, 3) == 4
    end

    @testset "HJSWwModel ewolucja i run_main" begin
        model_w = HJSWwModel()
        @test model_w.omega_R ≈ 9.80005
        @test model_w.omega_I ≈ 2.87631

        ics = generate_initial_conditions(model_w, 5; w_range=(0.2, 2.0), A_range=(-1.0, 8.0), B_range=(-1.0, 1.0))
        @test eltype(ics) == SVector{3, Float64}
        sols = generate_trajectories(model_w, ics, (0.1, 1.0); saveat=0.05)
        @test length(sols) == 5

        ds = build_dataset(sols; has_temperature=false)
        @test size(ds, 3) == 4 # [t, w, A, B]

        res = run_main(model_w, n_points = 5, tspan = (0.1, 1.0), saveat = 0.05, seed = 5)
        @test length(res.solutions) == 5
        @test size(res.dataset, 3) == 4 # [t, w, A, B]
    end




    @testset "LPCA high-dimensional and normalization" begin
        # Test normalize_max function
        X_test = [1.0 -4.0 0.5; 2.0 2.0 1.0; -0.5 0.0 0.0]
        X_norm_test = normalize_max(X_test)
        @test maximum(abs.(X_norm_test), dims = 1) ≈ [1.0 1.0 1.0]
        @test X_norm_test[:, 1] ≈ [0.5, 1.0, -0.25]
        @test X_norm_test[:, 2] ≈ [-1.0, 0.5, 0.0]
        @test X_norm_test[:, 3] ≈ [0.5, 1.0, 0.0]

        # High-dimensional (>2D feature space) test for LPCA
        # 1 time column + 4 feature columns (d=4)
        mock_data_4d = randn(20, 5)
        mock_data_4d[1:10, 1] .= 0.2
        mock_data_4d[11:20, 1] .= 0.5
        
        tau_lpca_4d, mean_d_4d, std_d_4d = compute_lpca(mock_data_4d, 5, 2)
        @test length(tau_lpca_4d) == 2
        @test length(mean_d_4d) == 2
    end

    @testset "stacked PR dimension histograms" begin
        # Generate small mock dataset: tau, T, A
        mock_data = zeros(40, 3)
        mock_data[1:20, 1] .= 0.25
        mock_data[21:40, 1] .= 0.50
        mock_data[:, 2] .= range(1.0, 3.0, length = 40)
        mock_data[:, 3] .= range(-0.5, 2.0, length = 40)

        # Default stacked histogram colored by initial anisotropy A0
        fig_default = plot_pr_dimension_histogram(mock_data, 0.25; k_neighbor = 5, n_bins = 15, n_color_bins = 6, colormap = lpca_style)
        @test fig_default isa GLMakie.Figure

        # Explicit color_by = :initial_A and :A0
        fig_a0 = plot_pr_dimension_histogram(mock_data, 0.50; color_by = :A0, k_neighbor = 5, n_bins = 15, n_color_bins = 6, colormap = lpca_style)
        @test fig_a0 isa GLMakie.Figure

        # Instantaneous anisotropy
        fig_curr = plot_pr_dimension_histogram(mock_data, 0.50; color_by = :current_A, k_neighbor = 5, n_bins = 15, n_color_bins = 6, colormap = lpca_style)
        @test fig_curr isa GLMakie.Figure

        # Single unstacked histogram (color_by = nothing)
        fig_unstacked = plot_pr_dimension_histogram(mock_data, 0.25; k_neighbor = 5, n_bins = 15, color_by = nothing)
        @test fig_unstacked isa GLMakie.Figure

        # Grid stacked histogram colored by initial anisotropy
        fig_grid = plot_pr_dimension_histogram_grid(mock_data, [0.25, 0.50]; k_neighbor = 5, n_bins = 15, n_color_bins = 6, colormap = lpca_style)
        @test fig_grid isa GLMakie.Figure

        # Test automatic k_neighbor heuristic (:auto)
        fig_auto = plot_pr_dimension_histogram(mock_data, 0.25; k_neighbor = :auto, n_bins = 15)
        @test fig_auto isa GLMakie.Figure

        # 3D array input (e.g. simulation dataset [trajectories, timesteps, features])
        mock_3d_arr = zeros(20, 2, 3)
        mock_3d_arr[:, 1, 1] .= 0.25
        mock_3d_arr[:, 2, 1] .= 0.50
        mock_3d_arr[:, :, 2] .= rand(20, 2)
        mock_3d_arr[:, :, 3] .= rand(20, 2)
        fig_3d_single = plot_pr_dimension_histogram(mock_3d_arr, 0.25; k_neighbor = 5, n_bins = 15, color_by = :A)
        @test fig_3d_single isa GLMakie.Figure
        fig_3d_grid = plot_pr_dimension_histogram_grid(mock_3d_arr, [0.25, 0.50]; k_neighbor = 5, n_bins = 15, color_by = :A, n_color_bins = 8)
        @test fig_3d_grid isa GLMakie.Figure
    end

    @testset "phase space plots with zscore normalization and physical units" begin
        mock_3d = zeros(40, 4)
        mock_3d[1:20, 1] .= 0.25
        mock_3d[21:40, 1] .= 0.50
        mock_3d[:, 2] .= range(100.0, 500.0, length = 40) # T in MeV
        mock_3d[:, 3] .= range(-0.5, 2.0, length = 40)    # A
        mock_3d[:, 4] .= range(-0.8, 0.8, length = 40)    # B

        fig_2d = plot_colored_phase_space_grid_2d(mock_3d, [0.25, 0.50]; k_neighbor = 5, n_calc = 20, n_plot = 20)
        @test fig_2d isa GLMakie.Figure

        fig_3d = plot_colored_phase_space_grid_3d(mock_3d, [0.25, 0.50]; k_neighbor = 5, n_calc = 20, n_plot = 20)
        @test fig_3d isa GLMakie.Figure

        fig_pr_2d = plot_pr_phase_space_grid_2d(mock_3d, [0.25, 0.50]; k = 5, n_calc = 20, n_plot = 20)
        @test fig_pr_2d isa GLMakie.Figure

        fig_pr_3d = plot_pr_phase_space_grid_3d(mock_3d, [0.25, 0.50]; k = 5, n_calc = 20, n_plot = 20)
        @test fig_pr_3d isa GLMakie.Figure
    end

    @testset "optimal neighborhood size heuristics" begin
        # Test suggest_optimal_k for various sample sizes and ambient dimensions
        k_2k_d2 = suggest_optimal_k(2000, 2)
        @test 30 <= k_2k_d2 <= 60

        k_2k_d3 = suggest_optimal_k(2000, 3)
        @test 40 <= k_2k_d3 <= 70

        k_10k_d2 = suggest_optimal_k(10000, 2)
        @test 70 <= k_10k_d2 <= 110

        # Safety clamps for very small datasets
        k_small = suggest_optimal_k(15, 2)
        @test 2 <= k_small < 15

        # Matrix overload
        dummy_matrix = randn(100, 3)
        @test suggest_optimal_k(dummy_matrix) == suggest_optimal_k(100, 3)

        # Test find_optimal_k
        res_opt = find_optimal_k(dummy_matrix; k_candidates = [8, 16, 24, 32])
        @test res_opt.optimal_k in [8, 16, 24, 32]
        @test length(res_opt.candidates) == 4
        @test length(res_opt.variances) == 4
    end

    @testset "citation" begin
        io = IOBuffer()
        citation(io)
        out = String(take!(io))
        @test contains(out, "@misc{AttractorsQGP")
        @test contains(out, "Krzysztof Bezubik")
        @test contains(out, "Michał Spaliński")
        @test contains(out, "https://github.com/kitajusSus/Attractors-in-QGP")
    end
end

