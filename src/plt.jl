include("lib.jl")

module modPlots

using Plots
gr()
using LaTeXStrings
using Printf
using Dates
using DataFrames
using ..modHydroSim

export kadr,
    wykres,
    wykres_Aw,
    wykres_fazowy,
    plot_explained_variance_evolution,
    visualize_pca_static_grid,
    plot_loadings_evolution,
    test_animation

function kadr(simres::SimResult, t::Float64)
    states, valid_mask = TA(simres, t)
    Ts_MeV = states[1][valid_mask] ./ MeV
    As = states[2][valid_mask]
    τ_str = round(t, digits=2)
    p = plot(
        title="Phase space  (T, A) for  τ = $(τ_str) fm/c [$(simres.settings.theory)]",
        xlabel="Temperature T [MeV]",
        ylabel="Anisotropy A ", legend=false,
        xlims=(0, simres.settings.T_range[2] * 1.1 / MeV),
        ylims=(simres.settings.A_range[1] - 1, simres.settings.A_range[2] + 1),
    )
    scatter!(p, Ts_MeV, As, markersize=2, markerstrokewidth=0, alpha=0.7)

    mkpath(NAMESPACES.plots)
    filename = "$(NAMESPACES.label)_kadr_$(simres.settings.theory)_tau_$(τ_str).png"
    savefig(p, joinpath(PLOTS_DIR, filename))
    println("Saved plot to: $(joinpath(PLOTS_DIR, filename))")
end


function wykres(simres::SimResult;
    lw=1.5, size=(1200, 1080), color_min=-12.0)
    settings = simres.settings
    color_max = settings.A_range[2]

    p = plot(
        title="Ewolution A(τ) for Theory$(settings.theory). Settings: Arange=$(settings.A_range),Trange=$(settings.T_range), npoints=$(settings.n_points)",
        xlabel="Czas własny τ [fm/c]",
        ylabel="Anizotropia A",
        size=size,
        xlims=settings.tspan,
        ylims=(settings.A_range[1] - 1, settings.A_range[2] + 1),
        legend=false, colorbar=true,
        colorbar_title="Initial Anisotropy  A_0",
    )

    for sol in simres.solutions
        A0 = sol.u[1][2]
        local line_color
        if A0 < color_min
            line_color = :blue
        else
            line_color = :red
        end

        A_values = getindex.(sol.u, 2)
        plot!(p, sol.t, A_values, lw=lw, alpha=0.4, color=line_color)
    end

    mkpath(PLOTS_DIR)
    filename = "$(NAMESPACES.label)_ewolucja_A_tau_$(settings.theory).png"
    savefig(p, joinpath(PLOTS_DIR, filename))
    println("Saved plot to: $(joinpath(PLOTS_DIR, filename))")
end



function wykres_Aw(simres::SimResult;
    lw=1.5, size=(1200, 750), color_min=-12.0)
    settings = simres.settings

    p = plot(
        title="Evolution  A(w)for theory $(settings.theory)",
        xlabel=" w = \tau T",
        ylabel="Anisotropy A",
        size=size,
        ylims=(settings.A_range[1] - 1, settings.A_range[2] + 1),
        legend=false,
    )
    max_w = 0.0
    for sol in simres.solutions

        T_values = getindex.(sol.u, 1)
        valid_length = min(length(sol.t), length(T_values))
        w_values = sol.t[1:valid_length] .* T_values[1:valid_length]

        finite_w = filter(isfinite, w_values)
        if !isempty(finite_w)
            max_w = max(max_w, maximum(finite_w))
        end
    end
    plot!(p, xlims=(0, max_w * 1.05))

    for sol in simres.solutions
        A0 =
            sol.u[1][2]
        line_color = (A0 < color_min) ?
                     :blue : :red

        T_values = getindex.(sol.u, 1)
        A_values = getindex.(sol.u, 2)

        valid_length = min(length(sol.t), length(T_values), length(A_values))
        w_values = sol.t[1:valid_length] .* T_values[1:valid_length]

        plot!(
            p,
            w_values,
            A_values[1:valid_length], lw=lw,
            alpha=0.4,
            color=line_color,
        )
    end

    mkpath(PLOTS_DIR)
    filename = "ewolucja_A_w_$(settings.theory).png"
    savefig(p, joinpath(PLOTS_DIR, filename))
    println("Saved plot to: $(joinpath(PLOTS_DIR, filename))")
end

function wykres_fazowy(simres::SimResult; tau::Float64=0.5, markersize::Int=3)
    settings = simres.settings

    tau_0 = settings.tspan[1]

    tau0_T_points = Float64[]
    tau0_T_dot_points = Float64[]

    for sol in simres.solutions
        if tau < sol.t[1] ||
           tau > sol.t[end]
            continue
        end

        T_at_tau = sol(tau)[1]

        dt = 1e-3
        if tau + dt <= sol.t[end] && tau - dt >= sol.t[1]
            T_plus = sol(tau + dt)[1]
            T_minus = sol(tau - dt)[1]

            T_dot_at_tau = (T_plus - T_minus) / (2 * dt)
        else
            continue
        end

        push!(tau0_T_points, tau_0 * T_at_tau)
        push!(tau0_T_dot_points, tau_0^2 * T_dot_at_tau)
    end

    p = scatter(
        tau0_T_points,
        tau0_T_dot_points,
        title="Phase space (τ₀T,
            τ₀^2Ṫ) at τ = $(round(tau, digits=2)) fm/c [$(settings.theory)]",
        xlab=L"τ₀T",
        ylab=L"τ₀^2Ṫ",
        markersize=markersize,
        alpha=0.6,
        legend=false,
        color=:viridis
    )

    mkpath(NAMESPACES.plots)
    filename = "$(NAMESPACES.label)_wykres_fazowy_$(settings.theory)_tau_$(round(tau, digits=2)).png"

    display(p)
    println("Saved phase space plot to: $(joinpath(PLOTS_DIR, filename))")

    return p
end


function test_animation(
    df::DataFrame;
    output_gif::String="phase_space_animation.gif",
    fps::Int=20,
    xlims::Tuple{Float64,Float64}=nothing,
    ylims::Tuple{Float64,Float64}=nothing
)
    println("\n" * "="^60)
    println(" Tworzenie animacji z DataFrame")
    println("="^60)

    df_cols = Symbol.(names(df))
    unique_taus = sort(unique(df.tau))
    n_frames = length(unique_taus)
    println("Znaleziono $n_frames unikalnych kroków czasowych do animacji.")

    grouped_data = groupby(df, :Run_ID)

    println("Rozpoczynam generowanie animacji...")
    theme(:wong)
    println("Podaj wartości τ, dla których chcesz zatrzymać animację (np. 0.2 0.5 1.0):")
    input_line = readline()

    xlims = (minimum(df.T) * 1.02, maximum(df.T) * 1.02)
    if isempty(df.T)
        xlims = (0.0, 1.0)
    end

    ylims = (minimum(df.dTdtau) * 1.01, maximum(df.dTdtau) * 1.02)
    if isempty(df.dTdtau)
        ylims = (0.0, 1.0)
    end

    tau_snapshots = try
        parse.(Float64, split(input_line, keepempty=false))
    catch e
        println("Błędne wejście. Nie będą wyświetlane żadne dodatkowe wykresy. (Błąd: $e)")
        Float64[]
    end

    println("OK. Wykresy zostaną wyświetlone dla τ ≈ $tau_snapshots")

    tau_0 = unique_taus[1]
    anim = @animate for (i, τ_current) in enumerate(unique_taus)
        print("\rGenerowanie klatki $i / $n_frames (τ = $(round(τ_current, digits=2)))")


        head_data = filter(row -> row.tau == τ_current, df)
        T_points = head_data.T * tau_0
        dT_points = head_data.dTdtau * tau_0^2
        p = plot(
            xlab=L"\tau_{0} T",
            ylab=L"\tau_{0}^{2} \dot{T} \quad [dT/d\tau]",
            xlims=xlims,
            ylims=ylims,
            legend=false
        )



        if !isempty(head_data)
            scatter!(p, T_points, dT_points,
                markersize=2.5,
                markerstrokewidth=0,
                zcolor=head_data.T_0 ./ modHydroSim.MeV, # Konwertuj T_0 do MeV dla legendy
                c=:plasma,
                colorbar_title=L"\n T₀ [MeV]",
                label=""
            )
        end

        if any(τ_snap -> isapprox(τ_current, τ_snap, atol=0.02), tau_snapshots)

            println("\nZatrzymano na wybranej wartości τ ≈ $(round(τ_current, digits=2)).
                Kliknij Enter, by kontynuować...")

            display(p)
            png(p, joinpath(NAMESPACES.plots, "$(τ_current)_wykres.png"))
        end




    end
    println("\n\nZapisywanie animacji do pliku '$output_gif'...")
    mkpath(NAMESPACES.plots)
    output_path = joinpath(NAMESPACES.plots, output_gif)
    gif(anim, output_path, fps=fps)
    println("✅ Gotowe! Animacja: $output_path")
    println("="^60)

    return output_path
end

function plot_explained_variance_evolution(
    pca_results::Vector,
    source_file::String,
    feature_names::Vector{String},
    pca_method_params::Dict
)
    println("\n--- Generowanie wykresu ewolucji explained variance (EVR)... ---")

    if isempty(pca_results)
        println("Brak wyników do narysowania wykresu wariancji.")
        return
    end

    taus = [res.tau for res in pca_results]
    n_components = length(pca_results[1].explained_variance_ratio)

    method_info = "Metoda: $(pca_method_params[:method])"
    if pca_method_params[:method] == :kernel
        method_info *= " (gamma=$(round(pca_method_params[:gamma], digits=4)))"
    end
    settings_info = "Plik: $(basename(source_file)) |
   $method_info | Cechy: $(join(feature_names, ", "))"

    p = plot(
        title="Ewolucja wariancji wyjaśnionej przez komponenty PCA",
        plot_title=settings_info,
        xlabel="Czas τ [fm/c]",
        ylabel="Proporcja wyjaśnionej wariancji",
        legend=:best,
        ylim=(0, 1.1)
    )

    for i in 1:n_components
        variance_data = [res.explained_variance_ratio[i] for res in pca_results]
        plot!(p, taus, variance_data, label="PC $i", linewidth=2)
    end

    if n_components > 1
        cumulative_variance = [sum(res.explained_variance_ratio) for res in pca_results]
        plot!(p, taus, cumulative_variance, label="Suma", linestyle=:dash, color=:black)
    end

    hline!(p,
        [1.0], linestyle=:dot, color=:grey, label="", alpha=0.7)

    mkpath("plots")
    filename = "pca_explained_variance_evolution.png"
    savefig(p, joinpath("plots", filename))
    println("Saved plot to: $(joinpath("plots", filename))")
end

function visualize_pca_static_grid(
    pca_results::Vector,
    sim_result::modHydroSim.SimResult,
    num_plots::Int;
    source_file::String,
    feature_names::Vector{String},
    pca_method_params::Dict
)
    println("\n--- Generowanie siatki wykresów PCA ---")

    if isempty(pca_results)
        println("Brak wyników PCA do wizualizacji.")
        return
    end

    initial_states_raw = [sol.u[1] for sol in sim_result.solutions]
    all_initial_temps = [s[1] for s in initial_states_raw]

    method_info = "Metoda: $(pca_method_params[:method])"
    if pca_method_params[:method] == :kernel
        method_info *= " (gamma=$(round(pca_method_params[:gamma], digits=4)))"
    end
    settings_info = "Plik: $(basename(source_file)) | $method_info | Cechy: $(join(feature_names, ", "))"

    total_steps = length(pca_results)
    indices_to_plot = unique(round.(Int, range(1, stop=total_steps, length=num_plots)))

    plots_array = []

    all_pc1 = try
        vcat([res.transformed_data[:,
            1] for res in pca_results]...)
    catch e
        println("Błąd przy zbieraniu PC1: $e. Przerywanie wizualizacji.")
        return
    end

    all_pc2 = try
        vcat([res.transformed_data[:, 2] for res in pca_results]...)
    catch e
        println("Błąd przy zbieraniu PC2: $e.
     Przerywanie wizualizacji.")
        return
    end

    min_pc1, max_pc1 = minimum(all_pc1), maximum(all_pc1)
    min_pc2, max_pc2 = minimum(all_pc2), maximum(all_pc2)

    x_margin = (max_pc1 - min_pc1) * 0.05
    y_margin = (max_pc2 - min_pc2) * 0.05

    x_margin = x_margin > 0 ? x_margin : 1.0
    y_margin = y_margin > 0 ? y_margin : 1.0

    global_xlims = (min_pc1 - x_margin, max_pc1 + x_margin)
    global_ylims = (min_pc2 - y_margin, max_pc2 + y_margin)

    for idx in indices_to_plot
        result = pca_results[idx]

        valid_initial_temps = all_initial_temps[result.valid_mask]

        if length(valid_initial_temps) != size(result.transformed_data, 1)

            @warn "Niezgodność liczby punktów danych i temperatur dla tau=$(result.tau).
     Używam domyślnego koloru."
            marker_z_data = nothing
            color_data = :blue
        else
            marker_z_data = valid_initial_temps
            color_data = :plasma
        end

        current_tau = result.tau
        total_explained_var = sum(result.explained_variance_ratio) * 100

        p = scatter(
            result.transformed_data[:, 1],
            result.transformed_data[:, 2],
            marker_z=marker_z_data,
            title="τ = $(round(current_tau, digits=2)) fm/c (Var: $(round(total_explained_var, digits=1))%)", xlabel="PC 1",
            ylabel="PC 2",
            label="",
            xlims=global_xlims,
            ylims=global_ylims,
            markersize=4,
            alpha=0.8,
            color=color_data,
            colorbar=(marker_z_data !== nothing),
            legend=false
        )
        push!(plots_array, p)
    end

    if isempty(plots_array)
        println("Nie wygenerowano żadnych wykresów.")
        return
    end

    layout_cols = ceil(Int, sqrt(length(plots_array)))
    layout_rows = ceil(Int, length(plots_array) / layout_cols)


    println("✅ Tworzenie siatki $(layout_rows)x$(layout_cols) wykresów...")

    final_grid = plot(plots_array...,
        layout=(layout_rows, layout_cols),
        plot_title=settings_info,
        size=(350 * layout_cols, 350 * layout_rows)
    )

    mkpath("plots")
    filename = "pca_static_grid.png"
    savefig(final_grid, joinpath("plots", filename))
    println("Saved plot to: $(joinpath("plots", filename))")
end

function plot_loadings_evolution(
    pca_results::Vector,
    selected_feature_names::Vector{String};
    source_file::String,
    pca_method_params::Dict
)

    if isempty(pca_results) || pca_method_params[:method] == :kernel
        println("Wykres 'loadings' nie jest dostępny dla Kernel PCA lub brak wyników.")
        return
    end

    taus = [res.tau for res in pca_results]

    n_features_expected = length(selected_feature_names)
    n_features, n_components = size(pca_results[1].principal_components)

    if n_features != n_features_expected
        @warn "Niezgodność liczby cech w wynikach PCA ($n_features) i na liście nazw ($n_features_expected). Pomijanie wykresu ładunków."
        return
    end

    method_info = "Metoda: $(pca_method_params[:method])"
    main_title_info = "Plik: $(basename(source_file)) | $method_info"

    for i in 1:n_components
        p = plot(
            title="Ewolucja ładunków dla PC$i",
            plot_title=main_title_info,
            xlabel="Czas τ [fm/c]",
            ylabel="Wartość ładunku (Loading)",
            legend=:outerright,
            ylim=(-1.1, 1.1)
        )

        for j in 1:n_features
            loading_data = [res.principal_components[j, i] for res in pca_results if size(res.principal_components, 2) >= i]

            feature_name = selected_feature_names[j]

            if length(loading_data) == length(pca_results)
                plot!(p, taus, loading_data, label=feature_name, linewidth=2)
            else
                @warn "Brak danych o ładunkach dla PC$i, cechy $j w niektórych krokach czasowych."
            end
        end

        hline!(p, [0.0], linestyle=:dot, color=:grey, label="", alpha=0.7)

        mkpath("plots")
        filename = "pca_loadings_evolution_pc$(i).png"
        savefig(p, joinpath("plots", filename))
        println("Saved plot to: $(joinpath("plots", filename))")
    end

    println("✅ Wykresy ładunków gotowe. Uwaga: Nagłe 'odbicia lustrzane' (zmiany znaku) są normalne.")
end


end
