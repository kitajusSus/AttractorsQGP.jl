

include("lib.jl")
include("pca.jl")

using .modHydroSim
using .PCAWorkflow
using Plots

function run_all_pca_and_plot()
    # Definicje zakresów temperatur (w MeV)
    temp_ranges = [
        (300.0, 600.0),
        (400.0, 700.0),
        (500.0, 800.0),
        (600.0, 1200.0),
    ]

    plots = []

    for (i, (Tmin, Tmax)) in enumerate(temp_ranges)
        settings = modHydroSim.SimSettings(T_range = (Tmin * modHydroSim.MeV, Tmax * modHydroSim.MeV))
        sim = modHydroSim.run_simulation(settings = settings)
        pca_results = PCAWorkflow.calc_pca(sim, features_indices = (1, 2), n_steps = 1000, n_components = 2)

        plt = PCAWorkflow.p_explained_variance(pca_results; prefix = "T=$(Tmin)-$(Tmax)")
        push!(plots, plt)
    end

    plot(plots..., layout = (2, 2), size=(1000,800), titlefontsize=10, guidefontsize=9, legendfontsize=8)
end

run_all_pca_and_plot()
