using DifferentialEquations
using CairoMakie
using LinearAlgebra
using Statistics
using Random
using Dates
using LaTeXStrings

println(">>> Inicjalizacja środowiska holograficznego...")

# ===================================================================================
# CZĘŚĆ 1: FIZYKA I SYMULACJA (Zastępuje lib.jl)
# ===================================================================================

# --- Parametry Teorii Holograficznej ---
struct HoloParams
    C_tau::Float64   # Czas relaksacji hydrodynamicznej
    C_eta::Float64   # Lepkość ścinająca (KSS: 1/4pi)
    Delta::Float64   # Wymiar konforemny (Constituent Counting Rule: Delta=3 dla mezonów)
    Gamma_H::Float64 # Tłumienie przez horyzont (zależne od temperatury)
    M_eff::Float64   # Masa efektywna modu w 4D
    Coupling::Float64 # Siła sprzężenia modu z płynem (Backreaction)
end

# Domyślne wartości inspirowane Twoimi artykułami (Hard-Wall Model)
const PARAMS_HOLO = HoloParams(
    (2 - log(2)) / (2 * π),  # C_tau (z N=4 SYM)
    1 / (4 * π),             # C_eta (KSS bound)
    3.0,                     # Delta (Mezon qq_bar)
    2 * π,                   # Gamma_H (proporcjonalna do 2pi T)
    1.5,                     # M_eff
    10.0                     # Coupling
)

# --- Równania Różniczkowe (AdS/CFT inspired) ---
function ode_holo_standalone!(du, u, p::HoloParams, τ)
    # Zmienne stanu: u[1]=T, u[2]=A, u[3]=phi, u[4]=dphi
    T, A, phi, dphi = u

    # Zabezpieczenia numeryczne
    if T <= 1e-9 || !isfinite(T); du .= 0.0; return; end

    # 1. Temperatura (Bjorken Expansion)
    # dT/dtau = -T/tau * (1/3 - A/18)
    du[1] = -(T / τ) * (1.0/3.0 - A / 18.0)

    # 2. Anizotropia (Relaxation + Holographic Forcing)
    # Równanie typu MIS z członem źródłowym od pola phi
    tau_pi = p.C_tau / T
    NS_term = 8 * p.C_eta / (τ * T)

    # Backreaction: energia pola phi zaburza tensor energii-pędu
    Holo_Force = p.Coupling * phi^2

    du[2] = (1.0 / tau_pi) * (NS_term + Holo_Force - A)

    # 3. Dynamika Pola Holograficznego (Dual Field Decay)
    du[3] = dphi

    # Tłumienie zależy od Temperatury Horyzontu! (Gamma ~ T)
    # To jest kluczowa cecha holografii (Quasi-Normal Modes)
    damping = p.Gamma_H * T

    # Równanie falowe z masą i tłumieniem
    du[4] = -damping * dphi - (p.M_eff^2 + 1.0/τ^2) * phi
end

# --- Funkcja Uruchamiająca Symulację ---
function run_standalone_simulation(; n_points=50, tspan=(0.2, 5.0), seed=123)
    println(">>> Rozpoczynanie symulacji numerycznej ($n_points trajektorii)...")
    rng = Xoshiro(seed)

    # Generowanie warunków początkowych
    # T w [0.3, 0.6] GeV, A w [-2, 6]
    # Phi i dPhi losowe (mody wzbudzone)
    u0s = []
    for _ in 1:n_points
        push!(u0s, [
            rand(rng) * 0.3 + 0.3,   # T
            rand(rng) * 8.0 - 2.0,   # A
            (rand(rng) - 0.5) * 4.0, # phi (amplituda)
            (rand(rng) - 0.5) * 4.0  # dphi
        ])
    end

    solutions = []
    prob = ODEProblem(ode_holo_standalone!, u0s[1], tspan, PARAMS_HOLO)

    for (i, u0) in enumerate(u0s)
        # Rozwiązujemy dla każdego warunku początkowego
        _prob = remake(prob, u0=u0)
        sol = solve(_prob, Rodas5(), abstol=1e-7, reltol=1e-7)
        push!(solutions, sol)
    end

    return solutions
end

# ===================================================================================
# CZĘŚĆ 2: ANALIZA PCA (Wbudowana)
# ===================================================================================

function perform_pca_standalone(solutions, t_eval)
    # Zbieramy dane w konkretnym momencie t_eval
    # Macierz X: [n_samples x n_features]
    # Features: [T, A, phi, dphi]

    data_matrix = zeros(length(solutions), 4)

    valid_rows = Int[]

    for (i, sol) in enumerate(solutions)
        if t_eval >= sol.t[1] && t_eval <= sol.t[end]
            u = sol(t_eval)
            data_matrix[i, :] = u
            push!(valid_rows, i)
        end
    end

    X = data_matrix[valid_rows, :]

    # Standaryzacja (odejmij średnią)
    X_mean = mean(X, dims=1)
    X_centered = X .- X_mean

    # SVD
    U, S, Vt = svd(X_centered)

    # Wariancja wyjaśniona
    explained_variance = (S .^ 2) / (size(X, 1) - 1)
    total_var = sum(explained_variance)
    explained_variance_ratio = explained_variance ./ total_var

    # Transformacja do PC1, PC2
    transformed = X_centered * Vt'

    return (explained_variance_ratio, transformed, Vt, X_mean)
end

# ===================================================================================
# CZĘŚĆ 3: WIZUALIZACJA I TEORIA (Plotting)
# ===================================================================================

# Ustawienia stylu wykresów
set_theme!(Theme(fontsize=18, font="TeX Gyre Heros"))

function plot_analytical_theory()
    # Teoretyczne przewidywanie atraktora holograficznego
    # A(w) = A_hydro(w) + A_holo_mode(w)

    w_vals = range(0.2, 8.0, length=200)
    p = PARAMS_HOLO

    # Navier-Stokes
    A_hydro = [8 * p.C_eta / w for w in w_vals]

    # Holographic Mode Decay (Constituent Counting Rule)
    # Skalowanie: w^(-3/2 * Delta) * exp(-Gamma*w)
    scaling_pow = 1.5 * p.Delta
    A_holo = [5.0 * w^(-scaling_pow) * exp(-p.Gamma_H * 0.2 * w) * cos(4*w) for w in w_vals]

    A_total = A_hydro .+ A_holo

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1],
        title=L"Holographic Attractor Prediction ($\Delta=3$)",
        xlabel=L"w = \tau T", ylabel=L"\mathcal{A}")

    lines!(ax, w_vals, A_hydro, label="Hydro (Navier-Stokes)", color=:black, linestyle=:dash)
    lines!(ax, w_vals, A_total, label="Holographic Theory", color=:red, linewidth=3)

    band!(ax, w_vals, A_hydro, A_total, color=(:red, 0.2), label="Dual Field Contribution")

    axislegend(ax)
    return fig
end

function plot_simulation_results(solutions)
    fig = Figure(size=(1000, 600))

    # Panel 1: Zbieżność A(w)
    ax1 = Axis(fig[1, 1], title="Attractor Convergence", xlabel=L"w = \tau T", ylabel=L"\mathcal{A}")

    for sol in solutions
        ws = [t * u[1] for (t, u) in zip(sol.t, sol.u)]
        As = [u[2] for u in sol.u]
        lines!(ax1, ws, As, color=(:blue, 0.3), linewidth=1)
    end

    # Dodaj teorię
    w_range = range(0.1, 5.0, length=100)
    lines!(ax1, w_range, [8*PARAMS_HOLO.C_eta/w for w in w_range], color=:red, linewidth=2, label="Attractor", linestyle=:dash)

    # Panel 2: Ewolucja pola Phi (Dual Field)
    ax2 = Axis(fig[1, 2], title="Dual Field Decay", xlabel=L"\tau", ylabel=L"\Phi")
    for sol in solutions
        phis = [u[3] for u in sol.u]
        lines!(ax2, sol.t, phis, color=(:purple, 0.3))
    end

    return fig
end

function plot_pca_result(pca_res, t_snapshot)
    exp_var, trans_data, components, mean_val = pca_res

    fig = Figure(size=(900, 450))

    # Panel 1: Scree Plot
    ax1 = Axis(fig[1, 1], title="PCA Variance (t=$t_snapshot)", ylabel="Ratio")
    barplot!(ax1, 1:length(exp_var), exp_var, color=:teal)

    # Panel 2: PC1 vs PC2
    ax2 = Axis(fig[1, 2], title="Holographic Phase Space", xlabel="PC1", ylabel="PC2")
    scatter!(ax2, trans_data[:, 1], trans_data[:, 2], color=:black, alpha=0.5)

    return fig
end

# ===================================================================================
# CZĘŚĆ 4: GŁÓWNY PROGRAM (Execution)
# ===================================================================================

function main()
    timestamp = Dates.format(now(), "HHMMss")
    out_dir = joinpath(@__DIR__, "..", "plots", "holo_standalone_$timestamp")
    mkpath(out_dir)
    println(">>> Wyniki będą w: $out_dir")

    # 1. Rysuj Teorię
    println(">>> Generowanie wykresów teoretycznych...")
    fig_theory = plot_analytical_theory()
    save(joinpath(out_dir, "01_theory.png"), fig_theory)

    # 2. Uruchom Symulację
    sols = run_standalone_simulation(n_points=40, tspan=(0.2, 4.0))

    # 3. Rysuj Wyniki Symulacji
    println(">>> Wizualizacja trajektorii...")
    fig_sim = plot_simulation_results(sols)
    save(joinpath(out_dir, "02_simulation.png"), fig_sim)

    # 4. Wykonaj PCA w czasie t=1.0 fm/c
    println(">>> Analiza PCA...")
    pca_res = perform_pca_standalone(sols, 1.0)
    fig_pca = plot_pca_result(pca_res, 1.0)
    save(joinpath(out_dir, "03_pca_snapshot.png"), fig_pca)

    println(">>> GOTOWE. Sprawdź folder plots.")
    return out_dir
end

# Automatyczny start
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
