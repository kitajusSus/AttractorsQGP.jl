"""
    marchenko_pastur_lpca.jl
    =============================================================================
    Weryfikacja rozkładu Marczenki-Pastura (Random Matrix Theory, RMT)
    oraz jego bezpośredniego związku z Local PCA (lPCA) i wymiarem d_PR
    w bibliotece AttractorsQGP.

    Stylistyka wykresów: `set_publication_theme(; cmap = lpca_style)`
    =============================================================================
"""

using LinearAlgebra
using Statistics
using Random
using Printf
using LaTeXStrings

# Obsługa CairoMakie / GLMakie
using CairoMakie
CairoMakie.activate!()

# Obsługa stylistyki publikacyjnej (AttractorsQGP lub wbudowana paleta)
using ColorSchemes, Colors

const LPCA_COLORS = [
    "#b00020",
    "#fb8500",
    "#2ec4b6",
    "#0096c7",
    "#0000FF",
]

const lpca_style_grad = cgrad(LPCA_COLORS)
const lpca_style = ColorScheme(Makie.resample_cmap(lpca_style_grad, 256))

function set_publication_theme(; cmap = lpca_style, fontsize = 20)
    set_theme!(
        Theme(
            fontsize = fontsize,
            figure_padding = 20,
            Axis = (
                backgroundcolor = RGBf(0.98, 0.98, 0.98),
                titlesize = fontsize + 4,
                xlabelsize = fontsize + 2,
                ylabelsize = fontsize + 2,
                xticklabelsize = fontsize - 4,
                yticklabelsize = fontsize - 4,
                xgridstyle = :dash,
                ygridstyle = :dash,
                xgridcolor = RGBAf(0.8, 0.8, 0.8, 0.7),
                ygridcolor = RGBAf(0.8, 0.8, 0.8, 0.7),
                spinewidth = 1.5,
                bottomspinecolor = :black,
                leftspinecolor = :black,
                topspinecolor = :black,
                rightspinecolor = :black,
                topspinevisible = true,
                rightspinevisible = true,
            ),
            Legend = (
                framevisible = true,
                framewidth = 1.2,
                framecolor = :black,
                backgroundcolor = RGBAf(1.0, 1.0, 1.0, 0.9),
                titlesize = fontsize - 2,
                labelsize = fontsize - 4,
            ),
            Lines = (linewidth = 2.5,),
            Scatter = (markersize = 10,),
            Palette = (
                color = Makie.resample_cmap(cmap, 10),
                patchcolor = Makie.resample_cmap(cmap, 10),
            )
        )
    )
end

# Inicjalizacja motywu publikacyjnego lpca_style
set_publication_theme(; cmap = lpca_style, fontsize = 18)

# =============================================================================
# 1. FUNKCJE ANALITYCZNE TEORII MACIERZY LOSOWYCH (MARCZENKO-PASTUR)
# =============================================================================

"""
    marchenko_pastur_limits(q::Real; σ::Real = 1.0) -> (λ_minus, λ_plus)

Oblicza granice teoretycznego pasma szumu (spectral bulk) rozkładu Marczenki-Pastura:
λ_± = σ² (1 ± √q)² dla q = N / T <= 1.
"""
function marchenko_pastur_limits(q::Real; σ::Real = 1.0)
    @assert q > 0 "Współczynnik q musi być dodatni (q = N/T > 0)"
    sq = sqrt(q)
    λ_minus = σ^2 * (1.0 - sq)^2
    λ_plus  = σ^2 * (1.0 + sq)^2
    return (λ_minus, λ_plus)
end

"""
    marchenko_pastur_pdf(λ::Real, q::Real; σ::Real = 1.0) -> Float64

Gęstość prawdopodobieństwa rozkładu Marczenki-Pastura ρ(λ).
"""
function marchenko_pastur_pdf(λ::Real, q::Real; σ::Real = 1.0)
    λ_minus, λ_plus = marchenko_pastur_limits(q; σ = σ)
    if λ < λ_minus || λ > λ_plus
        return 0.0
    end
    num = sqrt((λ_plus - λ) * (λ - λ_minus))
    den = 2.0 * π * (σ^2) * q * λ
    return num / den
end

"""
    participation_ratio_val(vals::AbstractVector{<:Real}) -> Float64

Wskaźnik Participation Ratio:
d_PR = (∑ λᵢ)² / ∑ λᵢ²
"""
function participation_ratio_val(vals::AbstractVector{<:Real})
    v = filter(x -> x > 1e-15, vals)
    isempty(v) && return 0.0
    s1 = sum(v)
    s2 = sum(v .^ 2)
    return s2 > 0 ? (s1^2) / s2 : 0.0
end

# =============================================================================
# 2. GENEROWANIE DANYCH EKSPERYMENTALNYCH
# =============================================================================

"""
    simulate_wishart_spectrum(N::Int, T::Int; σ::Real = 1.0, n_trials::Int = 100, signal_strength::Real = 0.0)

Generuje empiryczne wartości własne macierzy kowariancji C = (1/T) X' X
dla macierzy losowej X (T próbek, N cech) z czystym szumem Gaussa lub dodatkowym sygnałem fizycznym (Spike).
"""
function simulate_wishart_spectrum(
        N::Int,
        T::Int;
        σ::Real = 1.0,
        n_trials::Int = 100,
        signal_strength::Real = 0.0
    )
    all_evals = Float64[]
    sizehint!(all_evals, N * n_trials)

    for _ in 1:n_trials
        # Czysty szum Gaussa N(0, σ²)
        X = σ .* randn(T, N)

        # Opcjonalny sygnał (Spike model): dodajemy silny mod wzdłuż jednego kierunku
        if signal_strength > 0.0
            # v to losowy wektor jednostkowy w N wymiarach
            v = randn(N)
            v ./= norm(v)
            # Sygnał o wariancji signal_strength
            s = sqrt(signal_strength) .* randn(T)
            X .+= s * transpose(v)
        end

        # Empiryczna macierz kowariancji
        # Centrowanie kolumn
        Xc = X .- mean(X, dims = 1)
        C = (transpose(Xc) * Xc) ./ (T - 1)

        ev = eigvals(Symmetric(C))
        append!(all_evals, ev)
    end

    return sort(all_evals)
end

"""
    simulate_lpca_noise_dpr(D::Int, Ks::AbstractVector{Int}; n_eval::Int = 1000)

Symuluje działanie Local PCA na czystym szumie izotropowym w D wymiarach:
bada, jak mała liczba próbek K sztucznie zaniża wymiar partycypacji d_PR
z powodu rozpychania wartości własnych (eigenvalue repulsion).
"""
function simulate_lpca_noise_dpr(D::Int, Ks::AbstractVector{Int}; n_eval::Int = 1000)
    mean_dprs = zeros(Float64, length(Ks))
    std_dprs  = zeros(Float64, length(Ks))
    mean_ratio = zeros(Float64, length(Ks))

    for (idx, k) in enumerate(Ks)
        dpr_vals = zeros(Float64, n_eval)
        ratios   = zeros(Float64, n_eval)

        for i in 1:n_eval
            # Lokalna chmura k sąsiadów w D wymiarach z czystego szumu izotropowego
            Y = randn(k, D)
            Y_c = Y .- mean(Y, dims = 1)
            C = (transpose(Y_c) * Y_c) ./ (k - 1)

            ev = sort(eigvals(Symmetric(C)), rev = true)
            dpr_vals[i] = participation_ratio_val(ev)
            ratios[i] = ev[1] / max(ev[end], 1e-15)
        end

        mean_dprs[idx] = mean(dpr_vals)
        std_dprs[idx]  = std(dpr_vals)
        mean_ratio[idx] = mean(ratios)
    end

    return (mean_dprs = mean_dprs, std_dprs = std_dprs, mean_ratio = mean_ratio)
end

# =============================================================================
# 3. INTERAKTYWNE FUNKCJE DO WYSWIETLANIA WYKRESOW W REPL
# =============================================================================

"""
    plot_marchenko_pastur(; N=10, T=252, n_trials=80, signal_strength=0.0, save_path=nothing)

Wykres empirycznego histogramu wartości własnych na tle analitycznego rozkładu Marczenki-Pastura.
Domyślne parametry (N=10, T=252) odpowiadają przykładowi z cytatu!
"""
function plot_marchenko_pastur(;
        N::Int = 10,
        T::Int = 252,
        σ::Real = 1.0,
        n_trials::Int = 80,
        signal_strength::Real = 0.0,
        save_path = nothing
    )
    q = N / T
    λ_minus, λ_plus = marchenko_pastur_limits(q; σ = σ)
    ratio = λ_plus / λ_minus

    println("\n" * "="^65)
    println(" ROZKŁAD MARCZENKI-PASTURA (Parametry)")
    println("="^65)
    @printf("  Wymiar cech N        : %d\n", N)
    @printf("  Liczba próbek T       : %d\n", T)
    @printf("  Współczynnik q = N/T  : %.5f\n", q)
    @printf("  Pierwiastek √q        : %.5f\n", sqrt(q))
    @printf("  Dolna granica λ₋      : %.4f\n", λ_minus)
    @printf("  Górna granica λ₊      : %.4f\n", λ_plus)
    @printf("  Stosunek λ₊ / λ₋      : %.3f\n", ratio)
    if signal_strength > 0.0
        bbp_threshold = σ^2 * sqrt(q)
        @printf("  Sygnał (Spike θ)      : %.3f (Próg detekcji BBP = %.3f)\n", signal_strength, bbp_threshold)
        if signal_strength > bbp_threshold
            λ_spike_theor = (1.0 + signal_strength) * (1.0 + q / signal_strength)
            @printf("  -> Sygnał WYSTAJE ponad szum! Oczekiwane λ₁ ≈ %.3f > λ₊\n", λ_spike_theor)
        else
            println("  -> Sygnał SCHOWANY w pasmie szumu (θ <= √q). Niewykrywalny!")
        end
    end
    println("="^65)

    # Symulacja
    evals = simulate_wishart_spectrum(N, T; σ = σ, n_trials = n_trials, signal_strength = signal_strength)

    # Rysowanie w Makie
    fig = Figure(size = (900, 600))
    ax = Axis(
        fig[1, 1],
        title = L"\text{Rozkład Marczenki-Pastura: } N=%$(N),\, T=%$(T),\, q \approx %$(round(q, digits=4))",
        xlabel = L"\text{Wartość własna } \lambda",
        ylabel = L"\text{Gęstość prawdopodobieństwa } \rho(\lambda)",
    )

    # Histogram empiryczny
    hist!(
        ax,
        evals,
        bins = max(30, min(80, N * 4)),
        normalization = :pdf,
        color = (LPCA_COLORS[4], 0.65),
        strokewidth = 1,
        strokecolor = LPCA_COLORS[5],
        label = L"\text{Empiryczne } \lambda_i \text{ (Monte Carlo, %$(n_trials) powtórzeń)}"
    )

    # Krzywa analityczna MP
    λ_grid = range(λ_minus * 0.999, λ_plus * 1.001, length = 400)
    pdf_vals = [marchenko_pastur_pdf(λ, q; σ = σ) for λ in λ_grid]
    lines!(
        ax,
        λ_grid,
        pdf_vals,
        color = LPCA_COLORS[1],
        linewidth = 3.5,
        label = L"\text{Teoria: } \rho_{\mathrm{MP}}(\lambda)"
    )

    # Granice szumu λ_minus i λ_plus
    vlines!(
        ax,
        [λ_minus],
        color = LPCA_COLORS[2],
        linestyle = :dash,
        linewidth = 2.5,
        label = L"\lambda_- = %$(round(λ_minus, digits=3))"
    )
    vlines!(
        ax,
        [λ_plus],
        color = LPCA_COLORS[1],
        linestyle = :dash,
        linewidth = 2.5,
        label = L"\lambda_+ = %$(round(λ_plus, digits=3))"
    )

    # Pasmo szumu (zacienienie)
    vspan!(ax, λ_minus, λ_plus, color = (LPCA_COLORS[3], 0.12))

    axislegend(ax, position = :rt)

    if save_path !== nothing
        save(save_path, fig)
        println("Zapisano wykres do: $save_path")
    end

    display(fig)
    return fig
end

"""
    plot_repulsion_vs_q(; q_range=0.01:0.005:0.95, save_path=nothing)

Wykres rozpychania wartości własnych (eigenvalue repulsion) oraz stosunku λ₊ / λ₋
w funkcji stosunku wymiaru do próbek q = N/T (lub D/K w lPCA).
Zaznacza kluczowy punkt projektu AttractorsQGP: K = 12 D (q = 1/12 ≈ 0.0833).
"""
function plot_repulsion_vs_q(;
        q_range = 0.01:0.005:0.95,
        save_path = nothing
    )
    qs = collect(q_range)
    lp = [(1.0 + sqrt(q))^2 for q in qs]
    lm = [(1.0 - sqrt(q))^2 for q in qs]
    ratios = lp ./ lm

    fig = Figure(size = (1100, 500))

    # Panel lewy: granice pasma szumu
    ax1 = Axis(
        fig[1, 1],
        title = L"\text{Pasmo szumu MP: } \lambda_\pm(q) = (1 \pm \sqrt{q})^2",
        xlabel = L"q = N/T \quad (D/K \text{ w lPCA})",
        ylabel = L"\text{Wartości własne } \lambda",
    )

    lines!(ax1, qs, lp, color = LPCA_COLORS[1], linewidth = 3, label = L"\lambda_+(q) = (1+\sqrt{q})^2")
    lines!(ax1, qs, lm, color = LPCA_COLORS[4], linewidth = 3, label = L"\lambda_-(q) = (1-\sqrt{q})^2")
    hlines!(ax1, [1.0], color = :black, linestyle = :dot, linewidth = 1.5, label = L"\lambda_{\text{teor}} = 1.0 \text{ (idealne)}")

    # Zaznacz punkt K = 12 D -> q = 1/12
    q_12 = 1.0 / 12.0
    lp_12 = (1.0 + sqrt(q_12))^2
    lm_12 = (1.0 - sqrt(q_12))^2
    vlines!(ax1, [q_12], color = LPCA_COLORS[2], linestyle = :dash, linewidth = 2)
    scatter!(ax1, [q_12, q_12], [lm_12, lp_12], color = LPCA_COLORS[2], markersize = 12)

    text!(
        ax1,
        q_12 + 0.02,
        1.1,
        text = L"K \ge 12D \implies q \le 1/12",
        color = LPCA_COLORS[2],
        fontsize = 15
    )

    axislegend(ax1, position = :lt)

    # Panel prawy: Stosunek skrajnych wartości własnych (repulsion factor)
    ax2 = Axis(
        fig[1, 2],
        title = L"\text{Współczynnik rozpychania: } \lambda_+ / \lambda_-",
        xlabel = L"q = N/T \quad (D/K \text{ w lPCA})",
        ylabel = L"\lambda_+ / \lambda_-",
        yscale = log10
    )

    lines!(ax2, qs, ratios, color = LPCA_COLORS[1], linewidth = 3, label = L"\frac{\lambda_+}{\lambda_-} = \left(\frac{1+\sqrt{q}}{1-\sqrt{q}}\right)^2")
    vlines!(ax2, [q_12], color = LPCA_COLORS[2], linestyle = :dash, linewidth = 2)
    scatter!(ax2, [q_12], [lp_12 / lm_12], color = LPCA_COLORS[2], markersize = 12)

    # Zaznacz K = 4D (q = 0.25)
    q_4 = 0.25
    r_4 = (1.0 + 0.5)^2 / (1.0 - 0.5)^2
    scatter!(ax2, [q_4], [r_4], color = LPCA_COLORS[1], markersize = 12)
    text!(
        ax2,
        q_4 + 0.02,
        r_4,
        text = L"K=4D \implies \frac{\lambda_+}{\lambda_-}=9.0",
        color = LPCA_COLORS[1],
        fontsize = 15
    )

    text!(
        ax2,
        q_12 + 0.02,
        lp_12 / lm_12 * 1.3,
        text = L"K=12D \implies \frac{\lambda_+}{\lambda_-} \approx 3.3",
        color = LPCA_COLORS[2],
        fontsize = 15
    )

    axislegend(ax2, position = :lt)

    if save_path !== nothing
        save(save_path, fig)
        println("Zapisano wykres do: $save_path")
    end

    display(fig)
    return fig
end

"""
    plot_lpca_noise_artifact(; D=3, Ks=[4, 6, 8, 12, 16, 24, 36, 48, 64, 96, 128], n_eval=600, save_path=nothing)

Kluczowy eksperyment dla lPCA:
Badanie pozornego spadku wymiaru partycypacji d_PR dla CZYSTEGO SZUMU IZOTROPOWEGO w D wymiarach.
Pokazuje, że dla małego K algorytm estymuje d_PR << D pomimo braku jakiegokolwiek atraktora!
Dopiero K >= 12*D stabilizuje d_PR blisko rzeczywistego wymiaru D.
"""
function plot_lpca_noise_artifact(;
        D::Int = 3,
        Ks::AbstractVector{Int} = [4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48, 64, 80, 96, 128],
        n_eval::Int = 600,
        save_path = nothing
    )
    println("\nSymulacja Local PCA na czystym szumie izotropowym w D = $D wymiarach...")
    res = simulate_lpca_noise_dpr(D, Ks; n_eval = n_eval)

    k_min_rmt = 12 * D

    fig = Figure(size = (1100, 520))

    # Lewy panel: Wymiar d_PR vs K
    ax1 = Axis(
        fig[1, 1],
        title = L"\text{Artefakt skończonej próby: Pozorny spadek } d_{\mathrm{PR}} \text{ (Szum } D=%$(D)\text{)}",
        xlabel = L"\text{Liczba sąsiadów } K",
        ylabel = L"\text{Wymiar partycypacji } d_{\mathrm{PR}}",
    )

    # Prawdziwy wymiar tła D
    hlines!(ax1, [Float64(D)], color = :black, linestyle = :dot, linewidth = 2, label = L"\text{Prawdziwy wymiar: } D=%$(D)")

    # Wstęga odchylenia standardowego ± 1σ
    band!(
        ax1,
        Ks,
        res.mean_dprs .- res.std_dprs,
        res.mean_dprs .+ res.std_dprs,
        color = (LPCA_COLORS[4], 0.25),
        label = L"\pm 1\sigma \text{ rozrzut szumu}"
    )

    # Średnia empiryczna d_PR
    lines!(ax1, Ks, res.mean_dprs, color = LPCA_COLORS[5], linewidth = 3, label = L"\langle d_{\mathrm{PR}}(K) \rangle \text{ (izotropowy szum)}")
    scatter!(ax1, Ks, res.mean_dprs, color = LPCA_COLORS[5], markersize = 9)

    # Pionowa linia reguły K >= 12*D z kodu lpca.jl
    vlines!(ax1, [k_min_rmt], color = LPCA_COLORS[1], linestyle = :dash, linewidth = 2.5, label = L"K_{\min} = 12 D = %$(k_min_rmt) \text{ (AttractorsQGP)}")

    # Oznaczenie strefy artefaktu szumu
    vspan!(ax1, Ks[1], k_min_rmt, color = (LPCA_COLORS[1], 0.08))
    text!(
        ax1,
        k_min_rmt / 2.3,
        D - 0.7,
        text = "STREFA ARTEFAKTU\nSZUMU RMT (K < 12D)",
        color = LPCA_COLORS[1],
        fontsize = 13,
        align = (:center, :center)
    )

    axislegend(ax1, position = :rb)

    # Prawy panel: Stosunek λ_max / λ_min vs Teoria MP
    ax2 = Axis(
        fig[1, 2],
        title = L"\text{Rozpychanie wartości własnych: } \langle \lambda_1 / \lambda_D \rangle",
        xlabel = L"\text{Liczba sąsiadów } K",
        ylabel = L"\lambda_{\max} / \lambda_{\min}",
    )

    # Teoria MP dla q = D / K
    k_theor = range(Ks[1], Ks[end], length = 300)
    ratio_theor = [begin
        q = D / (k - 1)
        if q < 1.0
            ((1.0 + sqrt(q)) / (1.0 - sqrt(q)))^2
        else
            NaN
        end
    end for k in k_theor]

    lines!(ax2, k_theor, ratio_theor, color = LPCA_COLORS[1], linewidth = 3, label = L"\text{Teoria MP: } \left(\frac{1+\sqrt{D/K}}{1-\sqrt{D/K}}\right)^2")
    scatter!(ax2, Ks, res.mean_ratio, color = LPCA_COLORS[4], markersize = 10, label = L"\text{Empiryczne } \langle \lambda_1 / \lambda_D \rangle")

    vlines!(ax2, [k_min_rmt], color = LPCA_COLORS[1], linestyle = :dash, linewidth = 2.5)

    axislegend(ax2, position = :rt)

    if save_path !== nothing
        save(save_path, fig)
        println("Zapisano wykres do: $save_path")
    end

    display(fig)
    return fig
end

"""
    plot_full_rmt_lpca_suite(; N=10, T=252, D=3, K=12, save_path="marchenko_pastur_suite.png")

Generuje kompletny 4-panelowy zestaw publikacyjny łączący:
1. Rozkład Marczenki-Pastura z zadanymi N i T (z cytatu lub własnymi).
2. Rozpychanie wartości własnych w funkcji q = N/T.
3. Artefakt sztucznego spadku d_PR w lPCA dla czystego szumu D=3.
4. Porównanie czystego szumu z rzeczywistym atraktorem (redukcja wymiaru d*=1 i d*=2).
"""
function plot_full_rmt_lpca_suite(;
        N::Int = 10,
        T::Int = 252,
        D::Int = 3,
        K::Int = 12,
        save_path = "marchenko_pastur_suite.png"
    )
    println("\nGenerowanie pełnego zestawu publikacyjnego RMT & Local PCA...")

    fig = Figure(size = (1400, 1000))

    # --- PANEL 1: Rozkład MP (N, T) ---
    q1 = N / T
    lm1, lp1 = marchenko_pastur_limits(q1)
    evals1 = simulate_wishart_spectrum(N, T; n_trials = 60)

    ax1 = Axis(
        fig[1, 1],
        title = L"\text{(a) Rozkład MP: } N=%$(N),\, T=%$(T),\, q = %$(round(q1, digits=3))",
        xlabel = L"\lambda",
        ylabel = L"\rho(\lambda)",
    )
    hist!(ax1, evals1, bins = 35, normalization = :pdf, color = (LPCA_COLORS[4], 0.6), strokewidth = 1, strokecolor = LPCA_COLORS[5], label = L"\text{Empiryczne } \lambda_i")
    λ_grid = range(lm1 * 0.999, lp1 * 1.001, length = 300)
    lines!(ax1, λ_grid, [marchenko_pastur_pdf(λ, q1) for λ in λ_grid], color = LPCA_COLORS[1], linewidth = 3, label = L"\rho_{\mathrm{MP}}(\lambda)")
    vlines!(ax1, [lm1, lp1], color = [LPCA_COLORS[2], LPCA_COLORS[1]], linestyle = :dash, linewidth = 2)
    vspan!(ax1, lm1, lp1, color = (LPCA_COLORS[3], 0.12))
    axislegend(ax1, position = :rt)

    # --- PANEL 2: Pasmo szumu i rozpychanie w funkcji q ---
    ax2 = Axis(
        fig[1, 2],
        title = L"\text{(b) Rozpychanie widma: } \lambda_\pm(q) \text{ oraz granica } K \ge 12D",
        xlabel = L"q = N/T = D/K",
        ylabel = L"\text{Wartości własne } \lambda",
    )
    qs = 0.01:0.005:0.92
    lines!(ax2, qs, [(1.0 + sqrt(q))^2 for q in qs], color = LPCA_COLORS[1], linewidth = 2.5, label = L"\lambda_+(q)")
    lines!(ax2, qs, [(1.0 - sqrt(q))^2 for q in qs], color = LPCA_COLORS[4], linewidth = 2.5, label = L"\lambda_-(q)")
    vlines!(ax2, [1.0 / 12.0], color = LPCA_COLORS[2], linestyle = :dash, linewidth = 2.5, label = L"q = 1/12 \,(K=12D)")
    vlines!(ax2, [D / K], color = LPCA_COLORS[1], linestyle = :dot, linewidth = 2.5, label = L"q = %$(round(D/K, digits=2)) \,(K=%$(K))")
    axislegend(ax2, position = :lt)

    # --- PANEL 3: Pozorny spadek d_PR w lPCA dla czystego szumu ---
    Ks = [4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48, 64, 80, 96, 128]
    res3 = simulate_lpca_noise_dpr(D, Ks; n_eval = 400)
    ax3 = Axis(
        fig[2, 1],
        title = L"\text{(c) Artefakt lPCA: Fałszywa kompresja } d_{\mathrm{PR}} \text{ czystego szumu } D=%$(D)",
        xlabel = L"\text{Liczba sąsiadów } K",
        ylabel = L"d_{\mathrm{PR}}",
    )
    hlines!(ax3, [Float64(D)], color = :black, linestyle = :dot, linewidth = 2, label = L"\text{Fizyczny wymiar } D=%$(D)")
    band!(ax3, Ks, res3.mean_dprs .- res3.std_dprs, res3.mean_dprs .+ res3.std_dprs, color = (LPCA_COLORS[4], 0.25))
    lines!(ax3, Ks, res3.mean_dprs, color = LPCA_COLORS[5], linewidth = 3, label = L"\langle d_{\mathrm{PR}} \rangle \text{ (czysty szum)}")
    vlines!(ax3, [12 * D], color = LPCA_COLORS[1], linestyle = :dash, linewidth = 2.5, label = L"K \ge 12D = %$(12*D)")
    vspan!(ax3, Ks[1], 12 * D, color = (LPCA_COLORS[1], 0.08))
    axislegend(ax3, position = :rb)

    # --- PANEL 4: Sygnał vs Szum (Prawdziwy atraktor 1D/2D vs Pasmo MP) ---
    ax4 = Axis(
        fig[2, 2],
        title = L"\text{(d) Rozróżnialność atraktora: } \lambda_1 / \lambda_2 \text{ (Sygnał vs Szum MP)}",
        xlabel = L"\text{Liczba sąsiadów } K",
        ylabel = L"\text{Luka spektralna } g_1 = \lambda_1 / \lambda_2",
        yscale = log10
    )

    # Poziom tła szumu MP dla λ_1 / λ_2
    mp_noise_gap = [((1.0 + sqrt(D / (k - 1))) / (1.0 - sqrt(D / (k - 1))))^2 for k in Ks]
    lines!(ax4, Ks, mp_noise_gap, color = LPCA_COLORS[1], linestyle = :dash, linewidth = 2.5, label = L"\text{Szum RMT (Tło fluktuacji)}")

    # Symulacja atraktora 1D z grubością szumową σ_trans = 0.05
    n_sample = 400
    attractor_gaps = Float64[]
    for k in Ks
        gaps_k = Float64[]
        for _ in 1:n_sample
            # Punkty na linii 1D z szumem poprzecznym
            t = randn(k)
            noise_perp = 0.05 .* randn(k, D - 1)
            Y = hcat(t, noise_perp)
            Y_c = Y .- mean(Y, dims = 1)
            C = (transpose(Y_c) * Y_c) ./ (k - 1)
            ev = sort(eigvals(Symmetric(C)), rev = true)
            push!(gaps_k, ev[1] / max(ev[2], 1e-15))
        end
        push!(attractor_gaps, mean(gaps_k))
    end
    lines!(ax4, Ks, attractor_gaps, color = LPCA_COLORS[3], linewidth = 3, label = L"\text{Atraktor 1D } (\sigma_\perp = 0.05)")
    scatter!(ax4, Ks, attractor_gaps, color = LPCA_COLORS[3], markersize = 8)

    vlines!(ax4, [12 * D], color = LPCA_COLORS[1], linestyle = :dot, linewidth = 2)
    axislegend(ax4, position = :rt)

    if save_path !== nothing
        save(save_path, fig)
        println("\n>>> Zestaw publikacyjny zapisany w: $save_path <<<\n")
    end

    display(fig)
    return fig
end

# =============================================================================
# 4. INTERAKTYWNA KONSOLA W REPL (MENU WYBORU I WPISYWANIA PARAMETRÓW)
# =============================================================================

"""
    interactive_rmt_lpca()

Uruchamia w pełni interaktywny kreator w REPL, pytając użytkownika o parametry!
"""
function interactive_rmt_lpca()
    println("\n" * "="^70)
    println("   INTERAKTYWNY MODUŁ TEORII MACIERZY LOSOWYCH (RMT & lPCA)")
    println("="^70)
    println("Wybierz opcję do zbadania:")
    println(" [1] Rozkład Marczenki-Pastura dla własnych N i T (oraz sygnału Spike)")
    println(" [2] Rozpychanie wartości własnych (Eigenvalue Repulsion) vs q = N/T")
    println(" [3] Artefakt lPCA: Pozorny spadek d_PR dla czystego szumu izotropowego")
    println(" [4] Pełny 4-panelowy zestaw publikacyjny (Suite)")
    println(" [0] Wyjście")
    print("\nTwój wybór [1-4]: ")

    choice = strip(readline())

    if choice == "1"
        print("Podaj wymiar N (np. 10 lub 50): ")
        N = parse(Int, strip(readline()))
        print("Podaj liczbę próbek T (np. 252 lub 500): ")
        T = parse(Int, strip(readline()))
        print("Czy dodać sygnał fizyczny (Spike)? Podaj siłę sygnału (0.0 = czysty szum, np. 3.5): ")
        sig = parse(Float64, strip(readline()))
        plot_marchenko_pastur(N = N, T = T, signal_strength = sig)

    elseif choice == "2"
        println("Rysowanie krzywych rozpychania wartości własnych w funkcji q...")
        plot_repulsion_vs_q()

    elseif choice == "3"
        print("Podaj wymiar przestrzeni fazowej D (np. 2 dla MIS, 3 dla HJSW): ")
        D = parse(Int, strip(readline()))
        print("Podaj maksymalne K sąsiadów do przetestowania (np. 96 lub 128): ")
        max_k = parse(Int, strip(readline()))
        Ks = filter(k -> k >= D + 1, [4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 80, 96, 128])
        Ks = filter(k -> k <= max_k, Ks)
        plot_lpca_noise_artifact(D = D, Ks = Ks)

    elseif choice == "4"
        print("Podaj N dla panelu 1 (domyślnie 10): ")
        N_str = strip(readline())
        N = isempty(N_str) ? 10 : parse(Int, N_str)
        print("Podaj T dla panelu 1 (domyślnie 252): ")
        T_str = strip(readline())
        T = isempty(T_str) ? 252 : parse(Int, T_str)
        print("Podaj D dla lPCA (domyślnie 3): ")
        D_str = strip(readline())
        D = isempty(D_str) ? 3 : parse(Int, D_str)
        plot_full_rmt_lpca_suite(N = N, T = T, D = D)

    else
        println("Wyjście z modułu interaktywnego.")
    end
end

println("""
=============================================================================
 Załadowano moduł: marchenko_pastur_lpca.jl
 Dostępne gotowe funkcje do wpisywania parametrów w REPL:

 1. plot_marchenko_pastur(N=10, T=252; signal_strength=0.0, n_trials=80)
 2. plot_repulsion_vs_q()
 3. plot_lpca_noise_artifact(D=3, Ks=[4, 6, 8, 12, 16, 24, 36, 48, 64, 96])
 4. plot_full_rmt_lpca_suite(N=10, T=252, D=3, K=12)
 5. interactive_rmt_lpca()  <- Wpisuj parametry krok po kroku w konsoli!
=============================================================================
""")
