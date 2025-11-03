using SymPy
using Plots
using Roots

const h = 6.62607015e-34      # Plancka [J·s]
const c = 2.99792458e8        # światło [m/s]
const k = 1.380649e-23        # Boltzmanna [J/K]

function planck(λ, T)
  return (2h * c^2) / λ^5 * 1 / (exp(h * c / (λ * k * T)) - 1)
end

function skibidi()
  @syms λ T
  f = (2*pi*h*c^2)*λ^(-5) * exp(-h * c / (λ * k * T))
  I = integrate(f, (λ, 0, oo))
  println("Całka (symbolicznie):")
  println(I)
end

function wykres()
  λ = range(100e-9, 3000e-9, length=1000) # długości fal od 100nm do 3000nm
  Ts = [3000, 4000, 5000, 6000]           # przykładowe temperatury w K
  plot()
  for T in Ts
    plot!(λ .* 1e9, planck.(λ, T), label="T = $T K") # λ w nm na osi x
  end
  xlabel!("λ [nm]")
  ylabel!("B(λ, T) [W·m⁻³·sr⁻¹]")
  title!("Widmo Plancka dla różnych temperatur")
  gui()
end

function wien_shift()
  Temps = 2000:500:8000
  λmaxs = Float64[]
  for T in Temps
    λs = 100e-9:1e-9:3000e-9
    f_num(λ) = planck(λ, T)
    λmax = λs[argmax(f_num.(λs))]
    push!(λmaxs, λmax)
    println("T = $T K, λ_max = $(λmax*1e9) nm, λ_max*T = $(λmax*T) m·K")
  end
  plot(Temps, λmaxs .* Temps, label="λ_max * T", xlabel="T [K]", ylabel="λ_max * T [m·K]", legend=:bottomright)
  hline!([2.8978e-3], label="Stała Wiena", linestyle=:dash)
  title!("Prawo przesunięć Wiena")
  gui()
end
skibidi()
wykres()
wien_shift()
