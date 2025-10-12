using CSV, DataFrames, Random

# --- Ustawienia ---
const N_POINTS = 10000
const T_RANGE = (200.0, 1800.0)
const A_RANGE = (-12.0, 25.0)
const output_filename = "initial_conditions.csv"

# --- Konfiguracja generatora liczb losowych ---
# Ustaw ziarno (seed) dla powtarzalności wyników.
# Zmieniając tę liczbę, otrzymasz inny zestaw losowych warunków.
const seed = 5
# Stwórz instancję generatora MersenneTwister z podanym ziarnem.
const rng = Xoshiro(seed)

println("Generowanie 10000 losowych warunków początkowych z ziarnem: $seed...")

initial_T = T_RANGE[1] .+ (T_RANGE[2] - T_RANGE[1]) .* rand(rng, N_POINTS)
initial_A = A_RANGE[1] .+ (A_RANGE[2] - A_RANGE[1]) .* rand(rng, N_POINTS)

# Stwórz DataFrame z wygenerowanych danych
df = DataFrame(
    Run_ID = 1:N_POINTS,
    T_0 = initial_T,
    A_0 = initial_A
)

# Pokaż kilka pierwszych wierszy w konsoli
println("Wygenerowane dane (pierwsze 5 wierszy):")
println(first(df, 5))

# Zapisz wyniki do pliku CSV
println("\nZapisywanie wyników do pliku: $output_filename")
CSV.write(output_filename, df)

println("\nGotowe!")
