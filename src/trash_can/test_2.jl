include("lib.jl")
using .modHydroSim

println("Biblioteki wczytane. Program Do robienia wykresu na bazie run_simulation z lib.jl")

ustawienia = SimSettings(tspan=(0.2, 1.0))

wynik_symulacji = run_simulation(ustawienia)

println("Symulacja zakończona. Generowanie wykresu...")

kadr(wynik_symulacji, 0.5)

println("\nWykres gotowy. Naciśnij Enter, aby zakończyć.")

readline()
