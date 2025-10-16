# Thesis


## Hydrodynamic Attractors in Phase Space
Wszystkie Artykuły znajdują się w [HR Articles]

### youtube
[link do zapytaj fizyka z helerem](https://www.youtube.com/watch?v=6R2ASA7-g-c&t=9s)

### Źródło różnic w rozwiązaniach równania $A(w)$
[mis_vs_brsss](notes/mis_vs_brsss.md)




## Julia
czego potrzeba do pracy z julia?
1. [Julia](https://julialang.org/downloads/)
2. Instalowania pakietów w menadżerze pakietów julia (REPL) poprzez wpisanie `]` i potem `add` oraz nazwy pakietu.
```bash

using Pkg
Pkg.instantiate()

# or
cd src
julia
julia> ]
(@v1.11) pkg> activate .
(src) pkg> instantiate
# copy this commands without anything before `>`

```
3. Ważne by uruchamiać julia w terminal
```bash
> julia
> include("nazwa_pliku.jl")
```
Procedura używania REPL (Read-Eval-Print Loop) w julia jest następująca:
```bash
include("modHydroSim.jl")
using .modHydroSim

# --- Eksperyment 1: Szybki test z domyślnymi ustawieniami ---
# Tworzysz obiekt ustawień bez podawania argumentów - użyje domyślnych.
settings1 = SimSettings()
result1 = run_simulation(PARAMS_MIS_TOY_MODEL, settings1);
create_log_ratio_animation(result1, filename="run_default.gif")


# --- Eksperyment 2: Dłuższa ewolucja dla modelu SYM ---
# Tworzysz obiekt ustawień, nadpisując tylko czas końcowy.
settings2 = SimSettings(τ_end=2.5)
result2 = run_simulation(PARAMS_SYM_THEORY, settings2);
create_log_ratio_animation(result2, filename="run_sym_long.gif")


# --- Eksperyment 3: Więcej punktów, inny zakres temperatur ---
settings3 = SimSettings(n_points=500, T_range=(400.0, 800.0))
result3 = run_simulation(PARAMS_SYM_THEORY, settings3);
create_log_ratio_animation(result3, filename="run_sym_dense_hot.gif")

```

### Programy napisane w julia
Programy napisane w julia znajdują się w katalogu [src](/src/).

- [Generowanie danych](src/generowanie_AiT.jl) - program generujący ewolucję $A(\tau)$ i $T(\tau)$ dla  warunków początkowych. do pliku .csv

`
## Wygenerowane wykresy
Wszystkie rysunki i wykresy wygenerowane przez kod bede starał się umieszczać w katalogu [images](/images/). Jeśli nie będzie tak żadnego wykresu to zalecam sprawdzenie katalogu [src](/src/) gdzie powinny być wygenerowane wykresy których jeszcze nie przeniosłem.

# Raport
> ## 19.07.2025

## 24.07.2025
```julia
julia> include("modHydroSim.jl")
Main.modHydroSim

julia> using .modHydroSim

julia> settings1 = SimSettings()
SimSettings(200, 0.2, 1.2, (0.2, 1.2), (300.0, 500.0), (0.0, 4.0))

julia> settings2 = SimSettings(T_range(1.0,2.0))
ERROR: UndefVarError: `T_range` not defined in `Main`
Suggestion: check for spelling errors or missing imports.
Stacktrace:
 [1] top-level scope
   @ REPL[4]:1

julia> settings2 = SimSettings(T_range=(1.0,2.0))
SimSettings(200, 0.2, 1.2, (0.2, 1.2), (1.0, 2.0), (0.0, 4.0))

julia> result_default = run_simulation(PARAMS_MIS_TOY_MODEL, settings1);
--- Rozpoczynanie Obliczeń Numerycznych...
--- Obliczenia Zakończone. ---

julia> create_log_ratio_animation(result_default, filename= "testowy.gif")
Generowanie animacji: testowy.gif...
```
Już po nauczeniu się obsługi REPL. [program modHydroSim.jl](/src/modHydroSim.jl) jest gotowy do użycia. Wystarczy go załadować i można korzystać z funkcji `run_simulation` oraz `create_log_ratio_animation` i innych.



**Wykresy i zdjęcie**
- ![zdjecie](images/A_T/25.07.2025.png)

- ![zdjecie_2](images/A_T/27.07.2025.gif)

# **[HINTON_SNE](neural_networks/sne.pdf)** Implementacja Algorytmu SNE do badania dynamiki regukcji wymiarowości
Algorytm SNE (Stochastic Neighbor Embedding) jest używany do redukcji wymiarowości danych. W kontekście badania dynamiki redukcji wymiarowości, implementacja tego algorytmu może być przydatna do analizy danych z symulacji hydrodynamicznych m.in
takich których wyniki widoczne są w [Symulacja Ewolucji A i T](images/A_T/27.07.2025.gif) Gdzie dokładnie wydać jak zmienia się dynamika (tempo redukcji wymiarowości do z 2d do 1d)  w czasie.

- **Kod z implementacją tego Algorytmu jest umieszczony** -> [kod_sne](src/sne.jl)

*Jak używać w REPL?*

```bash
> julia
> ]
> pkg > activate . # aktywacja środowiska, jeśli jest w katalogu
> include("lib.jl")
> using .modHydroSim
# opcjonalnie ja osobiście używam using Revise by móc modyfikować kod bez restartowania REPL
#albo po prostu
> include("sne.jl")
```


## PCA
- ![pca1](images/pca/pca_1.png)
- ![pca2](images/pca/pca_2.png)
- ![pca3](images/pca/pca_3.png)
- ![pca4](images/pca/pca_4.png)



# T-SNE/sne
-![t-sne](images/sne/sne1.png)
-![t-sne2](images/sne/sne2.png)




# Citation
```tex
@misc{bezubik2025attractors,
  author       = {Krzysztof Bezubik and Michał Spaliński},
  title        = {Thesis; Atractors in Physics of quark gluon plasma},
  year         = {2025},
  version      = {1.0.0},
  url          = {https://github.com/kitajusSus/Atractors-in-QGP},
  note         = {If you use this work, please cite it using this entry.}
}
```
# Reference
