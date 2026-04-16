# NCBJ docs

Bardzo przepraszam ale w ramach zwiększenia zrozumienia kodu i ideii nazwy są pisane
dwunastozgłoskowcem

## Jak używać `example/ncbj_lle.jl` ?
> wszystko przygotowane pod prace z repl

- krok 1
uruchamianie środowiska
```julia
using AtractorsQGP
includet("examples/ncbj_lle.jl")
```

- krok 2 Tworzenie danych do  testowania

```julia
wektor_x = [0.2:5...]
 macierz = ncbj1_macierz_wszyskich_punktów!(wektor_x)
```


- krok 3 znajdowanie sąsiadów dla wybranego punktu
```julia
ile_sąsiadów = 3
indeks_wybranego_punktu = 3
sasiedzi, indeksy_sasiadow, wybrany_punkt = ncbj2_sąsiedzi!(macierz, indeks_wybranego_punktu, ile_sąsiadów )
```

> należy pamietać że nazwy służą  temu by każdy wiedział co jest czym bez szczególnego zastanawiania


- krok 4  obliczanie macierzy wag


```julia
ncbj3_calculate_wagi_dla_x_i!(sasiedzi, wybrany_punkt, ile_sąsiadów)




```

## Dodane nowe funkcje i porządna "refactoryzacja"

jak uruchamiać przykłady? 

```julia
julia src/examples/lle_examples/swissroll.jl
julia src/examples/lle_examples/helix.jl
```


