#!/bin/bash


echo "-> Aktywowanie i przygotowywanie środowiska Julia..."

julia --project=. -e '
  using Pkg;
  Pkg.instantiate();
  println("\n✅ Środowisko gotowe. Uruchamianie REPL z Revise...");
'

# Sprawdzenie, czy poprzednie polecenie się powiodło
if [ $? -ne 0 ]; then
    echo "❌ Błąd podczas przygotowywania środowiska. Przerwanie."
    exit 1
fi

# Uruchomienie Julii w trybie interaktywnym (-i) z załadowanym Revise
julia --project=. -i -e 'using Revise'
