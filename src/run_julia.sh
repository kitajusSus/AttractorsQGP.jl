#!/bin/bash


echo "-> Aktywowanie i przygotowywanie środowiska Julia..."

julia --project=. -e '
  using Pkg;
  Pkg.instantiate();
  println("\n✅ Środowisko gotowe. Uruchamianie REPL z Revise...");
'

if [ $? -ne 0 ]; then
    echo "❌ Błąd podczas przygotowywania środowiska. Przerwanie."
    exit 1
fi

julia --project=. -i -e 'using Revise'
