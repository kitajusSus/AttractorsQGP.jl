using Symbolics
function basic_lag()
    @variables x y λ
    f = x^2 + y^2
    g = x + y - 1
    L = f - λ * g
    dL_dx = Symbolics.derivative(L, x)
    dL_dy = Symbolics.derivative(L, y)
    dL_dλ = Symbolics.derivative(L, λ)

    println("mati ma chyba ?G == 0 oraz Ri == pochodna po lambda == 0 z Mathematiki):")
    println("Po x: ", dL_dx, " = 0")
    println("Po y: ", dL_dy, " = 0")
    println("Po λ: ", dL_dλ, " = 0")
end


function lagrang()

    # 3 wagi oraz jeden mnożnik Lagrange'a (λ)
    @variables w[1:3] λ

    # (X) oraz jego 3 sąsiadów (N1, N2, N3)
    @variables X[1:2] N1[1:2] N2[1:2] N3[1:2]

    # 2. Równania rekonstrukcji
    # X jako kombinację liniową jego sąsiadów
    rek_X = w[1] * N1[1] + w[2] * N2[1] + w[3] * N3[1]
    rek_Y = w[1] * N1[2] + w[2] * N2[2] + w[3] * N3[2]

    # Funkcja celu  (ścieżka przy wspinaniu na szczyt) (błąd rekonstrukcji do zminimalizowania)
    # Suma kwadratów różnic między oryginałem a rekonstrukcją w obu wymiarach
    błąd = (X[1] - rek_X)^2 + (X[2] - rek_Y)^2

    # wiąz
    # wagi_ij == 1.
    # f(w) = 0
    więz = w[1] + w[2] + w[3] - 1

    # Lagrangian (L)  z mnożnikiem λ
    L = błąd - λ * więz

    dL_dw1 = Symbolics.derivative(L, w[1])
    dL_dw2 = Symbolics.derivative(L, w[2])
    dL_dw3 = Symbolics.derivative(L, w[3])
    # powinna zwrócic więz
    dL_dλ = Symbolics.derivative(L, λ)

    println("∂L/∂w1 = ", dL_dw1)
    println("∂L/∂w2 = ", dL_dw2)
    println("∂L/∂w3 = ", dL_dw3)
    println("∂L/∂λ  = ", dL_dλ)
end
