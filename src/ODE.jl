using Symbolics
using SymPy

function solve_attractor()
    Cn = 1.0
    CtP = 1.0
    Cl1 = 1.0

    w = 0.5
    A = 8.0
    dw = 0.05
    steps = 20

    println("w \t A(w)")

    for i in 1:steps
        term_denom = CtP * (1 + A / 12)
        term_sq = (CtP / (3 * w) + Cl1 / (8 * Cn)) * A^2
        rhs = 1.5 * (8 * Cn / w - A)

        dAdw = (rhs - term_sq) / term_denom

        A = A + dAdw * dw
        w = w + dw

        println(round(w, digits=3), "\t", A)
    end
end

function solve_analytical_coefficients()
    @variables x Cn CtP Cl1
    @variables a1 a2 a3

    A = a1 * x + a2 * x^2 + a3 * x^3
    Ap = -x^2 * Symbolics.derivative(A, x)

    term1 = CtP * (1 + A / 12) * Ap
    term2 = (CtP * x / 3 + Cl1 / (8 * Cn)) * A^2
    LHS = term1 + term2
    RHS = 1.5 * (8 * Cn * x - A)

    Residue = Symbolics.expand(LHS - RHS)

    d1 = Symbolics.derivative(Residue, x)
    eq1 = Symbolics.substitute(d1, Dict(x => 0))
    sol_a1 = Symbolics.solve_for(eq1 ~ 0, a1)
    println("a1 = ", sol_a1)

    Residue_2 = Symbolics.substitute(Residue, Dict(a1 => sol_a1))
    d2 = Symbolics.derivative(Symbolics.derivative(Residue_2, x), x)
    eq2 = Symbolics.substitute(d2, Dict(x => 0)) / 2
    sol_a2 = Symbolics.solve_for(eq2 ~ 0, a2)
    println("a2 = ", Symbolics.simplify(sol_a2))

    Residue_3 = Symbolics.substitute(Residue_2, Dict(a2 => sol_a2))
    d3 = Symbolics.derivative(Symbolics.derivative(Symbolics.derivative(Residue_3, x), x), x)
    eq3 = Symbolics.substitute(d3, Dict(x => 0)) / 6
    sol_a3 = Symbolics.solve_for(eq3 ~ 0, a3)
    println("a3 = ", Symbolics.simplify(sol_a3))
end

function ODE_sympy()
    w = SymPy.symbols("w")
    Cn, CtP, Cl1 = SymPy.symbols("Cn CtP Cl1")
    a1, a2, a3 = SymPy.symbols("a1 a2 a3")

    A = a1 / w + a2 / w^2 + a3 / w^3
    dA = diff(A, w)

    LHS = CtP * (1 + A / 12) * dA + (CtP / (3 * w) + Cl1 / (8 * Cn)) * A^2
    RHS = (3 / 2) * (8 * Cn / w - A)
    EQ = LHS - RHS

    x = SymPy.symbols("x")
    EQ_sub = EQ.subs(w, 1 / x)
    EQ_series = SymPy.series(EQ_sub, x, 0, 4)

    c1 = EQ_series.coeff(x, 1)
    c2 = EQ_series.coeff(x, 2)
    c3 = EQ_series.coeff(x, 3)

    solution = SymPy.solve([c1, c2, c3], (a1, a2, a3))
    println(solution)
end

println("--- Symbolics Solution ---")
solve_analytical_coefficients()

println("\n--- SymPy Solution ---")
ODE_sympy()

println("\n--- Numerical Solution ---")
solve_attractor()
