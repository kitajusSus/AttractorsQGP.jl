using Symbolics

function compute_asymptotic_series(degree::Int)
    @variables z C_tau C_eta

    # 2. Create an array of symbolic variables for coefficients a_0, a_1, ...
    @variables a[1:degree]

    # 3. Define the series f(z) = a_0 + a_1*z + a_2*z^2 + ...
    # a[1] represents a_0, so the exponent is (i-1)
    f_z = sum(a[i] * z^(i-1) for i in 1:degree)

    # 4. Define the derivative operator and compute f'(z)
    ∂_z = Differential(z)
    ∂f_z = expand_derivatives(∂_z(f_z))

    # 5. Define the transformed MIS equation
    # Original: C_tau * w * f * (df/dw) + f^2 - (2/3)*f + C_eta / w = 0
    # Substituted: C_tau * (1/z) * f * (-z^2 * df/dz) + f^2 - (2/3)*f + C_eta * z = 0
    eq = -C_tau * z * f_z * ∂f_z + f_z^2 - (2/3) * f_z + C_eta * z

    println("Starting computation of asymptotic coefficients up to order ", degree - 1)

    # Dictionary to store our solved coefficients
    solutions = Dict()

    # Variable to hold the current derivative of the equation
    eq_deriv = eq

    for n in 0:(degree-1)
        # Evaluate the current equation at z = 0
        eq_at_zero = substitute(eq_deriv, Dict(z => 0))

        # Substitute the previously found coefficients (a_0 ... a_{n-1})
        eq_subbed = substitute(eq_at_zero, solutions)

        if n == 0
            # For n = 0, the equation is a_0^2 - (2/3)a_0 = 0.
            # We manually select the non-trivial, physical branch for ideal hydrodynamics.
            # Using 2//3 ensures exact rational arithmetic instead of floating-point numbers.
            solutions[a[1]] = 2/3
            println("a_0 = ", solutions[a[1]], " (Physical branch selected)")
        else
            # We set the substituted expression equal to 0 and solve for a[n+1].
            sol = Symbolics.solve_for(eq_subbed ~ 0, a[n+1])

            # Simplify the resulting expression
            sol_simplified = simplify(sol)
            solutions[a[n+1]] = sol_simplified

            println("a_$n = ", sol_simplified)
        end
        # for next iteration
        eq_deriv = expand_derivatives(  ∂_z(eq_deriv))
    end

    return solutions
end

results = @time compute_asymptotic_series(7)
