# How to use packages

Here I am showing examples of using different packages used in the project for better understanding by me and my colleagues.

# Symbolics.jl

!!! info "What Is this"
    `Symbolics.jl` is a package for algebraic and mathematical equations for
    providing a `Wolfram Mathematica` type experience in Julia (free == better).


## Mathematical Transformation

To avoid writing a complex text parser for negative powers of $w$, we define the substitution:
$z = \frac{1}{w}$

By doing this, the asymptotic series becomes a standard Taylor series expanded around $z = 0$:
$$f(z) = a_0 + a_1 z + a_2 z^2 + a_3 z^3 + \dots$$

Using the chain rule, the derivative transforms accordingly:
$$\frac{df}{dw} = \frac{df}{dz} \frac{dz}{dw} = -z^2 \frac{df}{dz}$$

With this transformation, we can find the `$n$`-th coefficient by isolating it through sequential differentiation of the entire equation with respect to `$z$`, evaluating it at `$z = 0$`, and solving the resulting linear equation.

!!! warning "Limitations of Symbolic Computation"
    No automated tool is infallible. While this mathematical approach is highly robust, `Symbolics.jl` handles
    nonlinear roots (such as the initial quadratic equation for $a_0$)
    In professional numerical code, it is standard practice to manually assign the physical branch of the $a_0$
    solution based on the known boundary conditions (e.g., ideal hydrodynamics limits).
    The code below implements this safeguard explicitly.

Basic
```julia
using Symbolics
```

1. Declare symbolic variables
```julia
@variables z C_tau C_eta
```
2. Create an array of symbolic variables for coefficients
```julia
@variables a[1:degree]
```
3. Define the Taylor series f(z)
```julia
fz = sum(a[i] * z^(i-1) for i in 1:degree)
```
4. Define the derivative operator and compute f'(z)
```julia
\partial_z = Differential(z)
\partial_fz = expand_derivatives(\partial_z(f_z))
```

5. Define the transformed MIS equation
```julia
eq = -C_tau * z * f_z * df_z + f_z^2 - (2//3) * f_z + C_eta * z
```

6. End
```julia

solutions = Dict()
eq_deriv = eq

for n in 0:(degree-1)
# Evaluate the current equation at z = 0
eq_at_zero = substitute(eq_deriv, Dict(z => 0))

# Substitute the previously found coefficients
eq_subbed = substitute(eq_at_zero, solutions)

if n == 0
    # Manually select the non-trivial, physical branch (2/3)
    solutions[a[1]] = 2//3
    println("a_0 = ", solutions[a[1]], " (Physical branch selected)")
else
    # Solve the strictly linear equation for a[n+1]
    sol = Symbolics.solve_for(eq_subbed ~ 0, a[n+1])
    sol_simplified = simplify(sol)
    solutions[a[n+1]] = sol_simplified

    println("a_$n = ", sol_simplified)
end

# Differentiate for the next iteration step
eq_deriv = expand_derivatives(D_z(eq_deriv))
end

return solutions
end
results = compute_asymptotic_series(5)

```


## API Reference & Function Glossary

This section explains the core `Symbolics.jl` functions utilized in the script.

!!! tip "Performance Note"
    Using `Symbolics.jl` natively within Julia avoids the overhead of calling Python-based wrappers, making operations like `expand_derivatives` exceptionally fast for deeply nested expressions.

### Variable Registration
* `@variables`: Registers symbols in the namespace. Calling `@variables a[1:5]`
dynamically generates an array of independent symbolic variables, forming the
foundation of the Abstract Syntax Tree (AST). (**Not important for us**)

### Calculus Operations
* `Differential(z)`: Creates an abstract differential operator. It marks the
expression for differentiation without computing it immediately.
* `expand_derivatives()`: The execution command. It applies product,
quotient, and chain rules to compute the exact analytical derivative of expressions wrapped in `Differential`.

### Algebraic Manipulation
* `~` (Tilde): Defines a symbolic equality constraint. For example, `eq ~ 0`
represents an equation where the left side is mathematically equal to zero.
* `substitute(expression, mapping)`: Replaces symbols with exact values or other symbols.
The `mapping` dictionary allows us to isolate terms by evaluating at `$z = 0$`.
* `Symbolics.solve_for(equation, variable)`: A linear algebraic solver that isolates
the requested `variable` on one side of the equation.
* `simplify()`: Reduces the expression to its shortest possible form by combining
like terms and applying basic algebraic rules.


# DifferentialEquations.jl




```julia
# ...
