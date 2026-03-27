using Symbolics

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





local N = 4

@variables x[1:N] y[1:N] W[1:N, 1:N] Λ[1:N]

I_n = [1, 1, 1, 1]

♉ = W * I_n .- I_n




👎 = sum( Λ[k] * ♉[k] for k in 1:N)

👍= Num[]
for i in 1:N
    rek_x = sum(W[i, j] * x[j] for j in 1:N)
    rek_y = sum(W[i, j] * y[j] for j in 1:N)

    blad = (x[i] - rek_x)^2 + (y[i] - rek_y)^2
    push!(👍, blad)
end

L = 👍 .- 👎

i = 2

L_i = substitute(L[i], Dict(W[i,i] => 0))

G_w21 = Symbolics.derivative(L_i, W[2, 1])
G_w23 = Symbolics.derivative(L_i, W[2, 3])
G_w24 = Symbolics.derivative(L_i, W[2, 4])

G_lambda = Symbolics.derivative(L_i, Λ[i])

println("Względem W[2,1]: ", G_w21)
println("Względem W[2,3]: ", G_w23)
println("Względem W[2,4]: ", G_w24)
println("Względem Λ[2]: ", G_lambda)
