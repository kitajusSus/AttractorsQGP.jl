

using TaylorSeries



using Symbolics


function main()
    @variables x
    expr = sin(x) / x
    substitute(expr, Dict(x => 0.000001))


    t = Taylor1(Float64, 5)
    f = sin(t) / t

    println("Rozwinięcie w szereg:")
    display(f)
    granica = f[0]


    println("\nGranica w zerze wynosi: ", granica)
end
