using Documenter
using AtractorsQGP


makedocs(
    sitename = "AtractorsQGP.jl",
    format = Documenter.HTML(),
    modules = [AtractorsQGP],
    pages = Any[
        "Home" => "index.md",
    #     "Examples" => "examples.md",
    #     "API" => "api.md",
    ],
)
