using Documenter
using AtractorsQGP

makedocs(
    sitename = "AtractorsQGP.jl",
    modules = [AtractorsQGP],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        ansicolor = true
    ),
    pages = [
        "Home" => "index.md",
        "User Guide" => "guide.md",
        "Extras" => "extras.md",
        "Examples of usage" => "examples.md",
        "Tutorials" => "tutorials.md",
    ],
    warnonly = [:missing_docs, :cross_references]
)

deploydocs(
    repo = "github.com/kitajusSus/Atractors-in-QGP.git",
    devbranch = "master",
)
