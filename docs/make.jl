using Documenter
using AttractorsQGP

makedocs(
    sitename = "AttractorsQGP.jl",
    modules = [AttractorsQGP],
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
        "NCBJ" => "ncbj.md",
        "HJSW Article Tutorials" => "hjsw.md",
        "LPCA Attractor Identification" => "lpca_publication.md",
        "Matlab" => "matlab.md",
    ],
    warnonly = [:missing_docs, :cross_references]
)

deploydocs(
    repo = "github.com/kitajusSus/AttractorsQGP.jl.git",
    devbranch = "master",
)
