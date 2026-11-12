"""
    citation([io::IO=stdout])

Displays the repository BibTeX citation for `AttractorsQGP.jl` in REPL or writes it to the given `io` stream.

# Example
```julia
AttractorsQGP.citation()
```
"""
function citation(io::IO = stdout)
    default_bib = """
    @misc{AttractorsQGP2026,
      author       = {Krzysztof Bezubik and Michał Spaliński},
      title        = {{AttractorsQGP.jl}: Hydrodynamic Attractors in Quark-Gluon Plasma},
      year         = {2026},
      howpublished = {\\url{https://github.com/kitajusSus/Attractors-in-QGP}},
      url          = {https://github.com/kitajusSus/Attractors-in-QGP}
    }"""
    root_citation = joinpath(pkgdir(@__MODULE__), "CITATION.bib")
    bib_text = isfile(root_citation) ? read(root_citation, String) : default_bib
    print(io, bib_text)
    if !endswith(bib_text, "\n")
        println(io)
    end
    return nothing
end
