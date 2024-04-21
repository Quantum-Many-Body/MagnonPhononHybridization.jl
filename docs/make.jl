using MagnonPhononHybridization
using Documenter

DocMeta.setdocmeta!(MagnonPhononHybridization, :DocTestSetup, :(using MagnonPhononHybridization); recursive=true)

makedocs(;
    modules=[MagnonPhononHybridization],
    authors="waltergu <waltergu1989@gmail.com> and contributors",
    repo="https://github.com/Quantum-Many-Body/MagnonPhononHybridization.jl/blob/{commit}{path}#{line}",
    sitename="MagnonPhononHybridization.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Quantum-Many-Body.github.io/MagnonPhononHybridization.jl",
        assets=["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/MagnonPhononHybridization.jl",
    devbranch="main",
)
