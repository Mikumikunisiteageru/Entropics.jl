# docs/make.jl

using Entropics

using Documenter

makedocs(
	sitename = "Entropics.jl",
	pages = [
		"Entropics.jl" => "index.md",
		],
)

deploydocs(
    repo = "github.com/Mikumikunisiteageru/Entropics.jl.git",
	versions = ["stable" => "v^", "v#.#.#"]
)
