# test/runtests.jl

using Entropics

tests = ["math", "types", "sample", "maxendist"]

for test = tests
	testfile = string(test, ".jl")
	println(" * $(testfile) ...")
	include(testfile)
end
