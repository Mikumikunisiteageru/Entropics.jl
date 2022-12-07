# runtests.jl
# 20220225

using Entropics

using SpecialFunctions: erfi

using Test

@testset "MATHEMATIC FUNCTIONS" begin
	
	@testset "erfiinv" begin
		for i = [-10, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 10, 100, 1000]
			@test isapprox(i, erfi(Entropics.erfiinv(i)))
		end
		for i = Float32[-10, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 10, 100, 1000]
			@test isapprox(i, erfi(Entropics.erfiinv(i)))
		end
	end
	
end
