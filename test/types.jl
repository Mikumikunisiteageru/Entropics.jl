# test/types.jl

using Entropics
using Test

@testset "Distribution" begin
	Distribution = Entropics.Distribution
	@test isabstracttype(Distribution)
	@test Distribution{Float64} <: Distribution
	@test ! (Distribution{Float64} <: Distribution{AbstractFloat})
	@test_throws TypeError Distribution{Int}
end

@testset "Bounded" begin
	Distribution = Entropics.Distribution
	Bounded = Entropics.Bounded
	@test isabstracttype(Bounded)
	@test Bounded{Float64} <: Bounded
	@test ! (Bounded{Float64} <: Bounded{AbstractFloat})
	@test_throws TypeError Bounded{Int}
end

@testset "Unbounded" begin
	Distribution = Entropics.Distribution
	Unbounded = Entropics.Unbounded
	@test isabstracttype(Unbounded)
	@test Unbounded{Float64} <: Unbounded
	@test ! (Unbounded{Float64} <: Unbounded{AbstractFloat})
	@test_throws TypeError Unbounded{Int}
end

@testset "pab" begin
	pab = Entropics.pab
	f(x) = pab(x -> x / 36.0, x, 3.0, 9.0)
	@test isapprox(f(0.0), 0.0)
	@test isapprox(f(3.0), 3.0 / 36.0)
	@test isapprox(f(6.0), 6.0 / 36.0)
	@test isapprox(f(9.0), 9.0 / 36.0)
	@test isapprox(f(12.0), 0.0)
	g(x) = pab(error, x, 3.0, 9.0)
	@test iszero(g(prevfloat(3.0)))
	@test_throws ErrorException g(3.0)
	@test_throws ErrorException g(9.0)
	@test iszero(g(nextfloat(9.0)))	
end

@testset "pamb" begin
	pamb = Entropics.pamb
	f(x) = pamb(x -> x / 16.0, x -> x / 56.0, x, 3.0, 5.0, 9.0)
	@test isapprox(f(0.0), 0.0)
	@test isapprox(f(3.0), 3.0 / 16.0)
	@test isapprox(f(5.0), 5.0 / 16.0)
	@test isapprox(f(nextfloat(5.0)), nextfloat(5.0) / 56.0)
	@test isapprox(f(6.0), 6.0 / 56.0)
	@test isapprox(f(9.0), 9.0 / 56.0)
	@test isapprox(f(12.0), 0.0)
	g(x) = pamb(error, identity, x, 3.0, 5.0, 9.0)
	@test iszero(g(prevfloat(3.0)))
	@test_throws ErrorException g(3.0)
	@test_throws ErrorException g(5.0)
	@test isapprox(g(nextfloat(5.0)), 5.0)	
	h(x) = pamb(identity, error, x, 3.0, 5.0, 9.0)
	@test isapprox(h(5.0), 5.0)
	@test_throws ErrorException h(nextfloat(5.0))
	@test_throws ErrorException h(9.0)
	@test isapprox(h(nextfloat(9.0)), 0.0)	
end

@testset "Pab" begin
	Pab = Entropics.Pab
	f(x) = Pab(x -> 6.0 \ (x - 3.0), x, 3.0, 9.0)
	@test isapprox(f(0.0), 0.0)
	@test isapprox(f(3.0), 0.0)
	@test isapprox(f(6.0), 0.5)
	@test isapprox(f(9.0), 1.0)
	@test isapprox(f(12.0), 1.0)
	g(x) = Pab(error, x, 3.0, 9.0)
	@test iszero(g(prevfloat(3.0)))
	@test_throws ErrorException g(3.0)
	@test_throws ErrorException g(9.0)
	@test isone(g(nextfloat(9.0)))	
end

@testset "Pamb" begin
	Pamb = Entropics.Pamb
	f(x) = Pamb(x -> 4.0 \ (x - 3.0), 
				x -> 8.0 \ (x - 5.0), x, 3.0, 5.0, 9.0)
	@test isapprox(f(0.0), 0.0)
	@test isapprox(f(3.0), 0.0)
	@test isapprox(f(4.0), 0.25)
	@test isapprox(f(5.0), 0.5)
	@test isapprox(f(nextfloat(5.0)), 0.5)
	@test isapprox(f(7.0), 0.75)
	@test isapprox(f(9.0), 1.0)
	@test isapprox(f(12.0), 1.0)
	g(x) = Pamb(error, identity, x, 3.0, 5.0, 9.0)
	@test iszero(g(prevfloat(3.0)))
	@test_throws ErrorException g(3.0)
	@test_throws ErrorException g(5.0)
	@test isapprox(g(nextfloat(5.0)), 5.5)	
	h(x) = Pamb(identity, error, x, 3.0, 5.0, 9.0)
	@test isapprox(h(5.0), 5.0)
	@test_throws ErrorException h(nextfloat(5.0))
	@test_throws ErrorException h(9.0)
	@test isapprox(h(nextfloat(9.0)), 1.0)	
end
