# test/math.jl

using Entropics
using Test

using SpecialFunctions: erf, erfi, erfc

@testset "binaryroot" begin
	binaryroot = Entropics.binaryroot
	Pab = Entropics.Pab
	f(x) = cos(x) - 0.39
	@test_throws ErrorException binaryroot(f, 0.0, 1.0)
	@test isapprox(cos(binaryroot(f, 1.0, 2.0)), 0.39)
	@test_throws ErrorException binaryroot(f, 2.0, 3.0)
	function g(x)
		h(t) = erfi(t) - x
		@test isapprox(erfi(binaryroot(h, 0.0, sqrt(log(1 + x)))), x)
	end
	g.([0.01, 0.1, 1])
	P(x) = Pab(sin, x, 0.0, pi/2)
	function phi(p)
		r(x) = P(x) - p
		isapprox(sin(binaryroot(r, 0.0, pi/2)), p)
	end
	@test_throws ErrorException phi(0.0 - 1e-15)
	@test phi(0.0)
	@test phi(0.2)
	@test phi(0.5)
	@test phi(0.8)
	@test phi(1.0)
	@test_throws ErrorException phi(1.0 + 1e-15)
end

@testset "secantroot" begin
	secantroot = Entropics.secantroot
	Pab = Entropics.Pab
	f(x) = cos(x) - 0.39
	@test isapprox(cos(secantroot(f, 0.0, 1.0)), 0.39)
	@test isapprox(cos(secantroot(f, 1.0, 2.0)), 0.39)
	@test isapprox(cos(secantroot(f, 2.0, 3.0)), 0.39)
	function g(x)
		h(t) = erfi(t) - x
		@test isapprox(erfi(secantroot(h, 0.0, sqrt(log(1 + x)))), x)
	end
	g.([0.01, 0.1, 1, 5, 10, 50])
	P(x) = Pab(sin, x, 0.0, pi/2)
	function phi(p)
		r(x) = P(x) - p
		isapprox(sin(secantroot(r, 0.0, pi/2)), p)
	end
	@test_throws ErrorException phi(0.0 - 1e-15)
	@test phi(0.0)
	@test phi(0.2)
	@test phi(0.5)
	@test phi(0.8)
	@test phi(1.0)
	@test_throws ErrorException phi(1.0 + 1e-15)
end

@testset "integrate" begin
	integrate = Entropics.integrate
	@test isapprox(integrate(one, 3.0, 9.0), 6.0)
	@test isapprox(integrate(identity, 3.0, 9.0), 36.0)
	@test isapprox(integrate(x -> x^2, 3.0, 9.0), 234.0)
	@test isapprox(integrate(x -> x^3, 3.0, 9.0), 1620.0)
	@test isapprox(integrate(inv, 3.0, 9.0), log(9.0) - log(3.0))
	@test isapprox(integrate(cos, 3.0, 9.0), sin(9.0) - sin(3.0))
end

@testset "xlog" begin
	xlog = Entropics.xlog
	@test isapprox(xlog(0.0), 0.0)
	@test isapprox(xlog(1.0), 0.0)
	@test isapprox(xlog(0.5), - log(2.0) / 2)
	@test isapprox(xlog(2.0), + log(2.0) * 2)
	@test_throws DomainError xlog(-1.0)
end

@testset "erfiinv" begin
	erfiinv = Entropics.erfiinv
	for i = Float64[-10, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 10, 100, 1000]
		@test isapprox(i, erfi(erfiinv(i)))
	end
	for i = Float32[-10, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 10, 100, 1000]
		@test isapprox(i, erfi(erfiinv(i)))
	end
end

@testset "cothminv" begin
	cothminv = Entropics.cothminv
	@test cothminv(-Inf) == -1.0
	@test cothminv(0.00) ==  0.0
	@test cothminv(+Inf) == +1.0
	@test isapprox(cothminv(nextfloat(1.18)), 0.3610722575251245)
	@test isapprox(cothminv(prevfloat(1.18)), 0.3610722575251243)
	for i = 1 : -1 : -9
		x = 10.0 ^ i
		bx = BigFloat(x)
		@test isapprox(cothminv(x), oftype(x, coth(bx) - inv(bx)))
		# @test false
	end
end

@testset "diffexp" begin
	diffexp = Entropics.diffexp
	function f(x::Float64, y::Float64)
		bx, by = BigFloat(x), BigFloat(y)
		@test isapprox(diffexp(x, y), Float64(exp(bx) - exp(by)))
	end
	f(0.0, 1.0)
	f(0.0, -1.0)
	f(1.0, 0.0)
	f(-1.0, 0.0)
	f(0.0, 50.0)
	f(0.0, -50.0)
	f(50.0, 0.0)
	f(-50.0, 0.0)
	f(1e-10, 1.1e-10)
	f(1.1e-10, 1e-10)
end

@testset "diffxexp" begin
	diffxexp = Entropics.diffxexp
	function f(x::Float64, y::Float64)
		bx, by = BigFloat(x), BigFloat(y)
		@test isapprox(diffxexp(x, y), Float64(bx*exp(bx) - by*exp(by)))
	end
	f(0.0, 1.0)
	f(0.0, -1.0)
	f(1.0, 0.0)
	f(-1.0, 0.0)
	f(0.0, 10.0)
	f(0.0, -10.0)
	f(10.0, 0.0)
	f(-10.0, 0.0)
	f(1e-10, 1.1e-10)
	f(1.1e-10, 1e-10)
end

@testset "diffxxexp" begin
	diffxxexp = Entropics.diffxxexp
	function f(x::Float64, y::Float64)
		bx, by = BigFloat(x), BigFloat(y)
		@test isapprox(diffxxexp(x, y), Float64(bx^2*exp(bx) - by^2*exp(by)))
	end
	f(0.0, 1.0)
	f(0.0, -1.0)
	f(1.0, 0.0)
	f(-1.0, 0.0)
	f(0.0, 20.0)
	f(0.0, -20.0)
	f(20.0, 0.0)
	f(-20.0, 0.0)
	f(1e-10, 1.1e-10)
	f(1.1e-10, 1e-10)
end

@testset "diffxexpf" begin
	diffxexpf = Entropics.diffxexpf
	function f(x::Float64, y::Float64)
		bx, by = BigFloat(x), BigFloat(y)
		@test isapprox(diffxexpf(t->t^2, x, y), 
					   Float64(bx*exp(bx^2) - by*exp(by^2)))
		@test isapprox(diffxexpf(t->-t^2, x, y), 
					   Float64(bx*exp(-bx^2) - by*exp(-by^2)))
	end
	f(0.0, 1.0)
	f(0.0, -1.0)
	f(1.0, 0.0)
	f(-1.0, 0.0)
	f(0.0, 4.0)
	f(0.0, -4.0)
	f(4.0, 0.0)
	f(-4.0, 0.0)
	f(1e-10, 1.1e-10)
	f(1.1e-10, 1e-10)
end

@testset "differf" begin
	differf = Entropics.differf
	function f(x::Float64, y::Float64)
		bx, by = BigFloat(x), BigFloat(y)
		@test isapprox(differf(x, y), Float64(erf(bx) - erf(by)))
	end
	f(4.0, 5.0)
	f(4.0, -5.0)
	f(-4.0, 5.0)
	f(-4.0, -5.0)
	f(5.0, 4.0)
	f(5.0, -4.0)
	f(-5.0, 4.0)
	f(-5.0, -4.0)
	f(1e-10, 1.1e-10)
	f(1.1e-10, 1e-10)
end

@testset "intexp" begin
	integrate = Entropics.integrate
	intexp = Entropics.intexp
	function f(A, B, C, a, b)
		@test isapprox(intexp(A, B, C, a, b), 
					   integrate(x -> exp(A*x^2 + B*x + C), a, b))
	end
	f(0.00, 0.00, 1.33, 0.3, 0.9)
	f(0.00, -2.14, 1.33, 0.3, 0.9)
	f(0.89, -2.14, 1.33, 0.3, 0.9)
	f(-0.89, -2.14, 1.33, 0.3, 0.9)
end

@testset "intxexp" begin
	integrate = Entropics.integrate
	intxexp = Entropics.intxexp
	function f(A, B, C, a, b)
		@test isapprox(intxexp(A, B, C, a, b), 
					   integrate(x -> x * exp(A*x^2 + B*x + C), a, b))
	end
	f(0.00, 0.00, 1.33, 0.3, 0.9)
	f(0.00, -2.14, 1.33, 0.3, 0.9)
	f(0.89, -2.14, 1.33, 0.3, 0.9)
	f(-0.89, -2.14, 1.33, 0.3, 0.9)
end

@testset "intxxexp" begin
	integrate = Entropics.integrate
	intxxexp = Entropics.intxxexp
	function f(A, B, C, a, b)
		@test isapprox(intxxexp(A, B, C, a, b), 
					   integrate(x -> x^2 * exp(A*x^2 + B*x + C), a, b))
	end
	f(0.00, 0.00, 1.33, 0.3, 0.9)
	f(0.00, -2.14, 1.33, 0.3, 0.9)
	f(0.89, -2.14, 1.33, 0.3, 0.9)
	f(-0.89, -2.14, 1.33, 0.3, 0.9)
end

@testset "invintexp" begin
	integrate = Entropics.integrate
	invintexp = Entropics.invintexp
	intexp = Entropics.intexp
	function f(A, B, C, a, p)
		b = invintexp(A, B, C, a, p)
		@test isapprox(p, intexp(A, B, C, a, b))
	end
	f(0.00, 0.00, 1.33, -0.3, 0.9)
	f(0.00, -2.14, 1.33, -0.3, 0.9)
	f(0.89, -2.14, 1.33, -0.3, 0.9)
	f(-0.89, -2.14, 1.33, -0.3, 0.9)
end

@testset "intqexp" begin
	integrate = Entropics.integrate
	intqexp = Entropics.intqexp
	function f(A, B, C, a, b)
		q(x) = A * x^2 + B * x + C
		@test isapprox(intqexp(A, B, C, a, b), 
					   - integrate(x -> q(x) * exp(q(x)), a, b))
	end
	f(0.00, 0.00, 1.33, 0.3, 0.9)
	f(0.00, -2.14, 1.33, 0.3, 0.9)
	f(0.89, -2.14, 1.33, 0.3, 0.9)
	f(-0.89, -2.14, 1.33, 0.3, 0.9)
end
