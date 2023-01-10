# test/sample.jl

using Entropics
using Test

@testset "Sample" begin
	Sample = Entropics.Sample
	Distribution = Entropics.Distribution
	Bounded = Entropics.Bounded
	@test isconcretetype(Sample{Float64})
	@test ! (isconcretetype(Sample))
	@test ! (isabstracttype(Sample))
	@test Sample{Float64} <: Sample
	@test Sample{Float64} <: Distribution{Float64}
	@test ! (Sample{Float64} <: Sample{AbstractFloat})
	@test ! (Sample{Float64} <: Bounded{Float64})
	@test_throws TypeError Sample{Int}
	s = Sample([1, 2, 3, 4])
	@test isa(s, Sample{Float64})
	@test all((===).(s.v, [1.0, 2.0, 3.0, 4.0]))
	s = Sample([3, 2, 1, 4])
	@test all((===).(s.v, [1.0, 2.0, 3.0, 4.0]))
end

@testset "sample" begin
	Sample = Entropics.Sample
	@test all((===).(sample([3, 9]).v, Sample([3, 9]).v))
	@test all((===).(sample([3, 9]).v, Sample([3.0, 9.0]).v))
end

@testset "support" begin
	@test support(sample([3, 6, 9])) == (3.0, 9.0)
	@test support(sample([9, 6, 3])) == (3.0, 9.0)
end

@testset "pdf" begin
	p = pdf(sample([3, 9]))
	@test isinf(p(3.0))
	@test iszero(p(6.0))
	@test isinf(p(9.0))
	@test isinf(p(3))
	@test iszero(p(6))
	@test isinf(p(9))
end

@testset "cdf" begin
	P = cdf(sample([3, 9]))
	@test isapprox(P(2.0), 0.0)
	@test isapprox(P(3.0), 0.25)
	@test isapprox(P(4.0), 0.5)
	@test isapprox(P(8.0), 0.5)
	@test isapprox(P(9.0), 0.75)
	@test isapprox(P(10.0), 1.0)
	@test isapprox(P(0), 0.0)
	@test isapprox(P(3), 0.25)
	@test isapprox(P(6), 0.5)
	@test isapprox(P(9), 0.75)
	@test isapprox(P(12), 1.0)
end

@testset "quantile" begin
	q1(p) = quantile(sample([3, 9]), p)
	@test_throws ErrorException q1(prevfloat(0.0))
	@test isapprox(3.0, q1(0.0))
	@test isapprox(3.0, q1(nextfloat(0.0)))
	@test isapprox(3.0, q1(0.25))
	@test isapprox(3.0, q1(prevfloat(0.5, 3)))
	@test isapprox(6.0, q1(0.5))
	@test isapprox(9.0, q1(nextfloat(0.5, 3)))
	@test isapprox(9.0, q1(0.75))
	@test isapprox(9.0, q1(prevfloat(1.0)))
	@test isapprox(9.0, q1(1.0))
	@test_throws ErrorException q1(nextfloat(1.0))
	q2(p) = quantile(sample([3, 5, 9]), p)
	@test_throws ErrorException q2(prevfloat(0.0))
	@test isapprox(3.0, q2(0.0))
	@test isapprox(3.0, q2(nextfloat(0.0)))
	@test isapprox(3.0, q2(prevfloat(1/3, 4)))
	@test isapprox(4.0, q2(prevfloat(1/3, 1)))
	@test isapprox(4.0, q2(1/3))
	@test isapprox(4.0, q2(nextfloat(1/3, 1)))
	@test isapprox(5.0, q2(nextfloat(1/3, 4)))
	@test isapprox(5.0, q2(0.5))
	@test isapprox(5.0, q2(prevfloat(2/3, 4)))
	@test isapprox(7.0, q2(prevfloat(2/3, 1)))
	@test isapprox(7.0, q2(2/3))
	@test isapprox(7.0, q2(nextfloat(2/3, 1)))
	@test isapprox(9.0, q2(nextfloat(2/3, 4)))
	@test isapprox(9.0, q2(prevfloat(1.0)))
	@test isapprox(9.0, q2(1.0))
	@test_throws ErrorException q2(nextfloat(1.0))
end

@testset "median" begin
	@test isapprox(6.0, median(sample([3, 9])))
	@test isapprox(5.0, median(sample([3, 5, 9])))
end

@testset "mean" begin
	@test isapprox(6.0, mean(sample([3, 9])))
	@test isapprox(17/3, mean(sample([3, 5, 9])))
end

@testset "moment2" begin
	@test isapprox(45.0, moment2(sample([3, 9])))
	@test isapprox(115/3, moment2(sample([3, 5, 9])))
end

@testset "var" begin
	@test isapprox(9.0, var(sample([3, 9])))
	@test isapprox(56/9, var(sample([3, 5, 9])))
	@test isapprox(56/9, var(sample([3, 5, 9]), u=mean(sample([3, 5, 9]))))
end

@testset "std" begin
	@test isapprox(sqrt(9.0), std(sample([3, 9])))
	@test isapprox(sqrt(56/9), std(sample([3, 5, 9])))
	@test isapprox(sqrt(56/9), 
		std(sample([3, 5, 9]), u=mean(sample([3, 5, 9]))))
end

@testset "entropy" begin
	@test isapprox(-Inf, entropy(sample([3, 9])))
	@test isapprox(-Inf, entropy(sample([3, 5, 9])))
end

@testset "bound" begin
	s = sample([3, 2, 1, 4])
	@test all((===).(bound(s, 0, 5).v, [1.0, 2.0, 3.0, 4.0]))
	@test all((===).(bound(s, 1, 4).v, [1.0, 2.0, 3.0, 4.0]))
	@test all((===).(bound(s, 1, 3).v, [1.0, 2.0, 3.0]))
	@test all((===).(bound(s, 2, 3).v, [2.0, 3.0]))
	@test all((===).(bound(s, 2.5, 3).v, [3.0]))
	@test all((===).(bound(s, 3, 3).v, [3.0]))
	@test_throws ErrorException bound(s, 3.5, 3)
	@test_throws ErrorException bound(s, 4.1, 4.2)
end

@testset "helrot" begin
	helrot = Entropics.helrot
	s = sample(collect(-10:10))
	@test isapprox(helrot(s; useiqr=true) , 3.4888283569469922)
	@test isapprox(helrot(s; useiqr=false), 3.4888283569469922)
	s = sample(collect((-10:10).^3 ./ 100))
	@test isapprox(helrot(s; useiqr=true) , 1.0677720461913982)
	@test isapprox(helrot(s; useiqr=false), 2.5009605388351015)
end

@testset "smooth" begin
	v = collect((-10:10).^3 ./ 100)
	s = sample(v)
	d0 = smooth(s)
	d = smooth(v)
	@test d0.d.v == d.d.v
	@test isapprox(d0.h, d.h)
	@test isapprox(median(d), 0.0)
	@test isapprox(mean(d), 0.0)
	@test isapprox(moment2(d), 395681/21000 + 1.0677720461913982 ^ 2)
	@test isapprox(var(d), 395681/21000 + 1.0677720461913982 ^ 2)
	@test isapprox(std(d), sqrt(395681/21000 + 1.0677720461913982 ^ 2))
	# @test_skip entropy(d)
	p = pdf(d)
	@test isapprox(p(-Inf), 0.0)
	@test isapprox(p(-10), 0.018502360464860357)
	@test isapprox(p(-5), 0.02606195681795083)
	@test isapprox(p(-1), 0.13317541405161373)
	@test isapprox(p(0), 0.1757885884031584)
	@test isapprox(p(1), 0.1331754140516137)
	@test isapprox(p(5), 0.02606195681795083)
	@test isapprox(p(10), 0.018502360464860357)
	@test isapprox(p(Inf), 0.0)
	P = cdf(d)
	@test isapprox(P(-Inf), 0.0)
	@test isapprox(P(-10), 0.024075091262040993)
	@test isapprox(P(-5), 0.12398328443098922)
	@test isapprox(P(-1), 0.3395197656610898)
	@test isapprox(P(0), 0.5)
	@test isapprox(P(1), 0.6604802343389102)
	@test isapprox(P(5), 0.8760167155690108)
	@test isapprox(P(10), 0.9759249087379589)
	@test isapprox(P(Inf), 1.0)
	q(p) = quantile(d, p)
	@test_throws ErrorException q(prevfloat(0.00))
	@test q(0.00) < -15
	@test isapprox(q(0.25), -1.8641445700011823)
	@test isapprox(q(0.50), 0.0)
	@test isapprox(q(0.75), 1.8641445700011816)
	@test q(1.00) > 15
	@test_throws ErrorException q(nextfloat(1.00))
end

@testset "smooth then bound" begin
	integrate = Entropics.integrate
	v = collect((-10:10).^3 ./ 100)
	s = smooth(v)
	b = bound(s)
	@test support(b) == (-10.0, 10.0)
	@test isapprox(integrate(x -> pdf(b)(x), support(b)...), 1.0)
	@test isapprox(mean(b), 0.0; atol=1e-15)
	@test isapprox(moment2(b), 15.020912499505704)
	@test isapprox(var(b), 15.020912499505704)
	@test isapprox(std(b), 3.875682197949892)
	@test isapprox(entropy(b), 2.6474740912679264)
	p = pdf(b)
	@test isapprox(p(prevfloat(-10.0)), 0.0)
	@test isapprox(p(-10.0), 0.019438319076347854)
	@test isapprox(p(-5.0), 0.027380324437169107)
	@test isapprox(p(0.0), 0.1846810128821671)
	@test isapprox(p(5.0), 0.027380324437169107)
	@test isapprox(p(10.0), 0.019438319076347854)
	@test isapprox(p(nextfloat(10.0)), 0.0)
	P = cdf(b)
	@test isapprox(P(prevfloat(-10.0)), 0.0)
	@test isapprox(P(-10.0), 0.0)
	@test isapprox(P(-5.0), 0.10496213933609955)
	@test isapprox(P(0.0), 0.5)
	@test isapprox(P(5.0), 0.8950378606639006)
	@test isapprox(P(10.0), 1.0)
	@test isapprox(P(nextfloat(10.0)), 1.0)
	q(p) = quantile(b, p)
	@test_throws ErrorException q(prevfloat(0.00))
	@test isapprox(q(0.00), -10.0)
	@test isapprox(q(0.25), -1.7151546408202658)
	@test isapprox(q(0.50), 0.0; atol=1e-15)
	@test isapprox(q(0.75), 1.7151546408202651)
	@test isapprox(q(1.00), 10.0)
	@test_throws ErrorException q(nextfloat(1.00))	
end
