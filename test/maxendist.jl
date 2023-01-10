# test/maxendist.jl

using Entropics
using Test

@testset "MaxEnDist" begin
	MaxEnDist = Entropics.MaxEnDist
	Bounded = Entropics.Bounded
	Distribution = Entropics.Distribution
	@test isabstracttype(MaxEnDist)
	@test MaxEnDist <: Bounded
	@test MaxEnDist <: Distribution
	@test MaxEnDist{Float64} <: Bounded{Float64}
	@test MaxEnDist{Float64} <: Distribution{Float64}
end

@testset "MED0" begin
	MED0 = Entropics.MED0
	MaxEnDist = Entropics.MaxEnDist
	Bounded = Entropics.Bounded
	Distribution = Entropics.Distribution
	@test isabstracttype(MED0)
	@test MED0 <: MaxEnDist
	@test MED0 <: Bounded
	@test MED0 <: Distribution
	@test MED0{Float64} <: MaxEnDist{Float64}
	@test MED0{Float64} <: Bounded{Float64}
	@test MED0{Float64} <: Distribution{Float64}
end

@testset "MED000" begin
	MED000 = Entropics.MED000
	MED0 = Entropics.MED0
	MaxEnDist = Entropics.MaxEnDist
	Bounded = Entropics.Bounded
	Distribution = Entropics.Distribution
	@test ! (isabstracttype(MED000))
	@test ! (isconcretetype(MED000))
	@test isconcretetype(MED000{Float64})
	@test MED000 <: MED0
	@test MED000 <: MaxEnDist
	@test MED000 <: Bounded
	@test MED000 <: Distribution
	@test MED000{Float64} <: MED0{Float64}
	@test MED000{Float64} <: MaxEnDist{Float64}
	@test MED000{Float64} <: Bounded{Float64}
	@test MED000{Float64} <: Distribution{Float64}
end

@testset "MED000 instance" begin
	MED000 = Entropics.MED000
	A = Entropics.A
	B = Entropics.B
	C = Entropics.C
	d = maxendist(3, 9)
	@test isa(d, MED000{Float64})
	@test isapprox(d.a, 3.0)
	@test isapprox(d.b, 9.0)
	@test isapprox(A(d), 0.0)
	@test isapprox(B(d), 0.0)
	@test isapprox(C(d), -log(6.0))
	@test isapprox(median(d), 6.0)
	@test isapprox(mean(d), 6.0)
	@test isapprox(moment2(d), 39.0)
	@test isapprox(var(d), 3.0)
	@test isapprox(std(d), sqrt(3.0))
	@test isapprox(entropy(d), log(6.0))
	p = pdf(d)
	@test isapprox(p(prevfloat(3.0)), 0.0)
	@test isapprox(p(3), 1/6)
	@test isapprox(p(6), 1/6)
	@test isapprox(p(9), 1/6)
	@test isapprox(p(nextfloat(9.0)), 0.0)
	P = cdf(d)
	@test isapprox(P(prevfloat(3.0)), 0.0)
	@test isapprox(P(3), 0.0)
	@test isapprox(P(6), 0.5)
	@test isapprox(P(9), 1.0)
	@test isapprox(P(nextfloat(9.0)), 1.0)
	q(p) = quantile(d, p)
	@test_throws ErrorException q(prevfloat(0.0))
	@test isapprox(q(0.00), 3.0)
	@test isapprox(q(0.25), 4.5)
	@test isapprox(q(0.50), 6.0)
	@test isapprox(q(0.75), 7.5)
	@test isapprox(q(1.00), 9.0)
	@test_throws ErrorException q(nextfloat(1.0))
end

@testset "MED010" begin
	MED010 = Entropics.MED010
	MED0 = Entropics.MED0
	MaxEnDist = Entropics.MaxEnDist
	Bounded = Entropics.Bounded
	Distribution = Entropics.Distribution
	@test ! (isabstracttype(MED010))
	@test ! (isconcretetype(MED010))
	@test isconcretetype(MED010{Float64})
	@test MED010 <: MED0
	@test MED010 <: MaxEnDist
	@test MED010 <: Bounded
	@test MED010 <: Distribution
	@test MED010{Float64} <: MED0{Float64}
	@test MED010{Float64} <: MaxEnDist{Float64}
	@test MED010{Float64} <: Bounded{Float64}
	@test MED010{Float64} <: Distribution{Float64}
end

@testset "MED010 instance" begin
	MED010 = Entropics.MED010
	A = Entropics.A
	B = Entropics.B
	C = Entropics.C
	d0 = MED010(3, 9, -0.5)
	u0 = mean(d0)
	@test isapprox(u0, (5*exp(3)-11) / (exp(3)-1))
	d = maxendist(3, 9; mean=u0)
	@test isa(d, MED010{Float64})
	@test isapprox(d.a, 3.0)
	@test isapprox(d.b, 9.0)
	@test isapprox(A(d), 0.0)
	@test isapprox(B(d), -0.5)
	@test isapprox(C(d), -log(2*(exp(-3/2)-exp(-9/2))))
	@test isapprox(median(d), 9 - 2 * log(2\(1+exp(3))))
	@test isapprox(mean(d), u0)
	@test isapprox(moment2(d), (29*exp(3)-125) / (exp(3)-1))
	@test isapprox(var(d), 4 * (exp(6)-11*exp(3)+1) / (exp(3)-1)^2)
	@test isapprox(std(d), 2 * sqrt(exp(6)-11*exp(3)+1) / (exp(3)-1))
	@test isapprox(entropy(d), log(2*expm1(3)) - 2 - 3/expm1(3))
	p = pdf(d)
	@test isapprox(p(prevfloat(3.0)), 0.0)
	@test isapprox(p(3), exp(3) / (2*expm1(3)))
	@test isapprox(p(6), exp(3/2) / (2*expm1(3)))
	@test isapprox(p(9), 1 / (2*expm1(3)))
	@test isapprox(p(nextfloat(9.0)), 0.0)
	P = cdf(d)
	@test isapprox(P(prevfloat(3.0)), 0.0)
	@test isapprox(P(3), 0.0)
	@test isapprox(P(6), 1/(1+exp(-3/2)))
	@test isapprox(P(9), 1.0)
	@test isapprox(P(nextfloat(9.0)), 1.0)
	q(p) = quantile(d, p)
	@test_throws ErrorException q(prevfloat(0.0))
	@test isapprox(q(0.00), 3.0)
	@test isapprox(q(0.25), 9 - 2 * log(4\(1+3*exp(3))))
	@test isapprox(q(0.50), 9 - 2 * log(2\(1+exp(3))))
	@test isapprox(q(0.75), 9 - 2 * log(4\(3+exp(3))))
	@test isapprox(q(1.00), 9.0)
	@test_throws ErrorException q(nextfloat(1.0))
end

@testset "MED011" begin
	MED011 = Entropics.MED011
	MED0 = Entropics.MED0
	MaxEnDist = Entropics.MaxEnDist
	Bounded = Entropics.Bounded
	Distribution = Entropics.Distribution
	@test ! (isabstracttype(MED011))
	@test ! (isconcretetype(MED011))
	@test isconcretetype(MED011{Float64})
	@test MED011 <: MED0
	@test MED011 <: MaxEnDist
	@test MED011 <: Bounded
	@test MED011 <: Distribution
	@test MED011{Float64} <: MED0{Float64}
	@test MED011{Float64} <: MaxEnDist{Float64}
	@test MED011{Float64} <: Bounded{Float64}
	@test MED011{Float64} <: Distribution{Float64}
end

@testset "MED011 instance N" begin
	MED011 = Entropics.MED011
	A = Entropics.A
	B = Entropics.B
	C = Entropics.C
	d0 = MED011(3, 9, -1, 10)
	u0 = mean(d0)
	v0 = var(d0)
	@test isapprox(u0, 5.005178827223492)
	@test isapprox(v0, 0.4896153343815399)
	d = maxendist(3, 9; mean=u0, var=v0)
	@test isa(d, MED011{Float64})
	@test isapprox(d.a, 3.0)
	@test isapprox(d.b, 9.0)
	@test isapprox(A(d), -1.0)
	@test isapprox(B(d), 10.0)
	@test isapprox(C(d), -25.57002332828383)
	@test isapprox(median(d), 5.002072763482002)
	@test isapprox(mean(d), u0)
	@test isapprox(moment2(d), u0^2 + v0)
	@test isapprox(var(d), v0)
	@test isapprox(std(d), sqrt(v0))
	@test isapprox(entropy(d), 1.0596654829176266)
	p = pdf(d)
	@test isapprox(p(prevfloat(3.0)), 0.0)
	@test isapprox(p(3), 0.010357718087000891)
	@test isapprox(p(6), 0.20804032907640074)
	@test isapprox(p(9), 6.364001942246248e-8)
	@test isapprox(p(nextfloat(9.0)), 0.0)
	P = cdf(d)
	@test isapprox(P(prevfloat(3.0)), 0.0)
	@test isapprox(P(3), 0.0)
	@test isapprox(P(6), 0.9211660213459609)
	@test isapprox(P(9), 1.0)
	@test isapprox(P(nextfloat(9.0)), 1.0)
	q(p) = quantile(d, p)
	@test_throws ErrorException q(prevfloat(0.0))
	@test isapprox(q(0.00), 3.0)
	@test isapprox(q(0.25), 4.5269597734813765)
	@test isapprox(q(0.50), 5.002072763482002)
	@test isapprox(q(0.75), 5.47823816633966)
	@test isapprox(q(1.00), 9.0)
	@test_throws ErrorException q(nextfloat(1.0))
end

@testset "MED011 instance P" begin
	MED011 = Entropics.MED011
	A = Entropics.A
	B = Entropics.B
	C = Entropics.C
	d0 = MED011(3, 9, 0.225, -3)
	u0 = mean(d0)
	v0 = var(d0)
	@test isapprox(u0, 4.707179319117533)
	@test isapprox(v0, 3.4512403668586735)
	d = maxendist(3, 9; mean=u0, var=v0)
	@test isa(d, MED011{Float64})
	@test isapprox(d.a, 3.0)
	@test isapprox(d.b, 9.0)
	@test isapprox(A(d), 0.225)
	@test isapprox(B(d), -3.0)
	@test isapprox(C(d), 7.029856753616281)
	@test isapprox(median(d), 3.8460355413307896)
	@test isapprox(mean(d), u0)
	@test isapprox(moment2(d), u0^2 + v0)
	@test isapprox(var(d), v0)
	@test isapprox(std(d), sqrt(v0))
	@test isapprox(entropy(d), 1.3297062641693405)
	p = pdf(d)
	@test isapprox(p(prevfloat(3.0)), 0.0)
	@test isapprox(p(3), 1.0563892798925478)
	@test isapprox(p(6), 0.056690805245344575)
	@test isapprox(p(9), 0.1746199734954389)
	@test isapprox(p(nextfloat(9.0)), 0.0)
	P = cdf(d)
	@test isapprox(P(prevfloat(3.0)), 0.0)
	@test isapprox(P(3), 0.0)
	@test isapprox(P(6), 0.7713150936555976)
	@test isapprox(P(9), 1.0)
	@test isapprox(P(nextfloat(9.0)), 1.0)
	q(p) = quantile(d, p)
	@test_throws ErrorException q(prevfloat(0.0))
	@test isapprox(q(0.00), 3.0)
	@test isapprox(q(0.25), 3.297784717730531)
	@test isapprox(q(0.50), 3.8460355413307896)
	@test isapprox(q(0.75), 5.646985522561653)
	@test isapprox(q(1.00), 9.0)
	@test_throws ErrorException q(nextfloat(1.0))
end

@testset "Case 001" begin
	MED011 = Entropics.MED011
	d0 = maxendist(3, 9; var=1)
	@test isa(d0, MED011)
	m0 = mean(d0)
	@test isapprox(median(d0), 6.0)
	@test isapprox(mean(d0), 6.0)
	@test isapprox(var(d0), 1.0)
	@test isapprox(entropy(d0), 1.4160330613834269)
	d = maxendist(3, 9; mean=m0*(1-1e-6), var=1)
	@test isa(d, MED011)
	@test entropy(d) < entropy(d0) 
	d = maxendist(3, 9; mean=m0*(1+1e-6), var=1)
	@test isa(d, MED011)
	@test entropy(d) < entropy(d0) 
	d0 = maxendist(3, 9; var=5)
	@test isa(d0, MED011)
	m0 = mean(d0)
	@test isapprox(median(d0), 6.0)
	@test isapprox(mean(d0), 6.0)
	@test isapprox(var(d0), 5.0)
	@test isapprox(entropy(d0), 1.53824230733611969)
	d = maxendist(3, 9; mean=m0*(1-1e-6), var=5)
	@test isa(d, MED011)
	@test entropy(d) < entropy(d0) 
	d = maxendist(3, 9; mean=m0*(1+1e-6), var=5)
	@test isa(d, MED011)
	@test entropy(d) < entropy(d0) 
end

@testset "MED1" begin
	MED1 = Entropics.MED1
	MaxEnDist = Entropics.MaxEnDist
	Bounded = Entropics.Bounded
	Distribution = Entropics.Distribution
	@test isabstracttype(MED1)
	@test MED1 <: MaxEnDist
	@test MED1 <: Bounded
	@test MED1 <: Distribution
	@test MED1{Float64} <: MaxEnDist{Float64}
	@test MED1{Float64} <: Bounded{Float64}
	@test MED1{Float64} <: Distribution{Float64}
end

@testset "MED100" begin
	MED100 = Entropics.MED100
	MED1 = Entropics.MED1
	MaxEnDist = Entropics.MaxEnDist
	Bounded = Entropics.Bounded
	Distribution = Entropics.Distribution
	@test ! (isabstracttype(MED100))
	@test ! (isconcretetype(MED100))
	@test isconcretetype(MED100{Float64})
	@test MED100 <: MED1
	@test MED100 <: MaxEnDist
	@test MED100 <: Bounded
	@test MED100 <: Distribution
	@test MED100{Float64} <: MED1{Float64}
	@test MED100{Float64} <: MaxEnDist{Float64}
	@test MED100{Float64} <: Bounded{Float64}
	@test MED100{Float64} <: Distribution{Float64}
end

@testset "MED100 instance" begin
	MED100 = Entropics.MED100
	A = Entropics.A
	B = Entropics.B
	Ca = Entropics.Ca
	Cb = Entropics.Cb
	d = maxendist(3, 9; median=5)
	@test isa(d, MED100{Float64})
	@test isapprox(d.a, 3.0)
	@test isapprox(d.b, 9.0)
	@test isapprox(A(d), 0.0)
	@test isapprox(B(d), 0.0)
	@test isapprox(Ca(d), -log(4.0))
	@test isapprox(Cb(d), -log(8.0))
	@test isapprox(median(d), 5.0)
	@test isapprox(mean(d), 5.5)
	@test isapprox(moment2(d), 100/3)
	@test isapprox(var(d), 37/12)
	@test isapprox(std(d), sqrt(37/12))
	@test isapprox(entropy(d), log(32.0)/2)
	p = pdf(d)
	@test isapprox(p(prevfloat(3.0)), 0.0)
	@test isapprox(p(3), 1/4)
	@test isapprox(p(5), 1/4)
	@test isapprox(p(nextfloat(5.0)), 1/8)
	@test isapprox(p(9), 1/8)
	@test isapprox(p(nextfloat(9.0)), 0.0)
	P = cdf(d)
	@test isapprox(P(prevfloat(3.0)), 0.0)
	@test isapprox(P(3), 0.0)
	@test isapprox(P(4), 0.25)
	@test isapprox(P(5), 0.5)
	@test isapprox(P(7), 0.75)
	@test isapprox(P(9), 1.0)
	@test isapprox(P(nextfloat(9.0)), 1.0)
	q(p) = quantile(d, p)
	@test_throws ErrorException q(prevfloat(0.0))
	@test isapprox(q(0.00), 3.0)
	@test isapprox(q(0.25), 4.0)
	@test isapprox(q(0.50), 5.0)
	@test isapprox(q(0.75), 7.0)
	@test isapprox(q(1.00), 9.0)
	@test_throws ErrorException q(nextfloat(1.0))
end

@testset "MED110" begin
	MED110 = Entropics.MED110
	MED1 = Entropics.MED1
	MaxEnDist = Entropics.MaxEnDist
	Bounded = Entropics.Bounded
	Distribution = Entropics.Distribution
	@test ! (isabstracttype(MED110))
	@test ! (isconcretetype(MED110))
	@test isconcretetype(MED110{Float64})
	@test MED110 <: MED1
	@test MED110 <: MaxEnDist
	@test MED110 <: Bounded
	@test MED110 <: Distribution
	@test MED110{Float64} <: MED1{Float64}
	@test MED110{Float64} <: MaxEnDist{Float64}
	@test MED110{Float64} <: Bounded{Float64}
	@test MED110{Float64} <: Distribution{Float64}
end

@testset "MED110 instance" begin
	MED110 = Entropics.MED110
	A = Entropics.A
	B = Entropics.B
	Ca = Entropics.Ca
	Cb = Entropics.Cb
	d0 = MED110(3, 9, 5, -0.5)
	u0 = mean(d0)
	@test isapprox(u0, 5.104988007631342)
	d = maxendist(3, 9; median=5, mean=u0)
	@test isa(d, MED110{Float64})
	@test isapprox(d.a, 3.0)
	@test isapprox(d.b, 9.0)
	@test isapprox(A(d), 0.0)
	@test isapprox(B(d), -0.5)
	@test isapprox(Ca(d), 0.5723807842671909)
	@test isapprox(Cb(d), 1.259119096748968)
	@test isapprox(median(d), 5.0)
	@test isapprox(mean(d), u0)
	@test isapprox(moment2(d), 28.381644378580116)
	@test isapprox(var(d), 2.3207418205202934)
	@test isapprox(std(d), 1.5233981162257926)
	@test isapprox(entropy(d), 1.6367440633075914)
	p = pdf(d)
	@test isapprox(p(prevfloat(3.0)), 0.0)
	@test isapprox(p(3), 0.39549417671733156)
	@test isapprox(p(5), 0.1454941767173316)
	@test isapprox(p(nextfloat(5.0)), 0.2891294106874163)
	@test isapprox(p(9), 0.03912941068741643)
	@test isapprox(p(nextfloat(9.0)), 0.0)
	P = cdf(d)
	@test isapprox(P(prevfloat(3.0)), 0.0)
	@test isapprox(P(3), 0.0)
	@test isapprox(P(4), 0.3112296656009273)
	@test isapprox(P(5), 0.5)
	@test isapprox(P(7), 0.8655292893150025)
	@test isapprox(P(9), 1.0)
	@test isapprox(P(nextfloat(9.0)), 1.0)
	q(p) = quantile(d, p)
	@test_throws ErrorException q(prevfloat(0.0))
	@test isapprox(q(0.00), 3.0)
	@test isapprox(q(0.25), 3.759770986083445)
	@test isapprox(q(0.50), 5.0)
	@test isapprox(q(0.75), 6.132438339033944)
	@test isapprox(q(1.00), 9.0)
	@test_throws ErrorException q(nextfloat(1.0))
end

@testset "MED111" begin
	MED111 = Entropics.MED111
	MED1 = Entropics.MED1
	MaxEnDist = Entropics.MaxEnDist
	Bounded = Entropics.Bounded
	Distribution = Entropics.Distribution
	@test ! (isabstracttype(MED111))
	@test ! (isconcretetype(MED111))
	@test isconcretetype(MED111{Float64})
	@test MED111 <: MED1
	@test MED111 <: MaxEnDist
	@test MED111 <: Bounded
	@test MED111 <: Distribution
	@test MED111{Float64} <: MED1{Float64}
	@test MED111{Float64} <: MaxEnDist{Float64}
	@test MED111{Float64} <: Bounded{Float64}
	@test MED111{Float64} <: Distribution{Float64}
end

@testset "MED111 instance N" begin
	MED111 = Entropics.MED111
	A = Entropics.A
	B = Entropics.B
	Ca = Entropics.Ca
	Cb = Entropics.Cb
	d0 = MED111(3, 9, 4.5, -1, 10)
	u0 = mean(d0)
	v0 = var(d0)
	@test isapprox(u0, 4.692680344102959)
	@test isapprox(v0, 0.5428922188225762)
	d = maxendist(3, 9; median=4.5, mean=u0, var=v0)
	@test isa(d, MED111{Float64})
	@test isapprox(d.a, 3.0)
	@test isapprox(d.b, 9.0)
	@test isapprox(d.m, 4.5)
	@test isapprox(A(d), -1.0)
	@test isapprox(B(d), 10.0)
	@test isapprox(Ca(d), -24.827550476379287)
	@test isapprox(Cb(d), -25.99140408056066)
	@test isapprox(median(d), 4.5)
	@test isapprox(mean(d), u0)
	@test isapprox(moment2(d), u0^2 + v0)
	@test isapprox(var(d), v0)
	@test isapprox(std(d), sqrt(v0))
	@test isapprox(entropy(d), 1.0468148681932394)
	p = pdf(d)
	@test isapprox(p(prevfloat(3.0)), 0.0)
	@test isapprox(p(3), 0.021762859029986183)
	@test isapprox(p(4.5), 0.925380313370109)
	@test isapprox(p(nextfloat(4.5)), 0.2889781843027494)
	@test isapprox(p(9), 4.175677678736231e-8)
	@test isapprox(p(nextfloat(9.0)), 0.0)
	P = cdf(d)
	@test isapprox(P(prevfloat(3.0)), 0.0)
	@test isapprox(P(3), 0.0)
	@test isapprox(P(3.75), 0.07626234431431854)
	@test isapprox(P(4.5), 0.5)
	@test isapprox(P(7.25), 0.9995190064775423)
	@test isapprox(P(9), 1.0)
	@test isapprox(P(nextfloat(9.0)), 1.0)
	q(p) = quantile(d, p)
	@test_throws ErrorException q(prevfloat(0.0))
	@test isapprox(q(0.00), 3.0)
	@test isapprox(q(0.25), 4.172839600786583)
	@test isapprox(q(0.50), 4.5)
	@test isapprox(q(0.75), 5.215775458615824)
	@test isapprox(q(1.00), 9.0)
	@test_throws ErrorException q(nextfloat(1.0))
end

@testset "MED111 instance P" begin
	MED111 = Entropics.MED111
	A = Entropics.A
	B = Entropics.B
	Ca = Entropics.Ca
	Cb = Entropics.Cb
	d0 = MED111(3, 9, 4, 0.225, -3)
	u0 = mean(d0)
	v0 = var(d0)
	@test isapprox(u0, 4.831872892364895)
	@test isapprox(v0, 3.588535286180793)
	d = maxendist(3, 9; median=4, mean=u0, var=v0)
	@test isa(d, MED111{Float64})
	@test isapprox(d.a, 3.0)
	@test isapprox(d.b, 9.0)
	@test isapprox(d.m, 4.0)
	@test isapprox(A(d), 0.225)
	@test isapprox(B(d), -3.0)
	@test isapprox(Ca(d), 6.94727307604397)
	@test isapprox(Cb(d), 7.11987944293971)
	@test isapprox(median(d), 4.0)
	@test isapprox(mean(d), u0)
	@test isapprox(moment2(d), u0^2 + v0)
	@test isapprox(var(d), v0)
	@test isapprox(std(d), sqrt(v0))
	@test isapprox(entropy(d), 1.4015479574187668)
	p = pdf(d)
	@test isapprox(p(prevfloat(3.0)), 0.0)
	@test isapprox(p(3), 0.9726539390286925)
	@test isapprox(p(4), 0.23393150410934196)
	@test isapprox(p(nextfloat(4.0)), 0.27800378311400903)
	@test isapprox(p(9), 0.19106901959758912)
	@test isapprox(p(nextfloat(9.0)), 0.0)
	P = cdf(d)
	@test isapprox(P(prevfloat(3.0)), 0.0)
	@test isapprox(P(3), 0.0)
	@test isapprox(P(3.5), 0.33620968546078717)
	@test isapprox(P(4), 0.5)
	@test isapprox(P(6.5), 0.7791032674010244)
	@test isapprox(P(9), 1.0)
	@test isapprox(P(nextfloat(9.0)), 1.0)
	q(p) = quantile(d, p)
	@test_throws ErrorException q(prevfloat(0.0))
	@test isapprox(q(0.00), 3.0)
	@test isapprox(q(0.25), 3.3312602153443667)
	@test isapprox(q(0.50), 4.0)
	@test isapprox(q(0.75), 6.003658576726803)
	@test isapprox(q(1.00), 9.0)
	@test_throws ErrorException q(nextfloat(1.0))
end

@testset "Case 101" begin
	MED111 = Entropics.MED111
	urange = Entropics.urange
	ul, ur = urange(3, 9, 5, 1)
	@test isapprox(ul, 4.0)
	@test isapprox(ur, 6.0)
	d0 = maxendist(3, 9; median=5, var=1)
	@test isa(d0, MED111)
	m0 = mean(d0)
	@test isapprox(median(d0), 5.0)
	@test isapprox(mean(d0), 5.073292946985342)
	@test isapprox(var(d0), 1.0)
	@test isapprox(entropy(d0), 1.3930453412416295)
	d = maxendist(3, 9; median=5, mean=m0*(1-1e-6), var=1)
	@test isa(d, MED111)
	@test entropy(d) < entropy(d0) 
	d = maxendist(3, 9; median=5, mean=m0*(1+1e-6), var=1)
	@test isa(d, MED111)
	@test entropy(d) < entropy(d0)
	ul, ur = urange(3, 9, 5, 4)
	@test isapprox(ul, 4.550510257216822)
	@test isapprox(ur, 7.0)
	d0 = maxendist(3, 9; median=5, var=4)
	@test isa(d0, MED111)
	m0 = mean(d0)
	@test isapprox(median(d0), 5.0)
	@test isapprox(mean(d0), 5.6516285432080355)
	@test isapprox(var(d0), 4.0)
	@test isapprox(entropy(d0), 1.682962249369854)
	d = maxendist(3, 9; median=5, mean=m0*(1-1e-6), var=4)
	@test isa(d, MED111)
	@test entropy(d) < entropy(d0) 
	d = maxendist(3, 9; median=5, mean=m0*(1+1e-6), var=4)
	@test isa(d, MED111)
	@test entropy(d) < entropy(d0) 
end

@testset "Case 000 to 100" begin
	MED000 = Entropics.MED000
	MED100 = Entropics.MED100
	d0 = maxendist(3, 9)
	m0, u0, v0, h0 = (d0,) .|> (median, mean, var, entropy)
	@test isa(d0, MED000)
	d = maxendist(3, 9; median=prevfloat(m0))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED100)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
	d = maxendist(3, 9; median=nextfloat(m0))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED100)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
end

@testset "Case 000 to 010" begin
	MED000 = Entropics.MED000
	MED010 = Entropics.MED010
	d0 = maxendist(3, 9)
	m0, u0, v0, h0 = (d0,) .|> (median, mean, var, entropy)
	@test isa(d0, MED000)
	d = maxendist(3, 9; mean=prevfloat(u0))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED010)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
	d = maxendist(3, 9; mean=nextfloat(u0))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED010)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
end

@testset "Case 000 to 001" begin
	MED000 = Entropics.MED000
	MED011 = Entropics.MED011
	d0 = maxendist(3, 9)
	m0, u0, v0, h0 = (d0,) .|> (median, mean, var, entropy)
	@test isa(d0, MED000)
	d = maxendist(3, 9; var=v0*(1-3e-5))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED011)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0; rtol=4e-5)
	@test isapprox(h, h0)
	d = maxendist(3, 9; var=v0*(1+3e-5))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED011)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0; rtol=4e-5)
	@test isapprox(h, h0)
end

@testset "Case 100 to 110" begin
	MED100 = Entropics.MED100
	MED110 = Entropics.MED110
	d0 = maxendist(3, 9; median=5)
	m0, u0, v0, h0 = (d0,) .|> (median, mean, var, entropy)
	@test isa(d0, MED100)
	d = maxendist(3, 9; median=5, mean=prevfloat(u0))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED110)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
	d = maxendist(3, 9; median=5, mean=nextfloat(u0))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED110)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
end

@testset "Case 100 to 101" begin
	MED100 = Entropics.MED100
	MED111 = Entropics.MED111
	d0 = maxendist(3, 9; median=5)
	m0, u0, v0, h0 = (d0,) .|> (median, mean, var, entropy)
	@test isa(d0, MED100)
	d = maxendist(3, 9; median=5, var=v0*(1-3e-4))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED111)
	@test isapprox(m, m0)
	@test isapprox(u, u0; rtol=1e-4)
	@test isapprox(v, v0; rtol=3e-4)
	@test isapprox(h, h0; rtol=3e-7)
	d = maxendist(3, 9; median=5, var=v0*(1+3e-3))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED111)
	@test isapprox(m, m0)
	@test isapprox(u, u0; rtol=1e-3)
	@test isapprox(v, v0; rtol=3e-3)
	@test isapprox(h, h0; rtol=1e-5)
end

@testset "Case 010 to 110" begin
	MED010 = Entropics.MED010
	MED110 = Entropics.MED110
	d0 = maxendist(3, 9; mean=5)
	m0, u0, v0, h0 = (d0,) .|> (median, mean, var, entropy)
	@test isa(d0, MED010)
	d = maxendist(3, 9; median=prevfloat(m0), mean=5)
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED110)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
	d = maxendist(3, 9; median=nextfloat(m0), mean=5)
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED110)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
end

@testset "Case 010 to 011" begin
	MED010 = Entropics.MED010
	MED011 = Entropics.MED011
	d0 = maxendist(3, 9; mean=5)
	m0, u0, v0, h0 = (d0,) .|> (median, mean, var, entropy)
	@test isa(d0, MED010)
	d = maxendist(3, 9; mean=5, var=v0*(1-3e-3))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED011)
	@test isapprox(m, m0; rtol=3e-3)
	@test isapprox(u, u0)
	@test isapprox(v, v0; rtol=4e-3)
	@test isapprox(h, h0; rtol=1e-4)
	d = maxendist(3, 9; mean=5, var=v0*(1+3e-3))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED011)
	@test isapprox(m, m0; rtol=3e-3)
	@test isapprox(u, u0)
	@test isapprox(v, v0; rtol=4e-3)
	@test isapprox(h, h0; rtol=1e-4)
end

@testset "Case 001 to 101" begin
	MED011 = Entropics.MED011
	MED111 = Entropics.MED111
	d0 = maxendist(3, 9; var=1)
	m0, u0, v0, h0 = (d0,) .|> (median, mean, var, entropy)
	@test isa(d0, MED011)
	d = maxendist(3, 9; median=prevfloat(m0), var=1)
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED111)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
	d = maxendist(3, 9; median=nextfloat(m0), var=1)
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED111)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
end

@testset "Case 001 to 011" begin
	MED011 = Entropics.MED011
	d0 = maxendist(3, 9; var=1)
	m0, u0, v0, h0 = (d0,) .|> (median, mean, var, entropy)
	@test isa(d0, MED011)
	d = maxendist(3, 9; mean=prevfloat(u0), var=1)
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED011)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
	d = maxendist(3, 9; mean=nextfloat(u0), var=1)
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED011)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
end

@testset "Case 110 to 111" begin
	MED110 = Entropics.MED110
	MED111 = Entropics.MED111
	d0 = maxendist(3, 9; median=4.5, mean=5)
	m0, u0, v0, h0 = (d0,) .|> (median, mean, var, entropy)
	@test isa(d0, MED110)
	d = maxendist(3, 9; median=4.5, mean=5, var=v0*(1-3e-3))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED111)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0; rtol=4e-3)
	@test isapprox(h, h0; rtol=1e-4)
	d = maxendist(3, 9; median=4.5, mean=5, var=v0*(1+3e-3))
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED111)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0; rtol=4e-3)
	@test isapprox(h, h0; rtol=1e-4)
end

@testset "Case 101 to 111" begin
	MED111 = Entropics.MED111
	d0 = maxendist(3, 9; median=4.5, var=1)
	m0, u0, v0, h0 = (d0,) .|> (median, mean, var, entropy)
	@test isa(d0, MED111)
	d = maxendist(3, 9; median=4.5, mean=prevfloat(u0), var=1)
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED111)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
	d = maxendist(3, 9; median=4.5, mean=nextfloat(u0), var=1)
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED111)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
end

@testset "Case 011 to 111" begin
	MED011 = Entropics.MED011
	MED111 = Entropics.MED111
	d0 = maxendist(3, 9; mean=4.5, var=1)
	m0, u0, v0, h0 = (d0,) .|> (median, mean, var, entropy)
	@test isa(d0, MED011)
	d = maxendist(3, 9; median=prevfloat(m0), mean=4.5, var=1)
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED111)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
	d = maxendist(3, 9; median=nextfloat(m0), mean=4.5, var=1)
	m, u, v, h = (d,) .|> (median, mean, var, entropy)
	@test isa(d, MED111)
	@test isapprox(m, m0)
	@test isapprox(u, u0)
	@test isapprox(v, v0)
	@test isapprox(h, h0)
end
