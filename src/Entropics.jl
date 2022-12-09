module Entropics

export maxendist, median, mean, var, entropy, PDF, CDF

using Base.Math: @horner

using NLsolve: nlsolve, converged

using SpecialFunctions # erf, erfc, erfi, erfinv

# MATHEMATIC FUNCTIONS

function SpecialFunctions.erfinv(x::BigFloat; start=zero(BigFloat))
	function f!(output, param)
		output[1] = erf(param[1]) - x
	end
	solution = nlsolve(f!, BigFloat[start], ftol=min(1e-16, abs(1-x)))
	converged(solution) || @warn "Low precision!"
	return solution.zero[1]
end

function erfiinv(x::T) where {T<:AbstractFloat}
	iszero(x) && return x
	x < 0 && return -erfiinv(-x)
	function f!(output, param)
		output[1] = erfi(param[1]) - x
	end
	return nlsolve(f!, [sqrt(log(1 + x))]).zero[1]
end
erfiinv(x::Real) = erfiinv(float(x))

function cothminv(x::T) where {T<:AbstractFloat}
	abs(x) > 0.5 && return coth(x) - inv(x)
	return T(x * @horner(x^2, 
		3.3333333333333333e-01, 2.2222222222222223e-02,
		2.1164021164021165e-03, 2.1164021164021165e-04,
		2.1377799155576935e-05, 2.1644042808063972e-06,
		2.1925947851873778e-07, 2.2214608789979678e-08,
		2.2507846516808994e-09, 2.2805151204592183e-10,
		2.3106432599002627e-11, 2.3411706819824886e-12,
		2.3721017400233653e-13, 2.4034415333307705e-14,
		2.4351954029183367e-15, 2.4673688045172075e-16,
		2.4999672771220810e-17, 2.5329964357406350e-18,
		2.5664619702826290e-19, 2.6003696460137274e-20))
end
cothminv(x::Real) = cothminv(float(x))

function differf(r::T, l::T) where {T<:AbstractFloat}
	if xor(signbit(r), signbit(l)) # different signs
		return erf(r) - erf(l)
	elseif signbit(r) # both negative
		return erfc(-r) - erfc(-l)
	else # both positive
		return erfc(l) - erfc(r)
	end
end
differf(r::AbstractFloat, l::AbstractFloat) = differf(promote(r, l)...)
differf(r::Real, l::Real) = differf(float(r), float(l))

### TYPES

abstract type MaxEnDist end

struct PDF{T<:MaxEnDist} <: Function
	d::T
end

struct CDF{T<:MaxEnDist} <: Function
	d::T
end

### KNOWN NOTHING

struct MED000{T<:AbstractFloat} <: MaxEnDist
	a::T
	b::T
end
MED000(a::AbstractFloat, b::AbstractFloat) = MED000(promote(a, b)...)
MED000(a::Real, b::Real) = MED000(float(a), float(b))
median(d::MED000)  = (d.a + d.b) / 2
mean(d::MED000)    = (d.a + d.b) / 2
var(d::MED000)     = (d.b - d.a)^2 / 12
entropy(d::MED000) = log(d.b - d.a)
(p::PDF{MED000{T}})(x::Real) where {T<:AbstractFloat} = 
	p.d.a <= x <= p.d.b ? inv(p.d.b-p.d.a) : zero(T)
(P::CDF{MED000{T}})(x::Real) where {T<:AbstractFloat} = 
	P.d.a <= x <= P.d.b ? (T(x)-P.d.a) / (P.d.b-P.d.a) : T(x > P.d.b)

med(a::Real, b::Real, ::Nothing, ::Nothing, ::Nothing) = MED000(a, b)

### KNOWN MEDIAN

struct MED100{T<:AbstractFloat} <: MaxEnDist
	a::T
	b::T
	m::T
end
MED100(a::AbstractFloat, b::AbstractFloat, m::AbstractFloat) = 
	MED100(promote(a, b, m)...)
MED100(a::Real, b::Real, m::Real) = MED100(float(a), float(b), float(m))
median(d::MED100)  = d.m
mean(d::MED100)    = (d.a + 2 * d.m + d.b) / 4
var(d::MED100)     = (5 * (d.b-d.a)^2 + 4 * (d.m-d.a) * (d.m-d.b)) / 48
entropy(d::MED100) = log(4 * (d.m-d.a) * (d.b-d.m)) / 2
(p::PDF{MED100{T}})(x::Real) where {T<:AbstractFloat} = 
	p.d.a <= x <= p.d.m ? inv(p.d.m-p.d.a)/2 : 
	p.d.m <  x <= p.d.b ? inv(p.d.b-p.d.m)/2 : zero(T)
(P::CDF{MED100{T}})(x::Real) where {T<:AbstractFloat} = 
	P.d.m <  x <= P.d.b ? (T(x)-P.d.m) / (2 * (P.d.b-P.d.m)) + one(T)/2 : 
	P.d.a <= x <= P.d.m ? (T(x)-P.d.a) / (2 * (P.d.m-P.d.a)) : T(x > P.d.b)

med(a::Real, b::Real, m::Real, ::Nothing, ::Nothing) = MED100(a, b, m)

### KNOWN MEAN

struct MED010{T<:AbstractFloat} <: MaxEnDist
	a::T
	b::T
	k::T
	g::T
	function MED010(a::T, b::T, g::T) where {T<:AbstractFloat}
		k = g / (exp(a*g) * expm1((b-a)*g))
		new{T}(a, b, k, g)
	end
end
MED010(a::AbstractFloat, b::AbstractFloat, g::AbstractFloat) = 
	MED010(promote(a, b, g)...)
MED010(a::Real, b::Real, g::Real) = MED010(float(a), float(b), float(g))
median(d::MED010) = abs(d.g) < 1 ? 
	log1p(expm1(d.a * d.g) + d.g / (2 * d.k)) / d.g : 
	log(exp(d.a * d.g) + d.g / (2 * d.k)) / d.g
mean(d::MED010) = 
	(d.a + d.b + (d.b-d.a) * cothminv((d.b-d.a)/2 * d.g)) / 2
function var(d::MED010)
	x = (d.b-d.a)/2 * d.g
	cmix = cothminv(x)
	return - mean(d)^2 + 
		((d.b^2+d.a^2) + (d.b^2-d.a^2) * cmix - (d.b-d.a)^2 * (cmix / x)) / 2
end
entropy(d::MED010) = -log(d.k) - d.g * mean(d)
(p::PDF{MED010{T}})(x::Real) where {T<:AbstractFloat} = 
	p.d.a <= x <= p.d.b ? p.d.k * exp(p.d.g * T(x)) : zero(T)
(P::CDF{MED010{T}})(x::Real) where {T<:AbstractFloat} = 
	P.d.a <= x <= P.d.b ? 
		P.d.k / P.d.g * (exp(T(x) * P.d.g) - exp(P.d.a * P.d.g)) : T(x > P.d.b)

function med(a::Real, b::Real, ::Nothing, u::Real, ::Nothing)
	t = u - (a + b) / 2
	d = (b - a) / 2
	function f!(output, param)
		output[1] = d * cothminv(d * param[1]) - t
	end
	solution = nlsolve(f!, [3 * t / d^2], ftol=1e-12)
	converged(solution) || @warn "Low precision!"
	g = solution.zero[1]
	return iszero(g) ? MED000(a, b) : MED010(a, b, g)
end

### KNOWN MEDIAN AND MEAN

struct MED110{T<:AbstractFloat} <: MaxEnDist
	a::T
	b::T
	m::T
	ka::T
	kb::T
	g::T
	function MED110(a::T, b::T, m::T, g::T) where {T<:AbstractFloat}
		ka = g / (2 * exp(a*g) * expm1((m-a)*g))
		kb = g / (2 * exp(m*g) * expm1((b-m)*g))
		new{T}(a, b, m, ka, kb, g)
	end
end
MED110(a::AbstractFloat, b::AbstractFloat, m::AbstractFloat, 
	g::AbstractFloat) = MED110(promote(a, b, m, g)...)
MED110(a::Real, b::Real, m::Real, g::Real) = 
	MED110(float(a), float(b), float(m), float(g))
median(d::MED110) = d.m
mean(d::MED110) = (d.a + 2 * d.m + d.b + 
	(d.m - d.a) * cothminv((d.m-d.a) * d.g / 2) + 
	(d.b - d.m) * cothminv((d.b-d.m) * d.g / 2)) / 4
function var(d::MED110)
	xa = (d.m-d.a)/2 * d.g
	xb = (d.b-d.m)/2 * d.g
	cmixa = cothminv(xa)
	cmixb = cothminv(xb)
	return - mean(d)^2 + 4 \ 
		((d.m^2+d.a^2) + (d.m^2-d.a^2) * cmixa - (d.m-d.a)^2 * (cmixa / xa) +
		 (d.b^2+d.m^2) + (d.b^2-d.m^2) * cmixb - (d.b-d.m)^2 * (cmixb / xb))
end
entropy(d::MED110) = -(log(d.ka) + log(d.kb)) / 2 - d.g * mean(d)
(p::PDF{MED110{T}})(x::Real) where {T<:AbstractFloat} = 
	p.d.a <= x <= p.d.m ? p.d.ka * exp(p.d.g * T(x)) : 
	p.d.m <  x <= p.d.b ? p.d.kb * exp(p.d.g * T(x)) : zero(T)
(P::CDF{MED110{T}})(x::Real) where {T<:AbstractFloat} = 
	P.d.m <  x <= P.d.b ? 
		P.d.kb / P.d.g * (exp(T(x) * P.d.g) - exp(P.d.m * P.d.g)) + one(T)/2 : 
	P.d.a <= x <= P.d.m ? 
		P.d.ka / P.d.g * (exp(T(x) * P.d.g) - exp(P.d.a * P.d.g)) : 
			T(x > P.d.b)

function med(a::Real, b::Real, m::Real, u::Real, ::Nothing)
	t = 2 * u - m - (a + b) / 2
	da = (m - a) / 2
	db = (b - m) / 2
	function f!(output, param)
		output[1] = da * cothminv(da * param[1]) + 
			db * cothminv(db * param[1]) - t
	end
	solution = nlsolve(f!, [3 * t / (da^2 + db^2)], ftol=1e-12)
	converged(solution) || @warn "Low precision!"
	g = solution.zero[1]
	return iszero(g) ? MED100(a, b, m) : MED110(a, b, m, g)
end

### KNOWN MEAN AND VARIANCE

struct MED011N{T<:AbstractFloat} <: MaxEnDist
	a::T
	b::T
	k::T
	o::T
	s::T
	t2::T
	t3::T
	mu::T
	function MED011N(a::T, b::T, o::T, s::T) where {T<:AbstractFloat}
		k = 2 / (sqrt(pi) * s * differf((b-o)/s, (a-o)/s))
		ea = exp(- ((a - o) / s) ^ 2)
		eb = exp(- ((b - o) / s) ^ 2)
		khalf = k / 2
		t2 = khalf * (eb - ea)
		t3 = khalf * (b * eb - a * ea)
		mu = o - s^2 * t2
		new{T}(a, b, k, o, s, t2, t3, mu)
	end
end
MED011N(a::AbstractFloat, b::AbstractFloat, 
	o::AbstractFloat, s::AbstractFloat) = MED011N(promote(a, b, o, s)...)
MED011N(a::Real, b::Real, o::Real, s::Real) = 
	MED011N(float(a), float(b), float(o), float(s))
median(d::MED011N{T}) where {T<:AbstractFloat} = 
	d.o + d.s * T(erfinv(erf(BigFloat(d.a-d.o)/d.s) + 1/(sqrt(pi)*d.k*d.s)))
mean(d::MED011N) = d.mu
var(d::MED011N{T}) where {T<:AbstractFloat} = 
	d.s^2 * (one(T)/2 - d.t3) + (d.o - d.mu) * d.mu
entropy(d::MED011N) = (1 - 2*log(d.k)) / 2 + d.o * d.t2 - d.t3
(p::PDF{MED011N{T}})(x::Real) where {T<:AbstractFloat} = 
	p.d.a <= x <= p.d.b ? p.d.k * exp(-((T(x)-p.d.o)/p.d.s)^2) : zero(T)
(P::CDF{MED011N{T}})(x::Real) where {T<:AbstractFloat} = 
	P.d.a <= x <= P.d.b ? sqrt(pi)/2*P.d.k*P.d.s * 
		(erf((T(x)-P.d.o)/P.d.s) - erf((P.d.a-P.d.o)/P.d.s)) : T(x > P.d.b)

struct MED011P{T<:AbstractFloat} <: MaxEnDist
	a::T
	b::T
	k::T
	o::T
	s::T
	t2::T
	t3::T
	mu::T
	function MED011P(a::T, b::T, o::T, s::T) where {T<:AbstractFloat}
		k = 2 / (sqrt(pi) * s * (erfi((b-o)/s) - erfi((a-o)/s)))
		eam1 = expm1(((a - o) / s) ^ 2)
		ebm1 = expm1(((b - o) / s) ^ 2)
		khalf = k / 2
		t2 = khalf * (ebm1 - eam1)
		t3 = khalf * (b * ebm1 - a * eam1 + (b-a))
		mu = o + s^2 * t2
		new{T}(a, b, k, o, s, t2, t3, mu)
	end
end
MED011P(a::AbstractFloat, b::AbstractFloat, 
	o::AbstractFloat, s::AbstractFloat) = MED011P(promote(a, b, o, s)...)
MED011P(a::Real, b::Real, o::Real, s::Real) = 
	MED011P(float(a), float(b), float(o), float(s))
median(d::MED011P) = 
	d.o + d.s * erfiinv(erfi((d.a-d.o)/d.s) + 1/(sqrt(pi)*d.k*d.s))
mean(d::MED011P) = d.mu
var(d::MED011P{T}) where {T<:AbstractFloat} = 
	d.s^2 * (one(T)/-2 + d.t3) + (d.o - d.mu) * d.mu
entropy(d::MED011P) = (1 - 2*log(d.k)) / 2 + d.o * d.t2 - d.t3
(p::PDF{MED011P{T}})(x::Real) where {T<:AbstractFloat} = 
	p.d.a <= x <= p.d.b ? p.d.k * exp(((T(x)-p.d.o)/p.d.s)^2) : zero(T)
(P::CDF{MED011P{T}})(x::Real) where {T<:AbstractFloat} = 
	P.d.a <= x <= P.d.b ? sqrt(pi)/2*P.d.k*P.d.s * 
		(erfi((T(x)-P.d.o)/P.d.s) - erfi((P.d.a-P.d.o)/P.d.s)) : T(x > P.d.b)

function med(a::Real, b::Real, ::Nothing, u::Real, v::Real)
	med_010 = med(a, b, nothing, u, nothing)
	var_010 = var(med_010)
	if v == var_010
		return med_010
	elseif v < var_010
		function f011n!(outputs, params)
			d = MED011N(a, b, params...)
			outputs[1] = mean(d) - u
			outputs[2] = var(d) - v
		end
		solution = nlsolve(f011n!, [u, sqrt(2*v)], 
			method=:trust_region, ftol=1e-12, autoscale=false, factor=0.001)
		converged(solution) || @warn "Low precision!"
		return MED011N(a, b, solution.zero...)
	else # v > var_010
		function f011p!(outputs, params)
			d = MED011P(a, b, params...)
			outputs[1] = mean(d) - u
			outputs[2] = var(d) - v
		end
		solution = nlsolve(f011p!, [a+b-u, sqrt(2*v)], 
			method=:trust_region, ftol=1e-12, autoscale=false, factor=0.001)
		converged(solution) || @warn "Low precision!"
		return MED011P(a, b, solution.zero...)
	end
end

### KNOWN VARIANCE

med(a::Real, b::Real, ::Nothing, ::Nothing, v::Real) = 
	med(a, b, nothing, (a+b)/2, v)

### KNOWN MEDIAN, MEAN, AND VARIANCE

struct MED111N{T<:AbstractFloat} <: MaxEnDist
	a::T
	b::T
	m::T
	ka::T
	kb::T
	o::T
	s::T
	t2::T
	t3::T
	mu::T
	function MED111N(a::T, b::T, m::T, o::T, s::T) where {T<:AbstractFloat}
		ka = 1 / (sqrt(pi) * s * differf((m-o)/s, (a-o)/s))
		kb = 1 / (sqrt(pi) * s * differf((b-o)/s, (m-o)/s))
		ea = exp(- ((a - o) / s) ^ 2)
		em = exp(- ((m - o) / s) ^ 2)
		eb = exp(- ((b - o) / s) ^ 2)
		kbmka = kb - ka
		t2 = (kb * eb - kbmka * em - ka * ea) / 2
		t3 = (kb * b * eb - kbmka * m * em - ka * a * ea) / 2
		mu = o - s^2 * t2
		new{T}(a, b, m, ka, kb, o, s, t2, t3, mu)
	end
end
MED111N(a::AbstractFloat, b::AbstractFloat, 
	m::AbstractFloat, o::AbstractFloat, s::AbstractFloat) = 
		MED111N(promote(a, b, m, o, s)...)
MED111N(a::Real, b::Real, m::Real, o::Real, s::Real) = 
	MED111N(float(a), float(b), float(m), float(o), float(s))
median(d::MED111N) = d.m
mean(d::MED111N) = d.mu
var(d::MED111N{T}) where {T<:AbstractFloat} = 
	d.s^2 * (one(T)/2 - d.t3) + (d.o - d.mu) * d.mu
entropy(d::MED111N) = (1 - log(d.ka) - log(d.kb)) / 2 + d.o * d.t2 - d.t3
(p::PDF{MED111N{T}})(x::Real) where {T<:AbstractFloat} = 
	p.d.a <= x <= p.d.m ? p.d.ka * exp(-((T(x)-p.d.o)/p.d.s)^2) : 
	p.d.m <  x <= p.d.b ? p.d.kb * exp(-((T(x)-p.d.o)/p.d.s)^2) : zero(T)
(P::CDF{MED111N{T}})(x::Real) where {T<:AbstractFloat} = 
	P.d.m <  x <= P.d.b ? sqrt(pi)/2*P.d.kb*P.d.s * 
		(erf((T(x)-P.d.o)/P.d.s) - erf((P.d.m-P.d.o)/P.d.s)) + one(T)/2 : 
	P.d.a <= x <= P.d.m ? sqrt(pi)/2*P.d.ka*P.d.s * 
		(erf((T(x)-P.d.o)/P.d.s) - erf((P.d.a-P.d.o)/P.d.s)) : T(x > P.d.b)

struct MED111P{T<:AbstractFloat} <: MaxEnDist
	a::T
	b::T
	m::T
	ka::T
	kb::T
	o::T
	s::T
	t2::T
	t3::T
	mu::T
	function MED111P(a::T, b::T, m::T, o::T, s::T) where {T<:AbstractFloat}
		ka = 1 / (sqrt(pi) * s * (erfi((m-o)/s) - erfi((a-o)/s)))
		kb = 1 / (sqrt(pi) * s * (erfi((b-o)/s) - erfi((m-o)/s)))
		ea = exp(((a - o) / s) ^ 2)
		em = exp(((m - o) / s) ^ 2)
		eb = exp(((b - o) / s) ^ 2)
		kbmka = kb - ka
		t2 = (kb * eb - kbmka * em - ka * ea) / 2
		t3 = (kb * b * eb - kbmka * m * em - ka * a * ea) / 2
		mu = o + s^2 * t2
		new{T}(a, b, m, ka, kb, o, s, t2, t3, mu)
	end
end
MED111P(a::AbstractFloat, b::AbstractFloat, 
	m::AbstractFloat, o::AbstractFloat, s::AbstractFloat) = 
		MED111P(promote(a, b, m, o, s)...)
MED111P(a::Real, b::Real, m::Real, o::Real, s::Real) = 
	MED111P(float(a), float(b), float(m), float(o), float(s))
median(d::MED111P) = d.m
mean(d::MED111P) = d.mu
var(d::MED111P{T}) where {T<:AbstractFloat} = 
	d.s^2 * (one(T)/-2 + d.t3) + (d.o - d.mu) * d.mu
entropy(d::MED111P) = (1 - log(d.ka) - log(d.kb)) / 2 + d.o * d.t2 - d.t3
(p::PDF{MED111P{T}})(x::Real) where {T<:AbstractFloat} = 
	p.d.a <= x <= p.d.m ? p.d.ka * exp(((T(x)-p.d.o)/p.d.s)^2) : 
	p.d.m <  x <= p.d.b ? p.d.kb * exp(((T(x)-p.d.o)/p.d.s)^2) : zero(T)
(P::CDF{MED111P{T}})(x::Real) where {T<:AbstractFloat} = 
	P.d.m <  x <= P.d.b ? sqrt(pi)/2*P.d.kb*P.d.s * 
		(erfi((T(x)-P.d.o)/P.d.s) - erfi((P.d.m-P.d.o)/P.d.s)) + one(T)/2 : 
	P.d.a <= x <= P.d.m ? sqrt(pi)/2*P.d.ka*P.d.s * 
		(erfi((T(x)-P.d.o)/P.d.s) - erfi((P.d.a-P.d.o)/P.d.s)) : T(x > P.d.b)

function med(a::Real, b::Real, m::Real, u::Real, v::Real)
	med_110 = med(a, b, m, u, nothing)
	var_110 = var(med_110)
	if v == var_110
		return med_110
	elseif v < var_110
		function f111n!(outputs, params)
			d = MED111N(a, b, m, params[1], params[2])
			outputs[1] = mean(d) - u
			outputs[2] = var(d) - v
		end
		solution = nlsolve(f111n!, [u, sqrt(2*v)], 
			method=:trust_region, ftol=1e-12, autoscale=false, factor=0.001)
		converged(solution) || @warn "Low precision!"
		return MED111N(a, b, m, solution.zero...)
	else # v > var_110
		function f111p!(outputs, params)
			d = MED111P(a, b, m, params[1], params[2])
			outputs[1] = mean(d) - u
			outputs[2] = var(d) - v
		end
		solution = nlsolve(f111p!, [a+b-u, sqrt(2*v)], 
			method=:trust_region, ftol=1e-12, autoscale=false, factor=0.001)
		converged(solution) || @warn "Low precision!"
		return MED111P(a, b, m, solution.zero...)
	end
end

### KNOWN MEDIAN AND VARIANCE

function med(a::Real, b::Real, m::Real, ::Nothing, v::Real)
	## TO-DO ##
	error("Not yet implemented!")
	## TO-DO ##
end

### WRAPPER

function xmuv(a::Real, b::Real, m::Real, u::Real, v::Real)
	## TO-DO ##
end

function xbhatiadavis(a::Real, b::Real, u::Real, v::Real)
	vt = (u - a) * (b - u)
	v > vt && error("It is impossible to satisfy all these conditions!")
	v == vt && error("These conditions yield a two-point distribution!")
end
xbhatiadavis(a::Real, b::Real, ::Nothing, v::Real) = 
	xbhatiadavis(a, b, (a+b)/2, v)
xbhatiadavis(::Real, ::Real, ::Any, ::Nothing) = nothing

function xmean(a::Real, b::Real, u::Real)
	isnan(u) && error("The mean is not real!")
	a <= u <= b || error("The mean falls out of the support interval!")
	a < u < b || error("The mean coincides with an endpoint of the interval!")
end
xmean(::Real, ::Real, ::Nothing) = nothing

function xmedian(a::Real, b::Real, m::Real)
	isnan(m) && error("The median is not real!")
	a <= m <= b || error("The median falls out of the support interval!")
	a < m < b || 
		error("The median coincides with an endpoint of the interval!")
end
xmedian(::Real, ::Real, ::Nothing) = nothing

function xinterval(a::Real, b::Real)
	(isnan(a) || isnan(b)) && 
		error("Endpoint(s) of the interval is/are not real!")
	(isinf(a) || isinf(b)) && error("The interval is not bounded!")
	a == b && error("The interval with zero length is degenerated!")
	a > b && error("The interval is reversed!")
end

function xvar(v::Real)
	v < 0 && error("The variance is negative!")
	v == 0 && error("The variance is zero!")
end
xvar(::Nothing) = nothing

function xstd(s::Real)
	s < 0 && error("The standard deviation is negative!")
	s == 0 && error("The standard deviation is zero!")
end
xstd(::Nothing) = nothing

constdvar(s::Real, v::Real) = isapprox(s^2, v) ? (return v) : 
	error("The standard deviation and the variance are inconsistent!")
constdvar(::Nothing, v::Real) = v
constdvar(s::Real, ::Nothing) = s^2
constdvar(::Nothing, ::Nothing) = nothing

function maxendist(a::Real, b::Real; 
		median::Union{Real,Nothing}=nothing, mean::Union{Real,Nothing}=nothing, 
		std::Union{Real,Nothing}=nothing, var::Union{Real,Nothing}=nothing)
	xinterval(a, b)
	xmean(a, b, mean)
	xmedian(a, b, median)
	xstd(std)
	xvar(var)
	var = constdvar(std, var)
	xbhatiadavis(a, b, mean, var)
	return med(a, b, median, mean, var)
end

end # module Entropics
