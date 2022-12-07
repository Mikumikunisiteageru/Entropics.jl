module Entropics

export maxendist, median, mean, var, entropy, PDF, CDF

using NLsolve: nlsolve, converged

using SpecialFunctions: erf, erfi, erfinv

# MATHEMATIC FUNCTIONS

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
	return T(x * evalpoly(x^2, 
		[3.3333333333333333e-01, 2.2222222222222223e-02,
		 2.1164021164021165e-03, 2.1164021164021165e-04,
		 2.1377799155576935e-05, 2.1644042808063972e-06,
		 2.1925947851873778e-07, 2.2214608789979678e-08,
		 2.2507846516808994e-09, 2.2805151204592183e-10,
		 2.3106432599002627e-11, 2.3411706819824886e-12,
		 2.3721017400233653e-13, 2.4034415333307705e-14,
		 2.4351954029183367e-15, 2.4673688045172075e-16,
		 2.4999672771220810e-17, 2.5329964357406350e-18,
		 2.5664619702826290e-19, 2.6003696460137274e-20]))
end
cothminv(x::Real) = cothminv(float(x))

abstract type MaxEnDist end

struct PDF{T<:MaxEnDist} <: Function
	d::T
end

struct CDF{T<:MaxEnDist} <: Function
	d::T
end

###

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

###

struct MED100{T<:AbstractFloat} <: MaxEnDist
	a::T
	b::T
	m::T
end
MED100(a::AbstractFloat, b::AbstractFloat, m::AbstractFloat) = 
	MED100(promote(a, b, m)...)
MED100(a::Real, b::Real, m::Real) = MED100(float(a), float(b), float(m))
median(d::MED100)  = m
mean(d::MED100)    = (a + 2m + b) / 4
var(d::MED100)     = (5 * (d.b-d.a)^2 + 4 * (d.m-d.a) * (d.m-d.b)) / 48
entropy(d::MED100) = log(4 * (d.m-d.a) * (d.b-d.m)) / 2
(p::PDF{MED100{T}})(x::Real) where {T<:AbstractFloat} = 
	p.d.a <= x <= p.d.m ? inv(p.d.m-p.d.a)/2 : 
	p.d.m <  x <= p.d.b ? inv(p.d.b-p.d.m)/2 : zero(T)
(P::CDF{MED100{T}})(x::Real) where {T<:AbstractFloat} = 
	P.d.m <  x <= P.d.b ? (T(x)-P.d.m) / (2 * (P.d.b-P.d.m)) + one(T)/2 : 
	P.d.a <= x <= P.d.m ? (T(x)-P.d.a) / (2 * (P.d.m-P.d.a)) : T(x > P.d.b)
med(a::Real, b::Real, m::Real, ::Nothing, ::Nothing) = MED100(a, b, m)

###

struct MED010{T<:AbstractFloat} <: MaxEnDist
	a::T
	b::T
	k::T
	g::T
	function MED010(a::T, b::T, g::T) where {T<:AbstractFloat}
		k = g / (exp(b*g) - exp(a*g))
		new{T}(a, b, k, g)
	end
end
MED010(a::AbstractFloat, b::AbstractFloat, g::AbstractFloat) = 
	MED010(promote(a, b, g)...)
MED010(a::Real, b::Real, g::Real) = MED010(float(a), float(b), float(g))
median(d::MED010) = log(exp(d.a * d.g) + d.g / (2 * d.k)) / d.g
mean(d::MED010) = 
	(d.k * (d.b * exp(d.b * d.g) - d.a * exp(d.a * d.g)) - 1) / d.g
var(d::MED010) = -(2 / d.g + mean(d)) * mean(d) + 
	d.k / d.g * (d.b^2 * exp(d.b * d.g) - d.a^2 * exp(d.a * d.g))
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
	return isapprox(g, 0, atol=1e-10) ? MED000(a, b) : MED010(a, b, g)
end

###

struct MED110{T<:AbstractFloat} <: MaxEnDist
	a::T
	b::T
	m::T
	ka::T
	kb::T
	g::T
	function MED110(a::T, b::T, m::T, g::T) where {T<:AbstractFloat}
		ka = g / (2 * (exp(m*g) - exp(a*g)))
		kb = g / (2 * (exp(b*g) - exp(m*g)))
		new{T}(a, b, m, ka, kb, g)
	end
end
MED110(a::AbstractFloat, b::AbstractFloat, m::AbstractFloat, 
	g::AbstractFloat) = MED110(promote(a, b, m, g)...)
MED110(a::Real, b::Real, m::Real, g::Real) = 
	MED110(float(a), float(b), float(m), float(g))
median(d::MED110) = d.m
mean(d::MED110) = 
	(d.kb * (d.b * exp(d.b * d.g) - d.m * exp(d.m * d.g)) + 
	 d.ka * (d.m * exp(d.m * d.g) - d.a * exp(d.a * d.g)) - 1) / d.g
var(d::MED110) = -(2 / d.g + mean(d)) * mean(d) + 
	(d.kb * (d.b^2 * exp(d.b * d.g) - d.m^2 * exp(d.m * d.g)) + 
	 d.ka * (d.m^2 * exp(d.m * d.g) - d.a^2 * exp(d.a * d.g))) / d.g
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
	return isapprox(g, 0, atol=1e-10) ? MED100(a, b, m) : MED110(a, b, m, g)
end

###

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
		k = 2 / (sqrt(pi) * s * (erf((b-o)/s) - erf((a-o)/s)))
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
median(d::MED011N) = 
	d.o + d.s * erfinv(erf((d.a-d.o)/d.s) + 1/(sqrt(pi)*d.k*d.s))
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
		ea = exp(((a - o) / s) ^ 2)
		eb = exp(((b - o) / s) ^ 2)
		khalf = k / 2
		t2 = khalf * (eb - ea)
		t3 = khalf * (b * eb - a * ea)
		mu = o - s^2 * t2
		new{T}(a, b, k, o, s, t2, t3, mu)
	end
end
MED011P(a::AbstractFloat, b::AbstractFloat, 
	o::AbstractFloat, s::AbstractFloat) = MED011P(promote(a, b, o, s)...)
MED011P(a::Real, b::Real, o::Real, s::Real) = 
	MED011P(float(a), float(b), float(o), float(s))
median(d::MED011P) = 
	d.o + d.s * erfiinv(erfi((d.a-d.o)/d.s) + 1/(sqrt(pi)*d.k*d.s))
mean(d::MED011P) =  d.mu
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
			d = MED011N(a, b, params[1], params[2])
			outputs[1] = mean(d) - u
			outputs[2] = var(d) - v
		end
		solution = nlsolve(f011n!, [u, sqrt(2*v)], ftol=1e-12)
		converged(solution) || @warn "Low precision!"
		return MED011N(a, b, solution.zero...)
	else # v > var_010
		function f011p!(outputs, params)
			d = MED011P(a, b, params[1], params[2])
			outputs[1] = mean(d) - u
			outputs[2] = var(d) - v
		end
		solution = nlsolve(f011p!, [a+b-u, sqrt(2*v)], ftol=1e-12)
		converged(solution) || @warn "Low precision!"
		return MED011P(a, b, solution.zero...)
	end
end

###

med(a::Real, b::Real, ::Nothing, ::Nothing, v::Real) = 
	med(a, b, nothing, (a+b)/2, v)

###

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
		ka = 1 / (sqrt(pi) * s * (erf((m-o)/s) - erf((a-o)/s)))
		kb = 1 / (sqrt(pi) * s * (erf((b-o)/s) - erf((m-o)/s)))
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
		solution = nlsolve(f111n!, [u, sqrt(2*v)], ftol=1e-12)
		converged(solution) || @warn "Low precision!"
		return MED111N(a, b, m, solution.zero...)
	else # v > var_110
		function f111p!(outputs, params)
			d = MED111P(a, b, m, params[1], params[2])
			outputs[1] = mean(d) - u
			outputs[2] = var(d) - v
		end
		solution = nlsolve(f111p!, [a+b-u, sqrt(2*v)], ftol=1e-12)
		converged(solution) || @warn "Low precision!"
		return MED111P(a, b, m, solution.zero...)
	end
end

###

function xinterval(a::Real, b::Real)
	(isnan(a) || isnan(b)) && throw(ArgumentError(
		"The endpoints of the interval should be both real numbers!"))
	(isinf(a) || isinf(b)) && throw(ArgumentError(
		"Only finite intervals are acceptable!"))
	a == b && throw(ArgumentError("The interval has zero length!"))
	a > b && throw(ArgumentError("The interval has reversed endpoints!"))
	return true
end

function maxendist(a::Real, b::Real; 
		median::Union{Real,Nothing}=nothing, mean::Union{Real,Nothing}=nothing, 
		std::Union{Real,Nothing}=nothing, var::Union{Real,Nothing}=nothing)
	xinterval(a, b)
	return med(a, b, median, mean, var)
end

###

end # module Entropics
