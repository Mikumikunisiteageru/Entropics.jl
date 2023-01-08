# src/maxendist.jl

### GENERAL BOUNDED TYPE MAXENDIST

abstract type MaxEnDist{T<:AbstractFloat} <: Bounded{T} end

support(d::MaxEnDist) = (d.a, d.b)

### GENERAL MAXENDIST TYPE MED0

abstract type MED0{T<:AbstractFloat} <: MaxEnDist{T} end

C(d::MED0) = d.C

(p::PDF{<:MED0{T}})(x::T) where {T<:AbstractFloat} = 
	pab(x -> exp(A(p.d) * x^2 + B(p.d) * x + C(p.d)), x, p.d.a, p.d.b)

(P::CDF{<:MED0{T}})(x::T) where {T<:AbstractFloat} = 
	Pab(x -> intexp(A(P.d), B(P.d), C(P.d), P.d.a, x), x, P.d.a, P.d.b)

quantile(d::MED0{T}, p::T) where {T<:AbstractFloat} = 
	q01(p -> invintexp(A(d), B(d), C(d), d.a, p), p)

mean(d::MED0)    = intxexp(A(d), B(d), C(d), d.a, d.b)

moment2(d::MED0) = intxxexp(A(d), B(d), C(d), d.a, d.b)

entropy(d::MED0) = intqexp(A(d), B(d), C(d), d.a, d.b)

### CASE 000: KNOWN NOTHING 

struct MED000{T<:AbstractFloat} <: MED0{T}
	a::T
	b::T
	C::T
	MED000(a::T, b::T) where {T<:AbstractFloat} = 
		new{T}(a, b, -log(intexp(zero(T), zero(T), zero(T), a, b)))
end
MED000(a::AbstractFloat, b::AbstractFloat) = MED000(promote(a, b)...)
MED000(a::Real, b::Real) = MED000(float(a), float(b))

A(::MED000{T}) where {T<:AbstractFloat} = zero(T)

B(::MED000{T}) where {T<:AbstractFloat} = zero(T)

median(d::MED000)  = (d.a + d.b) / 2

mean(d::MED000)    = (d.a + d.b) / 2

moment2(d::MED000) = (d.a * (d.a + d.b) + d.b ^ 2) / 3

var(d::MED000)     = (d.b - d.a) ^ 2 / 12

entropy(d::MED000) = - d.C

med000(a::T, b::T) where {T<:AbstractFloat} = MED000(a, b)
med000(a::AbstractFloat, b::AbstractFloat) = med000(promote(a, b)...)
med000(a::Real, b::Real) = med000(float(a), float(b))

med(a::Real, b::Real, ::Nothing, ::Nothing, ::Nothing) = med000(a, b)

### CASE 010: KNOWN THE MEAN

struct MED010{T<:AbstractFloat} <: MED0{T}
	a::T
	b::T
	B::T
	C::T
	MED010(a::T, b::T, B::T) where {T<:AbstractFloat} = 
		new{T}(a, b, B, -log(intexp(zero(T), B, zero(T), a, b)))
end
MED010(a::AbstractFloat, b::AbstractFloat, B::AbstractFloat) = 
	MED010(promote(a, b, B)...)
MED010(a::Real, b::Real, B::Real) = MED010(float(a), float(b), float(B))

A(::MED010{T}) where {T<:AbstractFloat} = zero(T)

B(d::MED010) = d.B

mean(d::MED010) = 
	(d.a + d.b + (d.b-d.a) * cothminv((d.b-d.a)/2 * d.B)) / 2

function moment2(d::MED010)
	x = (d.b-d.a)/2 * d.B
	cmix = cothminv(x)
	return 2 \ 
		((d.b^2+d.a^2) + (d.b^2-d.a^2) * cmix - (d.b-d.a)^2 * (cmix / x))
end

function med010(a::T, b::T, u::T) where {T<:AbstractFloat}
	t = u - (a + b) / 2
	w = (b - a) / 2
	function f!(output, param)
		output[1] = w * cothminv(w * param[1]) - t
	end
	solution = nlsolve(f!, [3 * t / w^2], ftol=1e-12)
	converged(solution) || @warn "Low precision!"
	B = solution.zero[1]
	return iszero(B) ? MED000(a, b) : MED010(a, b, B)
end
med010(a::AbstractFloat, b::AbstractFloat, u::AbstractFloat) = 
	med010(promote(a, b, u)...)
med010(a::Real, b::Real, u::Real) = med010(float(a), float(b), float(u))

med(a::Real, b::Real, ::Nothing, u::Real, ::Nothing) = med010(a, b, u)

### CASE 011: KNOWN THE MEAN AND THE VARIANCE

struct MED011{T<:AbstractFloat} <: MED0{T}
	a::T
	b::T
	A::T
	B::T
	C::T
	MED011(a::T, b::T, A::T, B::T) where {T<:AbstractFloat} = 
		new{T}(a, b, A, B, -log(intexp(A, B, zero(T), a, b)))
end
MED011(a::AbstractFloat, b::AbstractFloat, 
	A::AbstractFloat, B::AbstractFloat) = MED011(promote(a, b, A, B)...)
MED011(a::Real, b::Real, A::Real, B::Real) = 
	MED011(float(a), float(b), float(A), float(B))

A(d::MED011) = d.A

B(d::MED011) = d.B

function med011(a::T, b::T, u::T, v::T) where {T<:AbstractFloat}
	d0 = med(a, b, nothing, u, nothing)
	v0 = var(d0)
	v == v0 && return d0
	params0 = [v<v0 ? u : a+b-u, sqrt(2*v)]
	function f011!(outputs, params)
		A = sign(v-v0) * (params[2] ^ -2)
		B = -2 * A * params[1]
		d = MED011(a, b, A, B)
		ud = mean(d)
		outputs[1] = ud - u
		outputs[2] = var(d; u=ud) - v
		return d
	end
	solution = nlsolve(f011!, params0, 
		method=:trust_region, ftol=1e-12, autoscale=false, factor=0.001)
	converged(solution) || @warn "Low precision!"
	return f011!(zeros(T, 2), solution.zero)
end
med011(a::AbstractFloat, b::AbstractFloat, u::AbstractFloat, 
	v::AbstractFloat) = med011(promote(a, b, u, v)...)
med011(a::Real, b::Real, u::Real, v::Real) = 
	med011(float(a), float(b), float(u), float(v))

med(a::Real, b::Real, ::Nothing, u::Real, v::Real) = med011(a, b, u, v)

### CASE 001: KNOWN THE VARIANCE

med(a::Real, b::Real, ::Nothing, ::Nothing, v::Real) = 
	med(a, b, nothing, (a+b)/2, v)

### GENERAL MAXENDIST TYPE MED1

abstract type MED1{T<:AbstractFloat} <: MaxEnDist{T} end

Ca(d::MED1) = d.Ca

Cb(d::MED1) = d.Cb

(p::PDF{<:MED1{T}})(x::T) where {T<:AbstractFloat} = 
	pamb(x -> exp(A(p.d) * x^2 + B(p.d) * x + Ca(p.d)), 
		 x -> exp(A(p.d) * x^2 + B(p.d) * x + Cb(p.d)), 
		 x, p.d.a, p.d.m, p.d.b)

(P::CDF{<:MED1{T}})(x::T) where {T<:AbstractFloat} = 
	Pamb(x -> intexp(A(P.d), B(P.d), Ca(P.d), P.d.a, x), 
		 x -> intexp(A(P.d), B(P.d), Cb(P.d), P.d.m, x), 
		 x, P.d.a, P.d.m, P.d.b)

quantile(d::MED1{T}, p::T) where {T<:AbstractFloat} = 
	q051(p -> invintexp(A(d), B(d), Ca(d), d.a, p), 
		 p -> invintexp(A(d), B(d), Cb(d), d.m, p), p)

median(d::MED1) = d.m

mean(d::MED1)    = intxexp(A(d), B(d), Ca(d), d.a, d.m) + 
				   intxexp(A(d), B(d), Cb(d), d.m, d.b)

moment2(d::MED1) = intxxexp(A(d), B(d), Ca(d), d.a, d.m) + 
				   intxxexp(A(d), B(d), Cb(d), d.m, d.b)

entropy(d::MED1) = intqexp(A(d), B(d), Ca(d), d.a, d.m) + 
				   intqexp(A(d), B(d), Cb(d), d.m, d.b)

### CASE 100: KNOWN THE MEDIAN

struct MED100{T<:AbstractFloat} <: MED1{T}
	a::T
	b::T
	m::T
	Ca::T
	Cb::T
	MED100(a::T, b::T, m::T) where {T<:AbstractFloat} = 
		new{T}(a, b, m, -log(intexp(zero(T), zero(T), zero(T), a, m) * 2), 
						-log(intexp(zero(T), zero(T), zero(T), m, b) * 2))
end
MED100(a::AbstractFloat, b::AbstractFloat, m::AbstractFloat) = 
	MED100(promote(a, b, m)...)
MED100(a::Real, b::Real, m::Real) = MED100(float(a), float(b), float(m))

A(::MED100{T}) where {T<:AbstractFloat} = zero(T)

B(::MED100{T}) where {T<:AbstractFloat} = zero(T)

mean(d::MED100)    = (d.a + 2*d.m + d.b) / 4

moment2(d::MED100) = (d.a^2 + d.b^2 + 4 * d.m * mean(d)) / 6

var(d::MED100)     = (5 * (d.b-d.a)^2 + 4 * (d.m-d.a) * (d.m-d.b)) / 48

entropy(d::MED100) = - (d.Ca + d.Cb) / 2

med100(a::T, b::T, m::T) where {T<:AbstractFloat} = MED100(a, b, m)
med100(a::AbstractFloat, b::AbstractFloat, m::AbstractFloat) = 
	med100(promote(a, b, m)...)
med100(a::Real, b::Real, m::Real) = med100(float(a), float(b), float(m))

med(a::Real, b::Real, m::Real, ::Nothing, ::Nothing) = med100(a, b, m)

### CASE 110: KNOWN THE MEDIAN AND THE MEAN

struct MED110{T<:AbstractFloat} <: MED1{T}
	a::T
	b::T
	m::T
	B::T
	Ca::T
	Cb::T
	MED110(a::T, b::T, m::T, B::T) where {T<:AbstractFloat} = 
		new{T}(a, b, m, B, -log(intexp(zero(T), B, zero(T), a, m) * 2), 
						   -log(intexp(zero(T), B, zero(T), m, b) * 2))
end
MED110(a::AbstractFloat, b::AbstractFloat, m::AbstractFloat, 
	B::AbstractFloat) = MED110(promote(a, b, m, B)...)
MED110(a::Real, b::Real, m::Real, B::Real) = 
	MED110(float(a), float(b), float(m), float(B))

A(::MED110{T}) where {T<:AbstractFloat} = zero(T)

B(d::MED110) = d.B

mean(d::MED110) = (d.a + 2 * d.m + d.b + 
	(d.m - d.a) * cothminv((d.m-d.a) * d.B / 2) + 
	(d.b - d.m) * cothminv((d.b-d.m) * d.B / 2)) / 4

function moment2(d::MED110)
	xa = (d.m-d.a)/2 * d.B
	xb = (d.b-d.m)/2 * d.B
	cmixa = cothminv(xa)
	cmixb = cothminv(xb)
	return 4 \ 
		((d.m^2+d.a^2) + (d.m^2-d.a^2) * cmixa - (d.m-d.a)^2 * (cmixa / xa) + 
		 (d.b^2+d.m^2) + (d.b^2-d.m^2) * cmixb - (d.b-d.m)^2 * (cmixb / xb))
end

entropy(d::MED110) = -(d.Ca + d.Cb) / 2 - d.B * mean(d)

function med110(a::T, b::T, m::T, u::T) where {T<:AbstractFloat}
	t = 2 * u - m - (a + b) / 2
	wa = (m - a) / 2
	wb = (b - m) / 2
	function f!(output, param)
		output[1] = wa * cothminv(wa * param[1]) + 
					wb * cothminv(wb * param[1]) - t
	end
	solution = nlsolve(f!, [3 * t / (wa^2 + wb^2)], ftol=1e-12)
	converged(solution) || @warn "Low precision!"
	B = solution.zero[1]
	return iszero(B) ? MED100(a, b, m) : MED110(a, b, m, B)
end

med110(a::AbstractFloat, b::AbstractFloat, m::AbstractFloat, 
	u::AbstractFloat) = med110(promote(a, b, m, u)...)
med110(a::Real, b::Real, m::Real, u::Real) = 
	med110(float(a), float(b), float(m), float(u))

med(a::Real, b::Real, m::Real, u::Real, ::Nothing) = med110(a, b, m, u)

### CASE 111: KNOWN THE MEDIAN, THE MEAN, AND THE VARIANCE

struct MED111{T<:AbstractFloat} <: MED1{T}
	a::T
	b::T
	m::T
	A::T
	B::T
	Ca::T
	Cb::T
	MED111(a::T, b::T, m::T, A::T, B::T) where {T<:AbstractFloat} = 
		new{T}(a, b, m, A, B, -log(intexp(A, B, zero(T), a, m) * 2), 
							  -log(intexp(A, B, zero(T), m, b) * 2))
end
MED111(a::AbstractFloat, b::AbstractFloat, m::AbstractFloat, 
	A::AbstractFloat, B::AbstractFloat) = MED111(promote(a, b, m, A, B)...)
MED111(a::Real, b::Real, m::Real, A::Real, B::Real) = 
	MED111(float(a), float(b), float(m), float(A), float(B))

A(d::MED111) = d.A

B(d::MED111) = d.B

function med111(a::T, b::T, m::T, u::T, v::T) where {T<:AbstractFloat}
	d0 = med(a, b, m, u, nothing)
	v0 = var(d0)
	v == v0 && return d0
	params0 = [v<v0 ? u : a+b-u, sqrt(2*v)]
	function f111!(outputs, params)
		A = sign(v-v0) * (params[2] ^ -2)
		B = -2 * A * params[1]
		d = MED111(a, b, m, A, B)
		ud = mean(d)
		outputs[1] = ud - u
		outputs[2] = var(d; u=ud) - v
		return d
	end
	solution = nlsolve(f111!, params0, 
		method=:trust_region, ftol=1e-12, autoscale=false, factor=0.001)
	converged(solution) || @warn "Low precision!"
	return f111!(zeros(T, 2), solution.zero)
end
med111(a::AbstractFloat, b::AbstractFloat, m::AbstractFloat, u::AbstractFloat, 
	v::AbstractFloat) = med111(promote(a, b, m, u, v)...)
med111(a::Real, b::Real, m::Real, u::Real, v::Real) = 
	med111(float(a), float(b), float(m), float(u), float(v))

med(a::Real, b::Real, m::Real, u::Real, v::Real) = med111(a, b, m, u, v)

### CASE 101: KNOWN THE MEDIAN AND THE VARIANCE

"""
	urange(a::Real, b::Real, m::Real, v::Real)

Return the possible range of mean of a distribution bounded on ``[a, b]`` 
with median `m` and variance `v` as an interval ``(ul, ur)``.
"""
function urange(a::Real, b::Real, m::Real, v::Real)
	s = sqrt(v)
	l1 = m - s
	r1 = m + s
	e = (b - a) ^ 2 / 4 - v
	l2 = (a+b)/2 + (m-a)/2 - sqrt(e + (m-a)^2/4)
	r2 = (a+b)/2 - (b-m)/2 + sqrt(e + (b-m)^2/4)
	return max(l1, l2), min(r1, r2)
end

function med(a::Real, b::Real, m::Real, ::Nothing, v::Real)
	l, r = urange(a, b, m, v)
	h(u) = -entropy(med(a, b, m, u[1], v))
	result = optimize(h, [l], [r], [(l+r)/2], Fminbox(), Options(g_tol=1e-12))
	u = minimizer(result)[1]
	return med(a, b, m, u, v)
end

### THE MAIN FUNCTION WITH BOUND CHECKING

"""
	maxendist(a::Real, b::Real; 
		median::Union{Real,Nothing}=nothing, 
		mean::Union{Real,Nothing}=nothing, 
		std::Union{Real,Nothing}=nothing, var::Union{Real,Nothing}=nothing

Compute the maximum-entropy distribution bounded on ``[a, b]`` with given 
median, mean, standard deviation, and/or variance.
"""
function maxendist(a::Real, b::Real; 
		median::Union{Real,Nothing}=nothing, 
		mean::Union{Real,Nothing}=nothing, 
		std::Union{Real,Nothing}=nothing, var::Union{Real,Nothing}=nothing)
	xab(a, b)
	xabm(a, b, median)
	xabu(a, b, mean)
	var = xsv(xs(std), xv(var))
	xabuv(a, b, mean, var)
	xabmu(a, b, median, mean)
	xabmuv(a, b, median, mean, var)
	return med(a, b, median, mean, var)
end

"""
	xab(a::Real, b::Real)

Check if ``[a, b]`` is a well-defined real interval with positive length.
"""
function xab(a::Real, b::Real)
	(isnan(a) || isnan(b)) && 
		error("Endpoint(s) of the interval is/are not real!")
	(isinf(a) || isinf(b)) && error("The interval is not bounded!")
	a == b && error("The interval with zero length (one-point distribution)!")
	a > b && error("The interval is reversed!")
end

"""
	xabm(a::Real, b::Real, m::Real)

Check if median `m` is possible for a continuous distribution bounded on 
``[a, b]``. See Theorem 100 in the article.
"""
function xabm(a::Real, b::Real, m::Real)
	isnan(m) && error("The median is not real!")
	a <= m <= b || error("The median falls out of the support interval!")
	a < m < b || error("These conditions yield a one-point distribution!")
end
xabm(::Real, ::Real, ::Nothing) = nothing

"""
	xabu(a::Real, b::Real, u::Real)

Check if mean `u` is possible for a continuous distribution bounded on 
``[a, b]``. See Theorem 010 in the article.
"""
function xabu(a::Real, b::Real, u::Real)
	isnan(u) && error("The mean is not real!")
	a <= u <= b || error("The mean falls out of the support interval!")
	a < u < b || error("These conditions yield a one-point distribution!")
end
xabu(::Real, ::Real, ::Nothing) = nothing

"""
	xs(s::Real)

Check if standard deviation `s` is possible for a continuous distribution.
"""
function xs(s::Real)
	isnan(u) && error("The standard deviation is not real!")
	s < 0 && error("The standard deviation is negative!")
	s == 0 && error("The standard deviation is zero (one-point distribution)!")
	return s
end
xs(::Nothing) = nothing

"""
	xv(v::Real)

Check if variance `v` is possible for a continuous distribution.
"""
function xv(v::Real)
	isnan(v) && error("The variance is not real!")
	v < 0 && error("The variance is negative!")
	v == 0 && error("The variance is zero (one-point distribution)!")
	return v
end
xv(::Nothing) = nothing

"""
	xsv(s::Real, v::Real)

Check if standard deviation `s` and variance `v` is consistent for a 
distribution. If so, the variance is returned.
"""
xsv(s::Real, v::Real) = isapprox(s^2, v) ? (return v) : 
	error("The standard deviation and the variance are inconsistent!")
xsv(::Nothing, v::Real) = v
xsv(s::Real, ::Nothing) = s^2
xsv(::Nothing, ::Nothing) = nothing

"""
	xabuv(a::Real, b::Real, u::Union{Real, Nothing}, v::Union{Real, Nothing})

Check if mean `u` and variance `v` is simultaneously possible for a continuous 
distribution bounded on ``[a, b]``. See Theorem 001 (Popoviciu(1935)'s 
inequality) and Theorem 011 (Bhatia & Davis (2000)'s inequality) in the 
article.
"""
function xabuv(a::Real, b::Real, u::Real, v::Real)
	vt = (u - a) * (b - u)
	v > vt && error("The variance is too large under given mean!")
	v == vt && error("These conditions yield a two-point distribution!")
end
function xabuv(a::Real, b::Real, ::Nothing, v::Real)
	vt = (b - a) ^ 2 / 4
	v > vt && error("The variance is too large!")
	v == vt && error("These conditions yield a two-point distribution!")
end
xabuv(::Real, ::Real, ::Any, ::Nothing) = nothing

"""
	xabmu(a::Real, b::Real, m::Union{Real, Nothing}, u::Union{Real, Nothing})

Check if median `m` and mean `u` is simultaneously possible for a continuous 
distribution bounded on ``[a, b]``. See Theorem 110 in the article.
"""
function xabmu(a::Real, b::Real, m::Real, u::Real)
	a + m > 2 * u && error("The mean is too small under given median!")
	a + m == 2 * u && 
		error("These conditions yield a two-point distribution!")
	2 * u > b + m && error("The mean is too large under given median!")
	2 * u == b + m && 
		error("These conditions yield a two-point distribution!")
end
xabmu(::Real, ::Real, ::Any, ::Any) = nothing

"""
	xabmuv(a::Real, b::Real, m::Union{Real, Nothing}, 
		u::Union{Real, Nothing}, v::Union{Real, Nothing})

Check if median `m`, mean `u`, and variance `v` is simultaneously possible 
for a continuous distribution bounded on ``[a, b]``. See Theorem 111a and 
Theorem 111b in the article.
"""
function xabmuv(a::Real, b::Real, m::Real, u::Real, v::Real)
	v < (m - u) ^ 2 && 
		error("The variance is too small under given median and mean!")
	v == (m - u) ^ 2 && 
		error("These conditions yield a three-point distribution!")
	d = u - (a+b)/2
	vt = (b - a) ^ 2 / 4 + d * (m - d - (d > 0 ? b : a))
	v > vt && 
		error("The variance is too large under given median and mean!")
	v == vt && 
		error("These conditions yield a three-point distribution!")	
end
xabmuv(::Real, ::Real, ::Any, ::Any, ::Any) = nothing

### WHEN BOUNDED

function bound(d::MED000{T}, a::T, b::T) where {T<:AbstractFloat}
	a > b && error("Lower and upper bounds are out of order!")
	da, db = support(d)
	na, nb = max(a, da), min(b, db)
	na >= nb && error("No information remains after bounding!")
	return MED000(na, nb)
end

function bound(d::MED010{T}, a::T, b::T) where {T<:AbstractFloat}
	a > b && error("Lower and upper bounds are out of order!")
	da, db = support(d)
	na, nb = max(a, da), min(b, db)
	na >= nb && error("No information remains after bounding!")
	return MED010(na, nb, d.B)
end

function bound(d::MED011{T}, a::T, b::T) where {T<:AbstractFloat}
	a > b && error("Lower and upper bounds are out of order!")
	da, db = support(d)
	na, nb = max(a, da), min(b, db)
	na >= nb && error("No information remains after bounding!")
	return MED011(na, nb, d.A, d.B)
end

function bound(d::MED100{T}, a::T, b::T) where {T<:AbstractFloat}
	a > b && error("Lower and upper bounds are out of order!")
	da, db = support(d)
	na, nb = max(a, da), min(b, db)
	na >= nb && error("No information remains after bounding!")
	if na < d.m < nb
		return MED100(na, nb, d.m)
	else
		return MED000(na, nb)
	end
end

function bound(d::MED110{T}, a::T, b::T) where {T<:AbstractFloat}
	a > b && error("Lower and upper bounds are out of order!")
	da, db = support(d)
	na, nb = max(a, da), min(b, db)
	na >= nb && error("No information remains after bounding!")
	if na < d.m < nb
		return MED110(na, nb, d.m, d.B)
	else
		return MED010(na, nb, d.B)
	end
end

function bound(d::MED111{T}, a::T, b::T) where {T<:AbstractFloat}
	a > b && error("Lower and upper bounds are out of order!")
	da, db = support(d)
	na, nb = max(a, da), min(b, db)
	na >= nb && error("No information remains after bounding!")
	if na < d.m < nb
		return MED111(na, nb, d.m, d.A, d.B)
	else
		return MED011(na, nb, d.A, d.B)
	end
end

### WHEN SMOOTHED

function (p::PDF{<:Smooth{T,<:MED0{T}}})(x::T) where {T<:AbstractFloat}
	k = p.d.h ^ -2 / 2
	return sqrt(2*T(pi)) * p.d.h \ 
		intexp(A(p.d.d) - k, B(p.d.d) + 2*x * k, C(p.d.d) - x^2 * k, 
			p.d.d.a, p.d.d.b)
end

function (p::PDF{<:Smooth{T,<:MED1{T}}})(x::T) where {T<:AbstractFloat}
	k = p.d.h ^ -2 / 2
	return sqrt(2*T(pi)) * p.d.h \ 
		(intexp(A(p.d.d) - k, B(p.d.d) + 2*x * k, Ca(p.d.d) - x^2 * k, 
			p.d.d.a, p.d.d.m) + 
		 intexp(A(p.d.d) - k, B(p.d.d) + 2*x * k, Cb(p.d.d) - x^2 * k, 
			p.d.d.m, p.d.d.b))
end

function (P::CDF{<:Smooth{T,<:MED0{T}}})(x::T) where {T<:AbstractFloat}
	f(t) = erf((x-t) / (sqrt(T(2))*P.d.h)) * pdf(P.d.d)(t)
	return 2 \ (1 + integrate(f, P.d.d.a, P.d.d.b))
end

function (P::CDF{<:Smooth{T,<:MED1{T}}})(x::T) where {T<:AbstractFloat}
	f(t) = erf((x-t) / (sqrt(T(2))*P.d.h)) * pdf(P.d.d)(t)
	a, m, b = P.d.d.a, P.d.d.m, P.d.d.b
	mp, mn = prevfloat(m), nextfloat(m)
	return 2 \ (1 + eps(m) * (f(mp) + f(mn)) + 
		integrate(f, a, mp) + integrate(f, mn, P.d.d.b))
end
