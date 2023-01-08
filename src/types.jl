# src/types.jl

abstract type Distribution{T<:AbstractFloat} end

abstract type Bounded{T<:AbstractFloat} <: Distribution{T} end

abstract type Unbounded{T<:AbstractFloat} <: Distribution{T} end

### GENERAL PDF

"""
	PDF{D<:Distribution}

Container type for a probability density function.
"""
struct PDF{D<:Distribution} <: Function
	d::D
end

"""
	pdf(d::Distribution)

Get the probability density function of a distribution as a Julia function.
"""
pdf(d::Distribution) = PDF(d)

(p::PDF{<:Distribution{T}})(x::T) where {T<:AbstractFloat} = 
	error("Probability density function undefined!")
(p::PDF{<:Distribution{T}})(x::Real) where {T<:AbstractFloat} = p(T(x))

"""
	pab(p::Function, x::Real, a::Real, b::Real)

Return 
- zero, if `x < a`, 
- `p(x)`, if `a <= x <= b`, 
- zero, if `x > b`.
"""
pab(p::Function, x::T, a::T, b::T) where {T<:AbstractFloat} = 
	a <= x <= b ? p(x) : zero(T)
pab(p::Function, x::AbstractFloat, a::AbstractFloat, b::AbstractFloat) = 
	pab(p, promote(x, a, b)...)
pab(p::Function, x::Real, a::Real, b::Real) = 
	pab(p, float(x), float(a), float(b))

"""
	pamb(pa::Function, pb::Function, x::Real, a::Real, m::Real, b::Real)

Return 
- zero, if `x < a`, 
- `pa(x)`, if `a <= x <= m`, 
- `pb(x)`, if `m < x <= b`, 
- zero, if `x > b`.
"""
pamb(pa::Function, pb::Function, x::T, a::T, m::T, b::T) where 
	{T<:AbstractFloat} = 
		x < a ? zero(T) : x <= m ? pa(x) : x <= b ? pb(x) : zero(T)
pamb(pa::Function, pb::Function, x::AbstractFloat, a::AbstractFloat, 
	m::AbstractFloat, b::AbstractFloat) = pamb(pa, pb, promote(x, a, m, b)...)
pamb(pa::Function, pb::Function, x::Real, a::Real, m::Real, b::Real) = 
	pamb(pa, pb, float(x), float(a), float(m), float(b))

### GENERAL CDF

"""
	CDF{D<:Distribution}

Container type for a cumulative distribution function.
"""
struct CDF{D<:Distribution} <: Function
	d::D
end

"""
	cdf(d::Distribution)

Get the cumulative distribution function of a distribution as a Julia function.
"""
cdf(d::Distribution) = CDF(d)

(P::CDF{<:Distribution{T}})(x::T) where {T<:AbstractFloat} = 
	error("Cumulative distribution function defined!")
(P::CDF{<:Distribution{T}})(x::Real) where {T<:AbstractFloat} = P(T(x))

"""
	Pab(P::Function, x::Real, a::Real, b::Real)

Return 
- zero, if `x < a`, 
- `P(x)`, if `a <= x <= b`, 
- one, if `x > b`.
"""
Pab(P::Function, x::T, a::T, b::T) where {T<:AbstractFloat} = 
	x < a ? zero(T) : x <= b ? P(x) : one(T)
Pab(P::Function, x::AbstractFloat, a::AbstractFloat, b::AbstractFloat) = 
	Pab(P, promote(x, a, b)...)
Pab(P::Function, x::Real, a::Real, b::Real) = 
	Pab(P, float(x), float(a), float(b))

"""
	Pamb(Pa::Function, Pb::Function, x::Real, a::Real, m::Real, b::Real)

Return 
- zero, if `x < a`, 
- `Pa(x)`, if `a <= x <= m`, 
- `Pb(x)` + 1/2, if `m < x <= b`, 
- one, if `x > b`.
"""
Pamb(Pa::Function, Pb::Function, x::T, a::T, m::T, b::T) where 
	{T<:AbstractFloat} = 
		x < a ? zero(T) : x <= m ? Pa(x) : x <= b ? Pb(x) + inv(T(2)) : one(T)
Pamb(Pa::Function, Pb::Function, x::AbstractFloat, a::AbstractFloat, 
	m::AbstractFloat, b::AbstractFloat) = Pamb(Pa, Pb, promote(x, a, m, b)...)
Pamb(Pa::Function, Pb::Function, x::Real, a::Real, m::Real, b::Real) = 
	Pamb(Pa, Pb, float(x), float(a), float(m), float(b))

function (P::CDF{<:Bounded{T}})(x::T) where {T<:AbstractFloat}
	a, b = support(P.d)
	return Pab(x -> integrate(pdf(P.d), a, x), x, a, b)
end

### GENERAL QUANTILE

"""
	quantile(d::Distribution, p::Real, x0::Real)
	quantile(d::Distribution, p::Real)
	quantile(d::Distribution, pp::Vector{<:Real})
	quantile(d::Distribution)

Compute the quantile value in a distribution. The last syntax creates a Julia 
function similar to those from `pdf` and `cdf` that maps `p` to `q(d, p)`.
"""
function quantile(d::Distribution{T}, p::T, x0::T) where {T<:AbstractFloat}
	0 <= p <= 1 || error("The probability given is illegal!")
	Pcdf = cdf(d)
	ppdf = pdf(d)
	f!(e, x) = (e[1] = Pcdf(x[1]) - p)
	j!(j, x) = (j[1] = ppdf(x[1]))
	solution = nlsolve(f!, j!, [x0], method=:newton, ftol=1e-13)
	converged(solution) || @warn "Low precision!"
	return solution.zero[1]
end
quantile(d::Distribution{T}, p::Real, x0::Real) where {T<:AbstractFloat} = 
	quantile(d, T(p), T(x0))
quantile(d::Distribution{T}, p::T) where {T<:AbstractFloat} = 
	quantile(d, p, mean(d))
function quantile(d::Bounded{T}, p::T) where {T<:AbstractFloat}
	a, b = support(d)
	return max(a, min(b, quantile(d, p, a + p * (b - a))))
end
quantile(d::Distribution{T}, p::Real) where {T<:AbstractFloat} = 
	quantile(d, T(p))
quantile(d::Distribution, pp::Vector{<:Real}) = quantile.([d], pp)
quantile(d::Distribution) = p -> quantile(d, p)

"""
	q01(q::Function, p::Real)

Return `q(p)` if `0 <= p <= 1`, or an error otherwise. 
"""
q01(q::Function, p::Real) = 
	0 <= p <= 1 ? q(p) : error("Input probability out of [0,1] range!")

"""
	q051(q0::Function, q1::Function, p::Real)

Return 
- `q0(p)`, if `0 <= p <= 1/2`, 
- `q1(p-1/2)`, if `1/2 < p <= 1`, 
- an error, otherwise.
"""
q051(q0::Function, q1::Function, p::T) where {T<:AbstractFloat} = 
	0 <= p <= 1 ? p <= inv(T(2)) ? q0(p) : q1(p - inv(T(2))) : 
		error("Input probability out of [0,1] range!")
q051(q0::Function, q1::Function, p::Real) = q051(q0, q1, float(p))

### GENERAL OTHERS

"""
	support(d::Distribution)

Compute the support `(a, b)` of a distribution. For unbounded distributions, 
the support interval expands from negative infinity to positive infinity. 
"""
support(d::Unbounded{T}) where {T<:AbstractFloat} = (typemin(T), typemax(T))

"""
	median(d::Distribution)

Compute the median of a distribution.
"""
median(d::Distribution{T}) where {T<:AbstractFloat} = quantile(d, inv(T(2)))

"""
	mean(d::Distribution)

Compute the mean of a distribution.
"""
mean(d::Bounded) = integrate(x -> x * pdf(d)(x), support(d)...)

"""
	moment2(d::Distribution)

Compute the second moment (e.g. ``\\mathcal{E}[X^2]`` for a random variable 
``X``) of a distribution.
"""
moment2(d::Bounded) = integrate(x -> x^2 * pdf(d)(x), support(d)...)

"""
	var(d::Distribution; u=mean(d))

Compute the variance of a distribution. The mean can be provided if known.
"""
var(d::Distribution; u=mean(d)) = moment2(d) - u ^ 2

"""
	std(d::Distribution; u=mean(d))

Compute the standard deviation of a distribution. The mean can be provided if 
known.
"""
std(d::Distribution; u=mean(d)) = sqrt(var(d; u=u))

"""
	entropy(d::Bounded; u=mean(d))

Compute the differential entropy of a bounded distribution. 
"""
entropy(d::Bounded) = - integrate(x -> xlog(pdf(d)(x)), support(d)...)

### BOUNDING

struct Bound{T<:AbstractFloat, D<:Distribution{T}} <: Bounded{T}
	d::D
	a::T
	b::T
	k::T
	function Bound(d::D, a::T, b::T) where 
			{T<:AbstractFloat, D<:Distribution{T}}
		a >= b && 
			error("Lower and upper bounds are identical or out of order!")
		k = inv(cdf(d)(b) - cdf(d)(a))
		isinf(k) && 
			error("The distribution has no information between the bounds!")
		return new{T,D}(d, a, b, k)
	end
end
Bound(d::Distribution{T}, a::Real, b::Real) where 
	{T<:AbstractFloat} = Bound(d, T(a), T(b))

"""
	bound(d::Distribution, a::Real, b::Real)
	bound(d::Bounded)
	bound(d::Smooth{T,<:Sample{T}}) where {T<:AbstractFloat}

Bound or truncate a distribution by interval ``[a, b]``, and then return the 
new bounded distribution. Especially, bounding a bounded distribution without 
giving `a` and `b` returns the bounded distribution itself; when bounding a 
smoothed sample (see `smooth`) without giving `a` and `b`, the interval is set 
to the extrema of the original sample.
"""
bound(d::Distribution{T}, a::T, b::T) where {T<:AbstractFloat} = Bound(d, a, b)
bound(d::Distribution{T}, a::Real, b::Real) where {T<:AbstractFloat} = 
	bound(d, T(a), T(b))
bound(d::Bounded) = d

support(d::Bound) = (d.a, d.b)

(p::PDF{Bound{T,D}})(x::T) where {T<:AbstractFloat, D<:Distribution{T}} = 
	pab(x -> pdf(p.d.d)(x) * p.d.k, x, p.d.a, p.d.b)

(P::CDF{Bound{T,D}})(x::T) where {T<:AbstractFloat, D<:Distribution{T}} = 
	Pab(x -> (cdf(P.d.d)(x) - cdf(P.d.d)(P.d.a)) * P.d.k, x, P.d.a, P.d.b)

### SMOOTHING

struct Smooth{T<:AbstractFloat, D<:Distribution{T}} <: Unbounded{T}
	d::D
	h::T
	Smooth(d::D, h::T) where {T<:AbstractFloat, D<:Distribution{T}} = 
		iszero(h) ? d : new{T,D}(d, abs(h))
end
Smooth(d::Distribution{T}, h::Real) where {T<:AbstractFloat} = Smooth(d, T(h))

"""
	smooth(d::Distribution, h::Real)
	smooth(d::Smooth)
	smooth(d::Sample)
	smooth(v::Vector{<:Real})

Smooth a distribution with a Gaussian kernal with bandwidth `h`. Especially, 
smoothing a smoothed distribution without giving `h` returns the smoothed 
distribution itself; when smoothing a sample or a vector of reals, the 
bandwidth is calculated using Silverman (1986)'s rule of thumb (see 
`helrot` for more details).
"""
smooth(d::Distribution{T}, h::T) where {T<:AbstractFloat} = Smooth(d, h)
smooth(d::Distribution{T}, h::Real) where {T<:AbstractFloat} = smooth(d, T(h))
smooth(d::Smooth{T,D}, h::T=zero(T)) where 
	{T<:AbstractFloat, D<:Distribution{T}} = Smooth(d.d, hypot(d.h, h))

mean(d::Smooth) = mean(d.d)

moment2(d::Smooth) = moment2(d.d) + d.h^2

var(d::Smooth) = var(d.d) + d.h^2

(p::PDF{<:Smooth{T,<:Bounded{T}}})(x::T) where {T<:AbstractFloat} = 
	 integrate(t -> pdf(p.d.d)(t) * exp(-2 \ ((x-t)/p.d.h)^2), 
		support(p.d.d)...) / (sqrt(2*T(pi)) * p.d.h)

(P::CDF{<:Smooth{T,<:Bounded{T}}})(x::T) where {T<:AbstractFloat} = 
	2 \ (1 + integrate(t -> erf((x-t) / (sqrt(T(2))*P.d.h)) * pdf(P.d.d)(t), 
		support(P.d.d)...))

bound(d::Smooth{T,<:Bounded{T}}) where {T<:AbstractFloat} = 
	bound(d, support(d.d)...)
