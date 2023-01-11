# src/sample.jl

struct Sample{T<:AbstractFloat} <: Distribution{T}
	v::Vector{T}
	Sample(v::Vector{T}) where {T<:AbstractFloat} = new{T}(sort(v))
end
Sample(v::Vector{<:Real}) = Sample(float(v))

"""
	sample(v::Vector{<:Real})

Create a sample as a distribution.
"""
sample(v::Vector{<:Real}) = Sample(v)

support(d::Sample) = (first(d.v), last(d.v))

(p::PDF{<:Sample{T}})(x::T) where {T<:AbstractFloat} = 
	isempty(searchsorted(p.d.v, x)) ? zero(T) : typemax(T)

function (P::CDF{<:Sample{T}})(x::T) where {T<:AbstractFloat}
	pos = searchsorted(P.d.v, x)
	return (first(pos) + last(pos) - 1) / (2 * length(P.d.v))
end

function quantile(d::Sample{T}, p::T) where {T<:AbstractFloat}
	q01(identity, p);
	n = length(d.v)
	mf = n * p
	m = round(Int, mf)
	if isapprox(mf, m, atol=eps(mf)*2)
		m == 0 && return first(d.v)
		m == n && return last(d.v)
		return 2 \ (d.v[m] + d.v[m+1])
	else
		m = ceil(Int, mf)
		return d.v[m]
	end
end

function median(d::Sample)
	n = length(d.v)
	return isodd(n) ? d.v[(1+n)>>1] : 2 \ (d.v[n>>1] + d.v[1+n>>1])
end

mean(d::Sample) = sum(d.v) / length(d.v)

moment2(d::Sample) = sum(d.v .^ 2) / length(d.v)

var(d::Sample; u=mean(d)) = sum((d.v .- u) .^ 2) / length(d.v)

entropy(d::Sample{T}) where {T<:AbstractFloat} = typemin(T)

skewness(d::Sample; u=mean(d), s=std(d)) = sum(((d.v.-u)./s).^3) / length(d.v)

kurtosis(d::Sample; u=mean(d), s=std(d)) = sum(((d.v.-u)./s).^4) / length(d.v)

function rescale(d::Sample{T}; a=zero(T), b=one(T)) where {T<:AbstractFloat}
	da, db = support(d)
	return sample((d.v .- da) .* (T(b - a) / (db - da)) .+ T(a))
end

### WHEN BOUNDED

function bound(d::Sample{T}, a::T, b::T) where {T<:AbstractFloat}
	a > b && error("Lower and upper bounds are out of order!")
	v = d.v[a .<= d.v .<= b]
	isempty(v) && error("No sample remained after bounding!")
	return Sample(v)
end

bound(d::Sample) = d

### WHEN SMOOTHED

"""
	helrot(d::Sample{T}; useiqr::Bool=true) where {T<:AbstractFloat}

Compute the optimal bandwidth using Silverman (1986)'s rule of thumb for 
smoothing with a Gaussian kernal. If `useiqr` is `true`, then the version 
using interquartile range (IQR) is applied.
"""
function helrot(d::Sample{T}; useiqr::Bool=true) where {T<:AbstractFloat}
	c106 = (T(4)/3) ^ inv(T(5))
	s = std(d)
	if useiqr
		c135 = erfinv(inv(T(2))) * sqrt(T(8))
		s = min(s, (quantile(d, T(3)/4) - quantile(d, T(1)/4)) / c135)
	end
	return c106 * s * length(d.v) ^ -inv(T(5))
end

smooth(d::Sample) = smooth(d, helrot(d))
smooth(v::Vector{<:Real}) = smooth(sample(v))

(p::PDF{<:Smooth{T,<:Sample{T}}})(x::T) where {T<:AbstractFloat} = 
	sum(@. exp(- ((x - p.d.d.v) / p.d.h) ^ 2 / 2)) / 
		(sqrt(2*T(pi)) * p.d.h * length(p.d.d.v))

(P::CDF{<:Smooth{T,<:Sample{T}}})(x::T) where {T<:AbstractFloat} = 2 \ 
	(1 + sum(@. erf((x - P.d.d.v) / (P.d.h * sqrt(T(2))))) / length(P.d.d.v))

### WHEN BOUNDED AFTER SMOOTHED

bound(d::Smooth{T,<:Sample{T}}) where {T<:AbstractFloat} = 
	bound(d, support(d.d)...)

function mean(d::Bound{T,<:Smooth{T,<:Sample{T}}}) where {T<:AbstractFloat}
	q = -1 / (2 * d.d.h^2)
	return d.k / (sqrt(2*T(pi)) * d.d.h * length(d.d.d.v)) * 
		sum(@. intxexp(q, -2*d.d.d.v * q, d.d.d.v^2 * q, d.a, d.b))
end

function moment2(d::Bound{T,<:Smooth{T,<:Sample{T}}}) where {T<:AbstractFloat}
	q = -1 / (2 * d.d.h^2)
	return d.k / (sqrt(2*T(pi)) * d.d.h * length(d.d.d.v)) * 
		sum(@. intxxexp(q, -2*d.d.d.v * q, d.d.d.v^2 * q, d.a, d.b))
end
