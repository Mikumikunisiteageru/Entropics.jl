# src/smoothing.jl

import Statistics # median, mean, var, std, quantile

median(v::AbstractVector) = Statistics.median(v)
mean(v::AbstractVector) = Statistics.mean(v)
var(v::AbstractVector; corrected::Bool=true) = 
	Statistics.var(v; corrected=corrected)
std(v::AbstractVector; corrected::Bool=true) = 
	Statistics.std(v; corrected=corrected)
quantile(v::AbstractVector, p; sorted::Bool=false) = 
	Statistics.quantile(v, p; sorted=sorted)

export smooth

struct Smoothed{T<:AbstractFloat}
	data::Vector{T}
	h::T
	Smoothed(data::Vector{T}, h::T) where {T<:AbstractFloat} = 
		new{T}(sort(data), h)
end
smooth(data::Vector{T}, h::T) where {T<:AbstractFloat} = Smoothed(data, T)
smooth(data::Vector{<:AbstractFloat}, h::Real) = Smoothed(data, eltype(data)(h))
smooth(data::Vector{<:AbstractFloat}) = Smoothed(data, find_h_rot(data))
smooth(data::Vector{<:Real}) = smooth(float(data))

function find_h_cv(data::Vector{T}) where {T<:AbstractFloat}
	n = length(data)
	c1 = T(1) / (2 * sqrt(pi) * n^2)
	c2 = T(-2) / (sqrt(2 * pi) * n * (n-1))
	mqt = T(-1/4)
	function cv(h)
		x = data ./ h[1]
		s1 = zero(T)
		s2 = zero(T)
		for i = 1:n
			for j = 1:n
				e = exp(mqt * (x[i] - x[j]) ^ 2)
				s1 += e
				(i == j) || (s2 += e^2)
			end
		end
		return (c1 * s1 + c2 * s2) / h[1]
	end
	result = optimize(cv, [zero(T)], [T(Inf)], [std(data)])
	return result.minimizer
end

function find_h_rot(data::Vector{T}; R::Bool=true) where {T<:AbstractFloat}
	c106 = (4/3) ^ (1/5)
	c135 = erfinv(1/2) * sqrt(8)
	sigma = Statistics.std(data)
	if R
		sigma = min(sigma, -(quantile(data, [3/4, 1/4])...) / c135)
	end
	return c106 * sigma * length(data) ^ (-1/5)
end

function median(s::Smoothed)
	P = CDFSmoothed(s)
	f!(e, m) = (e[1] = P(m[1]) - 1/2)
	solution = nlsolve(f!, [median(s.data)], ftol=1e-16)
	converged(solution) || @warn "Low precision!"
	return solution.zero[1]
end

mean(s::Smoothed) = mean(s.data)

var(s::Smoothed) = s.h^2 + var(s.data; corrected=false)

std(s::Smoothed) = sqrt(var(s))

struct PDFSmoothed{T<:Smoothed} <: Function
	s::T
end

struct CDFSmoothed{T<:Smoothed} <: Function
	s::T
end

(p::PDFSmoothed)(x::Real) = 
	mean(@. exp(-((x-p.s.data)/p.s.h)^2)) / (sqrt(2*pi) * p.s.h)

(P::CDFSmoothed)(x::Real) = 
	(1 + mean(@. erf((x-P.s.data)/(P.s.h*sqrt(2))))) / 2

pdf(s::Smoothed) = PDFSmoothed(s)
cdf(s::Smoothed) = CDFSmoothed(s)

# TODO: entropy(smt::Smoothed)
