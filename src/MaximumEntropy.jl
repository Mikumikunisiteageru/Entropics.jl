module MaximumEntropy

export maxent

using Roots: find_zero
using NLsolve: nlsolve
using SpecialFunctions: erf

# Known min, max
function uniform(min_::T, max_::T, left_open=false) where {T <: AbstractFloat}
	range_ = max_ - min_
	isnan(range_) && error("Interval [$min_, $max_] has no meaning!")
	isinf(range_) && error("Interval [$min_, $max_] has infinite length!")
	iszero(range_) && error("Interval [$min_, $max_] has zero length!")
	range_ < 0 && error("Interval [$min_, $max_] is reversed!")
	if left_open
		return x -> min_ < x <= max_ ?
			one(T) / (max_ - min_) : zero(T)
	else
		return x -> min_ <= x <= max_ ?
			one(T) / (max_ - min_) : zero(T)
	end
end
uniform(min_, max_) = uniform(float(min_), float(max_))

# Known min, max, median
function uniform_piecewise(min_::T, median_::T, max_::T) where {T <: AbstractFloat}
	uniform_1 = uniform(min_, median_)
	uniform_2 = uniform(medina_, max_, true)
	x -> (uniform_1(x) + uniform_2(x)) / 2
end
uniform_piecewise(min_, median_, max_) = 
	uniform_piecewise(float(min_), float(median_), float(max_))

function cothminv(x)
	abs(x) > 0.5 && return coth(x) - 1/x
	x * evalpoly(x^2, 
		[+3.3333333333333333e-01, 2.2222222222222223e-02,
		 +2.1164021164021165e-03, 2.1164021164021165e-04,
		 +2.1377799155576935e-05, 2.1644042808063972e-06,
		 +2.1925947851873778e-07, 2.2214608789979678e-08,
		 +2.2507846516808994e-09, 2.2805151204592183e-10,
		 +2.3106432599002627e-11, 2.3411706819824886e-12,
		 +2.3721017400233653e-13, 2.4034415333307705e-14,
		 +2.4351954029183367e-15, 2.4673688045172075e-16,
		 +2.4999672771220810e-17, 2.5329964357406350e-18,
		 +2.5664619702826290e-19, 2.6003696460137274e-20])
end

function meantruncexp(lambda, a, b)
	isinf(b) && b > 0 && return a - 1/lambda
	d = (b - a) / 2
	a + d * (1 + cothminv(d * lambda))
end

function me_e_p(min_::T, max_::T, mean_::T, median_::T) where {T <: AbstractFloat}
	t = 2*mean_ - median_ - (min_+max_)/2
	d1 = (median_ - min_) / 2
	d2 = (max_ - median_) / 2
	f(g) = d1 * cothminv(d1 * g) + d2 * cothminv(d2 * g) - t
	gamma = find_zero(f, 0)
	gamma < 1e-10 &&
		return uniform_piecewise(min_, max_, median_)
	c1 = gamma / (2 * (exp(median_*gamma) - exp(min_*gamma)))
	c2 = gamma / (2 * (exp(max_*gamma) - exp(median_*gamma)))
	x -> min_ <= x <= median_ ? c1 * exp(gamma * x) :
		 median_ < x <= max_ ? c2 * exp(gamma * x) : 0
end

phi(x) = exp(-0.5 * x^2) / sqrt(2*pi)
Phi(x) = 0.5 * (1 + erf(x/sqrt(2)))
function normal_truncated(min_, max_, mean_, std_)
	var_ = std_ ^ 2
	function f!(F, p)
		mu = p[1]
		sigma = p[2]
		alpha = (min_ - mu) / sigma
		beta = (max_ - mu) / sigma
		phi_alpha = phi(alpha)
		phi_beta = phi(beta)
		z = phi(beta) - phi(alpha)
		Z = Phi(beta) - Phi(alpha)
		Q = z / Z
		F[1] = mu - sigma * Q - mean_
		F[2] = sigma^2 * (1 + (alpha*phi_alpha - beta*phi_beta)/Z - Q^2) - var_
	end
	mu, sigma = nlsolve(f!, [0., 1.]).zero
	F = zeros(2)
	f!(F, [mu, sigma])
	all(abs.(F) .< 1e-8) || error("No solution!")
	alpha = (min_ - mu) / sigma
	beta = (max_ - mu) / sigma
	Z = Phi(beta) - Phi(alpha)
	# entropy = log(sigma*Z*sqrt(2*pi*exp(1))+(alpha*phi(alpha)-beta*phi(beta))/(2*Z))
	# println(entropy)
	x -> min_ <= x <= max_ ?
		phi((x-mu)/sigma) / (sigma * Z) : 0
end


function maxent(; min=0, max=150, mean=nothing, median=nothing, std=nothing)
	arg_type = 0b111 - sum(isnothing.((mean, median, std)) .<< (2:-1:0))
	if arg_type == 0
		return uniform(min, max)
	elseif arg_type == 1
		error("Argument `std` given with neither `mean` nor `median`!")
	elseif arg_type == 2
		return uniform_piecewise(min, median, max)
	elseif arg_type == 3
	elseif arg_type == 4
	elseif arg_type == 5
	elseif arg_type == 6
	elseif arg_type == 7
	end
end

end # module
