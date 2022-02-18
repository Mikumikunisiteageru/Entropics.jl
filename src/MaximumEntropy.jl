module MaximumEntropy

export maxent

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
