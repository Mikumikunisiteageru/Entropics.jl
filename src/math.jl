# src/math.jl

"""
	binaryroot(f::Function, a::Real, b::Real)

Solve the equation ``f(x) = 0`` using binary search method.
"""
function binaryroot(f::Function, a::T, b::T) where {T<:AbstractFloat}
	fa, fb = f(a), f(b)
	iszero(fa) && return a
	iszero(fb) && return b
	sa, sb = signbit(fa), signbit(fb)
	xor(sa, sb) || error("`f(a)` and `f(b)` have the same sign!")
	while ! isapprox(a, b, rtol=eps(T)*2)
		m = (a + b) / 2
		fm = f(m)
		sm = signbit(fm)
		if xor(sa, sm)
			b, fb, sb = m, fm, sm
		else
			a, fa, sa = m, fm, sm
		end
	end
	return (a + b) / 2
end

"""
	secantroot(f::Function, a::Real, b::Real)

Solve the equation ``f(x) = 0`` using secant method from given initial values.
"""
function secantroot(f::Function, a::T, b::T) where {T<:AbstractFloat}
	fa = f(a)
	fb = f(b)
	e = eps(T)
	while ! isapprox(a, b; rtol=e*2)
		isapprox(fa, fb; rtol=e*10) && 
			(abs(fb) < e ? (return b) : error("Bad initial values!"))
		m = b - (b-a) / (fb-fa) * fb
		a, fa, b, fb = b, fb, m, f(m)
	end
	return b
end
secantroot(f::Function, a::AbstractFloat, b::AbstractFloat) = 
	secantroot(f, promote(a, b)...)
secantroot(f::Function, a::Real, b::Real) = secantroot(f, float(a), float(b))

"""
	integrate(f, a, b; abstol=1e-12, reltol=1e-12)

Compute ``\\int_a^b f(x) \\operatorname{d}x`` with Cuhre algorithm in Cuba.jl.
"""
function integrate(f, a, b; abstol=1e-12, reltol=1e-12)
	d = b - a
	g(x, y) = y[1] = f(a + d * x[1])
	result, err = cuhre(g; atol=abstol/d, rtol=reltol)
	return d * result[1]
end

"""
	xlog(x::AbstractFloat)
	xlog(x::Real)

Compute ``x * \\log(x)``, with a convention at ``x=0`` (returning ``0``).
"""
xlog(x::AbstractFloat) = iszero(x) ? zero(x) : x * log(x)
xlog(x::Real) = xlog(float(x))

"""
	erfiinv(y::AbstractFloat)
	erfiinv(y::Real)

The inverse of `erfi`, finding ``x`` satisfying 
``y = \\operatorname{erfi}(x)``.
"""
function erfiinv(y::AbstractFloat)
	iszero(y) && return y
	y < 0 && return -erfiinv(-y)
	function f!(output, param)
		output[1] = erfi(param[1]) - y
	end
	return nlsolve(f!, [sqrt(log(1 + y))]).zero[1]
end
erfiinv(y::Real) = erfiinv(float(y))

"""
	cothminv(x::T)
	cothminv(x::Real)

Compute ``\\coth(x) - 1/x`` in a numerically stable scheme.
"""
function cothminv(x::T) where {T<:AbstractFloat}
	abs(x) > 1.18 && return coth(x) - inv(x)
	return T(x * @horner(x^2, 
		+3.3333333333333333e-01, -2.2222222222222223e-02,
		+2.1164021164021165e-03, -2.1164021164021165e-04,
		+2.1377799155576935e-05, -2.1644042808063972e-06,
		+2.1925947851873778e-07, -2.2214608789979678e-08,
		+2.2507846516808994e-09, -2.2805151204592183e-10,
		+2.3106432599002627e-11, -2.3411706819824886e-12,
		+2.3721017400233653e-13, -2.4034415333307705e-14,
		+2.4351954029183367e-15, -2.4673688045172075e-16,
		+2.4999672771220810e-17, -2.5329964357406350e-18,
		+2.5664619702826290e-19, -2.6003696460137274e-20))
end
cothminv(x::Real) = cothminv(float(x))

"""
	diffexp(b::T, a::T) where {T<:AbstractFloat}
	diffexp(b::AbstractFloat, a::AbstractFloat)
	diffexp(b::Real, a::Real)

Compute ``\\exp(b) - \\exp(a)`` in a numerically stable scheme.
"""
diffexp(b::T, a::T) where {T<:AbstractFloat} = 
	a <= b ? exp(b) * -expm1(a-b) : exp(a) * expm1(b-a)
diffexp(b::AbstractFloat, a::AbstractFloat) = diffexp(promote(b, a)...)
diffexp(b::Real, a::Real) = diffexp(float(b), float(a))

"""
	diffxexp(b::T, a::T) where {T<:AbstractFloat}
	diffxexp(b::AbstractFloat, a::AbstractFloat)
	diffxexp(b::Real, a::Real)

Compute ``b \\exp(b) - a \\exp(a)`` in a numerically stable scheme.
"""
diffxexp(b::T, a::T) where {T<:AbstractFloat} = 
	a <= b ? exp(b) * ((b - a) - a * expm1(a-b)) : 
			 exp(a) * (b * expm1(b-a) - (a - b))
diffxexp(b::AbstractFloat, a::AbstractFloat) = diffxexp(promote(b, a)...)
diffxexp(b::Real, a::Real) = diffxexp(float(b), float(a))

"""
	diffxxexp(b::T, a::T) where {T<:AbstractFloat}
	diffxxexp(b::AbstractFloat, a::AbstractFloat)
	diffxxexp(b::Real, a::Real)

Compute ``b^2 \\exp(b) - a^2 \\exp(a)`` in a numerically stable scheme.
"""
diffxxexp(b::T, a::T) where {T<:AbstractFloat} = 
	a <= b ? exp(b) * ((b^2 - a^2) - a^2 * expm1(a-b)) : 
			 exp(a) * (b^2 * expm1(b-a) - (a^2 - b^2))
diffxxexp(b::AbstractFloat, a::AbstractFloat) = diffxxexp(promote(b, a)...)
diffxxexp(b::Real, a::Real) = diffxxexp(float(b), float(a))

"""
	diffxexpf(f::Function, b::T, a::T) where {T<:AbstractFloat}
	diffxexpf(f::Function, b::AbstractFloat, a::AbstractFloat)
	diffxexpf(f::Function, b::Real, a::Real)

Compute ``b \\exp(f(b)) - a \\exp(f(a))`` in a numerically stable scheme.
"""
diffxexpf(f::Function, b::T, a::T) where {T<:AbstractFloat} = 
	f(a) <= f(b) ? exp(f(b)) * ((b - a) - a * expm1(f(a)-f(b))) : 
				   exp(f(a)) * (b * expm1(f(b)-f(a)) - (a - b))
diffxexpf(f::Function, b::AbstractFloat, a::AbstractFloat) = 
	diffxexpf(f, promote(b, a)...)
diffxexpf(f::Function, b::Real, a::Real) = diffxexpf(f, float(b), float(a))

"""
	differf(b::T, a::T) where {T<:AbstractFloat}
	differf(b::AbstractFloat, a::AbstractFloat)
	differf(b::Real, a::Real)

Compute ``\\erf(b) - \\erf(a)`` in a numerically stable scheme.
"""
function differf(b::T, a::T) where {T<:AbstractFloat}
	if min(a, b) > 0.84
		return erfc(a) - erfc(b)
	elseif max(a, b) < -0.84
		return erfc(-b) - erfc(-a)
	else
		return erf(b) - erf(a)
	end
end
differf(b::AbstractFloat, a::AbstractFloat) = differf(promote(b, a)...)
differf(b::Real, a::Real) = differf(float(b), float(a))

"""
	differfi(b::T, a::T) where {T<:AbstractFloat}
	differfi(b::AbstractFloat, a::AbstractFloat)
	differfi(b::Real, a::Real)

Compute ``\\erfi(b) - \\erfi(a)`` in a numerically stable scheme.
"""
differfi(b::T, a::T) where {T<:AbstractFloat} = erfi(b) - erfi(a)
differfi(b::AbstractFloat, a::AbstractFloat) = differfi(promote(b, a)...)
differfi(b::Real, a::Real) = differfi(float(b), float(a))

"""
	intexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat}
	intexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, 
		a::AbstractFloat, b::AbstractFloat)
	intexp(A::Real, B::Real, C::Real, a::Real, b::Real)

Compute ``\\int_a^b \\exp(A*x^2 + B*x + C) \\operatorname{d}x``.
"""
function intexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat}
	if iszero(A)
		if iszero(B)
			return (b - a) * exp(C)
		else # B != 0
			return B \ diffexp(B*b+C, B*a+C)
		end
	elseif A > 0
		rA = sqrt(A)
		Bd2rA = B / (2 * rA)
		return (2*rA) \ exp(C - Bd2rA^2) * sqrt(T(pi)) * 
			differfi(rA*b+Bd2rA, rA*a+Bd2rA)
	else # A < 0
		rA = sqrt(-A)
		Bd2rA = B / (2 * rA)
		return (2*rA) \ exp(C + Bd2rA^2) * sqrt(T(pi)) * 
			differf(rA*b-Bd2rA, rA*a-Bd2rA)
	end
end
intexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, 
	a::AbstractFloat, b::AbstractFloat) = intexp(promote(A, B, C, a, b)...)
intexp(A::Real, B::Real, C::Real, a::Real, b::Real) = 
	intexp(float.((A, B, C, a, b))...)

"""
	intxexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat}
	intxexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, 
		a::AbstractFloat, b::AbstractFloat)
	intxexp(A::Real, B::Real, C::Real, a::Real, b::Real)

Compute ``\\int_a^b x * \\exp(A*x^2 + B*x + C) \\operatorname{d}x``.
"""
function intxexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat}
	if iszero(A)
		if iszero(B)
			return (b - a) * (b + a) / 2 * exp(C)
		else # B != 0
			fb, fa = B * b, B * a
			return B^2 \ exp(C) * (diffxexp(fb, fa) - diffexp(fb, fa))
		end
	elseif A > 0
		rA = sqrt(A)
		Bd2rA = B / (2 * rA)
		fb, fa = rA * b + Bd2rA, rA * a + Bd2rA
		return (2*A) \ exp(C - Bd2rA^2) * 
			(diffexp(fb^2, fa^2) - sqrt(T(pi)) * Bd2rA * differfi(fb, fa))
	else # A < 0
		rA = sqrt(-A)
		Bd2rA = B / (2 * rA)
		fb, fa = rA * b - Bd2rA, rA * a - Bd2rA
		return (2*A) \ exp(C + Bd2rA^2) * 
			(diffexp(-fb^2, -fa^2) - sqrt(T(pi)) * Bd2rA * differf(fb, fa))
	end
end
intxexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, 
	a::AbstractFloat, b::AbstractFloat) = intxexp(promote(A, B, C, a, b)...)
intxexp(A::Real, B::Real, C::Real, a::Real, b::Real) = 
	intxexp(float.((A, B, C, a, b))...)

"""
	intxxexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat}
	intxxexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, 
		a::AbstractFloat, b::AbstractFloat)
	intxxexp(A::Real, B::Real, C::Real, a::Real, b::Real)

Compute ``\\int_a^b x^2 * \\exp(A*x^2 + B*x + C) \\operatorname{d}x``.
"""
function intxxexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat}
	if iszero(A)
		if iszero(B)
			return (b - a) * (b * (b+a) + a^2) / 3 * exp(C)
		else # B != 0
			fb, fa = B * b, B * a
			return B^3 \ exp(C) * (diffxxexp(fb, fa) - 2 * (
				diffxexp(fb, fa) - diffexp(fb, fa)))
		end
	elseif A > 0
		rA = sqrt(A)
		Bd2rA = B / (2 * rA)
		fb, fa = rA * b + Bd2rA, rA * a + Bd2rA
		return (4*A^2) \ exp(C - Bd2rA^2) * 
			(2*rA * diffxexpf(x->x^2, fb, fa) - 2*B * diffexp(fb^2, fa^2) - 
			sqrt(T(pi)) * rA*(1-2*Bd2rA^2) * differfi(fb, fa))
	else # A < 0
		rA = sqrt(-A)
		Bd2rA = B / (2 * rA)
		fb, fa = rA * b - Bd2rA, rA * a - Bd2rA
		return (4*A^2) \ exp(C + Bd2rA^2) * 
			(-2*rA * diffxexpf(x->-x^2, fb, fa) - 2*B * diffexp(-fb^2, -fa^2) + 
			sqrt(T(pi)) * rA*(1+2*Bd2rA^2) * differf(fb, fa))
	end
end
intxxexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, 
	a::AbstractFloat, b::AbstractFloat) = intxxexp(promote(A, B, C, a, b)...)
intxxexp(A::Real, B::Real, C::Real, a::Real, b::Real) = 
	intxxexp(float.((A, B, C, a, b))...)

"""
	invintexp(A::T, B::T, C::T, a::T, p::T) where {T<:AbstractFloat}
	invintexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, 
		a::AbstractFloat, p::AbstractFloat)
	invintexp(A::Real, B::Real, C::Real, a::Real, p::Real)

The inverse of `intexp(A, B, C, a, b)` with respect to `b`, which finds ``b`` 
satisfying ``p = \\int_a^b \\exp(A*x^2 + B*x + C) \\operatorname{d}x``.
"""
function invintexp(A::T, B::T, C::T, a::T, p::T) where {T<:AbstractFloat}
	if iszero(A)
		if iszero(B)
			return a + p / exp(C)
		else # B != 0
			r = B * a
			s = B * p / exp(C)
			return B \ (abs(B) >= 1 ? log(exp(r)+s) : log1p(expm1(r)+s))
		end
	elseif A > 0
		rA = sqrt(A)
		Bd2rA = B / (2 * rA)
		return rA \ (erfiinv(erfi(rA*a+Bd2rA) + 
			p * (2*rA) / (exp(C - Bd2rA^2) * sqrt(T(pi)))) - Bd2rA)
	else # A < 0
		rA = sqrt(-A)
		Bd2rA = B / (2 * rA)
		return rA \ (erfinv(erf(rA*a-Bd2rA) + 
			p * (2*rA) / (exp(C + Bd2rA^2) * sqrt(T(pi)))) + Bd2rA)
	end
end
invintexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, 
	a::AbstractFloat, p::AbstractFloat) = intexp(promote(A, B, C, a, p)...)
invintexp(A::Real, B::Real, C::Real, a::Real, p::Real) = 
	intexp(float.((A, B, C, a, p))...)

"""
	intqexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat}
	intqexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, 
		a::AbstractFloat, b::AbstractFloat)
	intqexp(A::Real, B::Real, C::Real, a::Real, b::Real)

Compute ``- \\int_a^b f(x) * \\exp(f(x)) \\operatorname{d}x``, where 
``f(x) = A*x^2 + B*x + C``.
"""
intqexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat} = 
	- sum((A, B, C) .* 
		(((A, B, C, a, b),) .|> Base.splat.((intxxexp, intxexp, intexp))))
intqexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, 
	a::AbstractFloat, b::AbstractFloat) = intqexp(promote(A, B, C, a, b)...)
intqexp(A::Real, B::Real, C::Real, a::Real, b::Real) = 
	intqexp(float.((A, B, C, a, b))...)
