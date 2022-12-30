# src/smmaxendist.jl

### MATHEMATIC FUNCTIONS

function intexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat}
	if iszero(A)
		if iszero(B)
			return (b-a) * exp(C)
		else # B != 0
			return exp(B * a + C) * expm1(B * (b-a)) / B
		end
	elseif A > 0
		sA = sqrt(A)
		return -((@. erfi(sA * [b, a] + B / (2 * sA)))...) * 
			exp(C - B^2 / (4 * A)) * sqrt(T(pi)) / (2 * sA)
	else # A < 0
		sA = sqrt(-A)
		return differf((@. sA * [b, a] - B / (2 * sA))...) * 
			exp(C - B^2 / (4 * A)) * sqrt(T(pi)) / (2 * sA)
	end
end
intexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, a::AbstractFloat, 
	b::AbstractFloat) = intexp(promote(A, B, C, a, b)...)
intexp(A::Real, B::Real, C::Real, a::Real, b::Real) = 
	intexp(float.((A, B, C, a, b))...)

### SMED000

struct SMED000{T<:AbstractFloat} <: SmMaxEnDist
	d::MED000{T}
	h::T
end
SMED000(d::MED000{T}, h::Real) where {T<:AbstractFloat} = SMED000(d, T(h))

smooth(d::MED000, h::Real) = SMED000(d, h)

(p::PDF{SMED000{T}})(x::T) where {T<:AbstractFloat} = 
	differf((@.([p.d.d.b, p.d.d.a] - x) / (sqrt(T(2)) * p.d.h))...) / 
		(2 * (p.d.d.b - p.d.d.a))
(p::PDF{SMED000{T}})(x::Real) where {T<:AbstractFloat} = p(T(x))

### SMED100

struct SMED100{T<:AbstractFloat} <: SmMaxEnDist
	d::MED100{T}
	h::T
end
SMED100(d::MED100{T}, h::Real) where {T<:AbstractFloat} = SMED100(d, T(h))

smooth(d::MED100, h::Real) = SMED100(d, h)

(p::PDF{SMED100{T}})(x::T) where {T<:AbstractFloat} = 
	differf((@.([p.d.d.m, p.d.d.a] - x) / (sqrt(T(2)) * p.d.h))...) / 
		(4 * (p.d.d.m - p.d.d.a)) + 
	differf((@.([p.d.d.b, p.d.d.m] - x) / (sqrt(T(2)) * p.d.h))...) / 
		(4 * (p.d.d.b - p.d.d.m))
(p::PDF{SMED100{T}})(x::Real) where {T<:AbstractFloat} = p(T(x))

### SMED010

struct SMED010{T<:AbstractFloat} <: SmMaxEnDist
	d::MED010{T}
	h::T
end
SMED010(d::MED010{T}, h::Real) where {T<:AbstractFloat} = SMED010(d, T(h))

smooth(d::MED010, h::Real) = SMED010(d, h)

(p::PDF{SMED010{T}})(x::T) where {T<:AbstractFloat} = 
	intexp(- p.d.h^-2 / 2, p.d.d.g + x / p.d.h^2, - (x / p.d.h) ^ 2 / 2, 
		p.d.d.a, p.d.d.b) * p.d.d.k / (sqrt(T(2*pi)) * p.d.h)
(p::PDF{SMED010{T}})(x::Real) where {T<:AbstractFloat} = p(T(x))

### SMED110

struct SMED110{T<:AbstractFloat} <: SmMaxEnDist
	d::MED110{T}
	h::T
end
SMED110(d::MED110{T}, h::Real) where {T<:AbstractFloat} = SMED110(d, T(h))

smooth(d::MED110, h::Real) = SMED110(d, h)

(p::PDF{SMED110{T}})(x::T) where {T<:AbstractFloat} = 
	intexp(- p.d.h^-2 / 2, p.d.d.g + x / p.d.h^2, - (x / p.d.h) ^ 2 / 2, 
		p.d.d.a, p.d.d.m) * p.d.d.ka / (sqrt(T(2*pi)) * p.d.h) + 
	intexp(- p.d.h^-2 / 2, p.d.d.g + x / p.d.h^2, - (x / p.d.h) ^ 2 / 2, 
		p.d.d.m, p.d.d.b) * p.d.d.kb / (sqrt(T(2*pi)) * p.d.h)
(p::PDF{SMED110{T}})(x::Real) where {T<:AbstractFloat} = p(T(x))

### SMED011N

struct SMED011N{T<:AbstractFloat} <: SmMaxEnDist
	d::MED011N{T}
	h::T
end
SMED011N(d::MED011N{T}, h::Real) where {T<:AbstractFloat} = SMED011N(d, T(h))

smooth(d::MED011N, h::Real) = SMED011N(d, h)

(p::PDF{SMED011N{T}})(x::T) where {T<:AbstractFloat} = 
	intexp(- p.d.h^-2 / 2 - p.d.d.s^-2, 
		2 * p.d.d.o / p.d.d.s^2 + x / p.d.h^2, 
		- (p.d.d.o / p.d.d.s) ^ 2 - (x / p.d.h) ^ 2 / 2, 
		p.d.d.a, p.d.d.b) * p.d.d.k / (sqrt(T(2*pi)) * p.d.h)
(p::PDF{SMED011N{T}})(x::Real) where {T<:AbstractFloat} = p(T(x))

### SMED011P

struct SMED011P{T<:AbstractFloat} <: SmMaxEnDist
	d::MED011P{T}
	h::T
end
SMED011P(d::MED011P{T}, h::Real) where {T<:AbstractFloat} = SMED011P(d, T(h))

smooth(d::MED011P, h::Real) = SMED011P(d, h)

(p::PDF{SMED011P{T}})(x::T) where {T<:AbstractFloat} = 
	intexp(- p.d.h^-2 / 2 + p.d.d.s^-2, 
		-2 * p.d.d.o / p.d.d.s^2 + x / p.d.h^2, 
		(p.d.d.o / p.d.d.s) ^ 2 - (x / p.d.h) ^ 2 / 2, 
		p.d.d.a, p.d.d.b) * p.d.d.k / (sqrt(T(2*pi)) * p.d.h)
(p::PDF{SMED011P{T}})(x::Real) where {T<:AbstractFloat} = p(T(x))

### SMED111N

struct SMED111N{T<:AbstractFloat} <: SmMaxEnDist
	d::MED111N{T}
	h::T
end
SMED111N(d::MED111N{T}, h::Real) where {T<:AbstractFloat} = SMED111N(d, T(h))

smooth(d::MED111N, h::Real) = SMED111N(d, h)

(p::PDF{SMED111N{T}})(x::T) where {T<:AbstractFloat} = 
	intexp(- p.d.h^-2 / 2 - p.d.d.s^-2, 
		2 * p.d.d.o / p.d.d.s^2 + x / p.d.h^2, 
		- (p.d.d.o / p.d.d.s) ^ 2 - (x / p.d.h) ^ 2 / 2, 
		p.d.d.a, p.d.d.m) * p.d.d.ka / (sqrt(T(2*pi)) * p.d.h) + 
	intexp(- p.d.h^-2 / 2 - p.d.d.s^-2, 
		2 * p.d.d.o / p.d.d.s^2 + x / p.d.h^2, 
		- (p.d.d.o / p.d.d.s) ^ 2 - (x / p.d.h) ^ 2 / 2, 
		p.d.d.m, p.d.d.b) * p.d.d.kb / (sqrt(T(2*pi)) * p.d.h)
(p::PDF{SMED111N{T}})(x::Real) where {T<:AbstractFloat} = p(T(x))

### SMED111P

struct SMED111P{T<:AbstractFloat} <: SmMaxEnDist
	d::MED111P{T}
	h::T
end
SMED111P(d::MED111P{T}, h::Real) where {T<:AbstractFloat} = SMED111P(d, T(h))

smooth(d::MED111P, h::Real) = SMED111P(d, h)

(p::PDF{SMED111P{T}})(x::T) where {T<:AbstractFloat} = 
	intexp(- p.d.h^-2 / 2 + p.d.d.s^-2, 
		-2 * p.d.d.o / p.d.d.s^2 + x / p.d.h^2, 
		(p.d.d.o / p.d.d.s) ^ 2 - (x / p.d.h) ^ 2 / 2, 
		p.d.d.a, p.d.d.m) * p.d.d.ka / (sqrt(T(2*pi)) * p.d.h) + 
	intexp(- p.d.h^-2 / 2 + p.d.d.s^-2, 
		-2 * p.d.d.o / p.d.d.s^2 + x / p.d.h^2, 
		(p.d.d.o / p.d.d.s) ^ 2 - (x / p.d.h) ^ 2 / 2, 
		p.d.d.m, p.d.d.b) * p.d.d.kb / (sqrt(T(2*pi)) * p.d.h)
(p::PDF{SMED111P{T}})(x::Real) where {T<:AbstractFloat} = p(T(x))
