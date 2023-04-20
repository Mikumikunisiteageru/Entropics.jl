var documenterSearchIndex = {"docs":
[{"location":"#Entropics.jl","page":"Entropics.jl","title":"Entropics.jl","text":"","category":"section"},{"location":"","page":"Entropics.jl","title":"Entropics.jl","text":"Entropics","category":"page"},{"location":"#Entropics","page":"Entropics.jl","title":"Entropics","text":"Entropics.jl\n\n(Image: Documentation) (Image: Documentation) (Image: CI) (Image: Codecov) (Image: Aqua.jl Quality Assurance)\n\nEntropics.jl is a Julia package for computing the maximum-entropy distribution bounded on a close interval with given median, mean, and variance values.\n\nThe package in still under development.\n\nAn article introducing this package and its interesting applications is also in preparation.\n\nExamples\n\nThe most important function provided in Entropics.jl is maxendist, which computes a maximum-entropy distribution with given conditions. For example, we may find the maximum-entropy distribution on interval 01 with median 03 and mean 035, and then examine its median, mean, variance, skewness, kurtosis, and entropy.\n\njulia> using Entropics\n\njulia> d = maxendist(0, 1; median=0.3, mean=0.35)\nEntropics.MED110{Float64}(0.0, 1.0, 0.3, -2.134967865645346, 0.8140360297855087, 0.9598690412098198)\n\njulia> median(d)\n0.3\n\njulia> mean(d)\n0.35\n\njulia> var(d)\n0.06861280728114773\n\njulia> skewness(d)\n0.6318688292278568\n\njulia> kurtosis(d)\n2.405231674579595\n\njulia> entropy(d)\n-0.13971378252179312\n\n\n\n\n\n","category":"module"},{"location":"#Mathematical-base","page":"Entropics.jl","title":"Mathematical base","text":"","category":"section"},{"location":"","page":"Entropics.jl","title":"Entropics.jl","text":"Entropics.binaryroot\nEntropics.secantroot\nEntropics.integrate\nEntropics.xlog\nEntropics.erfiinv\nEntropics.cothminv\nEntropics.diffexp\nEntropics.diffxexp\nEntropics.diffxxexp\nEntropics.diffxexpf\nEntropics.differf\nEntropics.differfi\nEntropics.intexp\nEntropics.intxexp\nEntropics.intxxexp\nEntropics.invintexp\nEntropics.intqexp","category":"page"},{"location":"#Entropics.binaryroot","page":"Entropics.jl","title":"Entropics.binaryroot","text":"binaryroot(f::Function, a::Real, b::Real)\n\nSolve the equation f(x) = 0 using binary search method.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.secantroot","page":"Entropics.jl","title":"Entropics.secantroot","text":"secantroot(f::Function, a::Real, b::Real)\n\nSolve the equation f(x) = 0 using secant method from given initial values.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.integrate","page":"Entropics.jl","title":"Entropics.integrate","text":"integrate(f, a, b; abstol=1e-12, reltol=1e-12)\n\nCompute int_a^b f(x) operatornamedx with Cuhre algorithm in Cuba.jl.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.xlog","page":"Entropics.jl","title":"Entropics.xlog","text":"xlog(x::AbstractFloat)\nxlog(x::Real)\n\nCompute x * log(x), with a convention at x=0 (returning 0).\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.erfiinv","page":"Entropics.jl","title":"Entropics.erfiinv","text":"erfiinv(y::AbstractFloat)\nerfiinv(y::Real)\n\nThe inverse of erfi, finding x satisfying  y = operatornameerfi(x).\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.cothminv","page":"Entropics.jl","title":"Entropics.cothminv","text":"cothminv(x::T)\ncothminv(x::Real)\n\nCompute coth(x) - 1x in a numerically stable scheme.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.diffexp","page":"Entropics.jl","title":"Entropics.diffexp","text":"diffexp(b::T, a::T) where {T<:AbstractFloat}\ndiffexp(b::AbstractFloat, a::AbstractFloat)\ndiffexp(b::Real, a::Real)\n\nCompute exp(b) - exp(a) in a numerically stable scheme.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.diffxexp","page":"Entropics.jl","title":"Entropics.diffxexp","text":"diffxexp(b::T, a::T) where {T<:AbstractFloat}\ndiffxexp(b::AbstractFloat, a::AbstractFloat)\ndiffxexp(b::Real, a::Real)\n\nCompute b exp(b) - a exp(a) in a numerically stable scheme.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.diffxxexp","page":"Entropics.jl","title":"Entropics.diffxxexp","text":"diffxxexp(b::T, a::T) where {T<:AbstractFloat}\ndiffxxexp(b::AbstractFloat, a::AbstractFloat)\ndiffxxexp(b::Real, a::Real)\n\nCompute b^2 exp(b) - a^2 exp(a) in a numerically stable scheme.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.diffxexpf","page":"Entropics.jl","title":"Entropics.diffxexpf","text":"diffxexpf(f::Function, b::T, a::T) where {T<:AbstractFloat}\ndiffxexpf(f::Function, b::AbstractFloat, a::AbstractFloat)\ndiffxexpf(f::Function, b::Real, a::Real)\n\nCompute b exp(f(b)) - a exp(f(a)) in a numerically stable scheme.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.differf","page":"Entropics.jl","title":"Entropics.differf","text":"differf(b::T, a::T) where {T<:AbstractFloat}\ndifferf(b::AbstractFloat, a::AbstractFloat)\ndifferf(b::Real, a::Real)\n\nCompute erf(b) - erf(a) in a numerically stable scheme.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.differfi","page":"Entropics.jl","title":"Entropics.differfi","text":"differfi(b::T, a::T) where {T<:AbstractFloat}\ndifferfi(b::AbstractFloat, a::AbstractFloat)\ndifferfi(b::Real, a::Real)\n\nCompute erfi(b) - erfi(a) in a numerically stable scheme.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.intexp","page":"Entropics.jl","title":"Entropics.intexp","text":"intexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat}\nintexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, \n\ta::AbstractFloat, b::AbstractFloat)\nintexp(A::Real, B::Real, C::Real, a::Real, b::Real)\n\nCompute int_a^b exp(A*x^2 + B*x + C) operatornamedx.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.intxexp","page":"Entropics.jl","title":"Entropics.intxexp","text":"intxexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat}\nintxexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, \n\ta::AbstractFloat, b::AbstractFloat)\nintxexp(A::Real, B::Real, C::Real, a::Real, b::Real)\n\nCompute int_a^b x * exp(A*x^2 + B*x + C) operatornamedx.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.intxxexp","page":"Entropics.jl","title":"Entropics.intxxexp","text":"intxxexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat}\nintxxexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, \n\ta::AbstractFloat, b::AbstractFloat)\nintxxexp(A::Real, B::Real, C::Real, a::Real, b::Real)\n\nCompute int_a^b x^2 * exp(A*x^2 + B*x + C) operatornamedx.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.invintexp","page":"Entropics.jl","title":"Entropics.invintexp","text":"invintexp(A::T, B::T, C::T, a::T, p::T) where {T<:AbstractFloat}\ninvintexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, \n\ta::AbstractFloat, p::AbstractFloat)\ninvintexp(A::Real, B::Real, C::Real, a::Real, p::Real)\n\nThe inverse of intexp(A, B, C, a, b) with respect to b, which finds b  satisfying p = int_a^b exp(A*x^2 + B*x + C) operatornamedx.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.intqexp","page":"Entropics.jl","title":"Entropics.intqexp","text":"intqexp(A::T, B::T, C::T, a::T, b::T) where {T<:AbstractFloat}\nintqexp(A::AbstractFloat, B::AbstractFloat, C::AbstractFloat, \n\ta::AbstractFloat, b::AbstractFloat)\nintqexp(A::Real, B::Real, C::Real, a::Real, b::Real)\n\nCompute - int_a^b f(x) * exp(f(x)) operatornamedx, where  f(x) = A*x^2 + B*x + C.\n\n\n\n\n\n","category":"function"},{"location":"#Types","page":"Entropics.jl","title":"Types","text":"","category":"section"},{"location":"","page":"Entropics.jl","title":"Entropics.jl","text":"Entropics.PDF\nEntropics.pdf\nEntropics.pab\nEntropics.pamb\nEntropics.CDF\nEntropics.Pab\nEntropics.Pamb\nEntropics.quantile\nEntropics.q01\nEntropics.support\nEntropics.median\nEntropics.mean\nEntropics.moment2\nEntropics.var\nEntropics.std\nEntropics.entropy\nEntropics.skewness\nEntropics.kurtosis\nEntropics.bound\nEntropics.smooth","category":"page"},{"location":"#Entropics.PDF","page":"Entropics.jl","title":"Entropics.PDF","text":"PDF{D<:Distribution}\n\nContainer type for a probability density function.\n\n\n\n\n\n","category":"type"},{"location":"#Entropics.pdf","page":"Entropics.jl","title":"Entropics.pdf","text":"pdf(d::Distribution)\n\nGet the probability density function of a distribution as a Julia function.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.pab","page":"Entropics.jl","title":"Entropics.pab","text":"pab(p::Function, x::Real, a::Real, b::Real)\n\nReturn \n\nzero, if x < a, \np(x), if a <= x <= b, \nzero, if x > b.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.pamb","page":"Entropics.jl","title":"Entropics.pamb","text":"pamb(pa::Function, pb::Function, x::Real, a::Real, m::Real, b::Real)\n\nReturn \n\nzero, if x < a, \npa(x), if a <= x <= m, \npb(x), if m < x <= b, \nzero, if x > b.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.CDF","page":"Entropics.jl","title":"Entropics.CDF","text":"CDF{D<:Distribution}\n\nContainer type for a cumulative distribution function.\n\n\n\n\n\n","category":"type"},{"location":"#Entropics.Pab","page":"Entropics.jl","title":"Entropics.Pab","text":"Pab(P::Function, x::Real, a::Real, b::Real)\n\nReturn \n\nzero, if x < a, \nP(x), if a <= x <= b, \none, if x > b.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.Pamb","page":"Entropics.jl","title":"Entropics.Pamb","text":"Pamb(Pa::Function, Pb::Function, x::Real, a::Real, m::Real, b::Real)\n\nReturn \n\nzero, if x < a, \nPa(x), if a <= x <= m, \nPb(x) + 1/2, if m < x <= b, \none, if x > b.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.quantile","page":"Entropics.jl","title":"Entropics.quantile","text":"quantile(d::Distribution, p::Real, a::Real, b::Real)\nquantile(d::Distribution, p::Real)\nquantile(d::Distribution, pp::Vector{<:Real})\nquantile(d::Distribution)\n\nCompute the quantile value in a distribution. The last syntax creates a Julia  function similar to those from pdf and cdf that maps p to q(d, p).\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.q01","page":"Entropics.jl","title":"Entropics.q01","text":"q01(q::Function, p::Real)\n\nReturn q(p) if 0 <= p <= 1, or an error otherwise. \n\n\n\n\n\n","category":"function"},{"location":"#Entropics.support","page":"Entropics.jl","title":"Entropics.support","text":"support(d::Distribution)\n\nCompute the support (a, b) of a distribution. For unbounded distributions,  the support interval expands from negative infinity to positive infinity. \n\n\n\n\n\n","category":"function"},{"location":"#Entropics.median","page":"Entropics.jl","title":"Entropics.median","text":"median(d::Distribution)\n\nCompute the median of a distribution.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.mean","page":"Entropics.jl","title":"Entropics.mean","text":"mean(d::Distribution)\n\nCompute the mean of a distribution.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.moment2","page":"Entropics.jl","title":"Entropics.moment2","text":"moment2(d::Distribution)\n\nCompute the second moment (e.g. mathcalEX^2 for a random variable  X) of a distribution.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.var","page":"Entropics.jl","title":"Entropics.var","text":"var(d::Distribution; u=mean(d))\n\nCompute the variance of a distribution. The mean can be provided if known.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.std","page":"Entropics.jl","title":"Entropics.std","text":"std(d::Distribution; u=mean(d))\n\nCompute the standard deviation of a distribution. The mean can be provided if  known.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.entropy","page":"Entropics.jl","title":"Entropics.entropy","text":"entropy(d::Bounded; u=mean(d))\n\nCompute the differential entropy of a bounded distribution. \n\n\n\n\n\n","category":"function"},{"location":"#Entropics.skewness","page":"Entropics.jl","title":"Entropics.skewness","text":"skewness(d::Bounded; u=mean(d), s=std(d))\n\nCompute the skewness of a bounded distribution. \n\n\n\n\n\n","category":"function"},{"location":"#Entropics.kurtosis","page":"Entropics.jl","title":"Entropics.kurtosis","text":"kurtosis(d::Bounded; u=mean(d), s=std(d))\n\nCompute the kurtosis of a bounded distribution. \n\n\n\n\n\n","category":"function"},{"location":"#Entropics.bound","page":"Entropics.jl","title":"Entropics.bound","text":"bound(d::Distribution, a::Real, b::Real)\nbound(d::Bounded)\nbound(d::Smooth{T,<:Sample{T}}) where {T<:AbstractFloat}\n\nBound or truncate a distribution by interval a b, and then return the  new bounded distribution. Especially, bounding a bounded distribution without  giving a and b returns the bounded distribution itself; when bounding a  smoothed sample (see smooth) without giving a and b, the interval is set  to the extrema of the original sample.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.smooth","page":"Entropics.jl","title":"Entropics.smooth","text":"smooth(d::Distribution, h::Real)\nsmooth(d::Smooth)\nsmooth(d::Sample)\nsmooth(v::Vector{<:Real})\n\nSmooth a distribution with a Gaussian kernal with bandwidth h. Especially,  smoothing a smoothed distribution without giving h returns the smoothed  distribution itself; when smoothing a sample or a vector of reals, the  bandwidth is calculated using Silverman (1986)'s rule of thumb (see  helrot for more details).\n\n\n\n\n\n","category":"function"},{"location":"#Maximum-entropy-distributions","page":"Entropics.jl","title":"Maximum-entropy distributions","text":"","category":"section"},{"location":"","page":"Entropics.jl","title":"Entropics.jl","text":"Entropics.urange\nEntropics.maxendist\nEntropics.xab\nEntropics.xabm\nEntropics.xabu\nEntropics.xs\nEntropics.xv\nEntropics.xsv\nEntropics.xabuv\nEntropics.xabmu\nEntropics.xabmuv","category":"page"},{"location":"#Entropics.urange","page":"Entropics.jl","title":"Entropics.urange","text":"urange(a::Real, b::Real, m::Real, v::Real)\n\nReturn the possible range of mean of a distribution bounded on a b  with median m and variance v as an interval (ul ur).\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.maxendist","page":"Entropics.jl","title":"Entropics.maxendist","text":"maxendist(a::Real, b::Real; \n\tmedian::Union{Real,Nothing}=nothing, \n\tmean::Union{Real,Nothing}=nothing, \n\tstd::Union{Real,Nothing}=nothing, var::Union{Real,Nothing}=nothing\n\nCompute the maximum-entropy distribution bounded on a b with given  median, mean, standard deviation, and/or variance.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.xab","page":"Entropics.jl","title":"Entropics.xab","text":"xab(a::Real, b::Real)\n\nCheck if a b is a well-defined real interval with positive length.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.xabm","page":"Entropics.jl","title":"Entropics.xabm","text":"xabm(a::Real, b::Real, m::Real)\n\nCheck if median m is possible for a continuous distribution bounded on  a b. See Theorem 100 in the article.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.xabu","page":"Entropics.jl","title":"Entropics.xabu","text":"xabu(a::Real, b::Real, u::Real)\n\nCheck if mean u is possible for a continuous distribution bounded on  a b. See Theorem 010 in the article.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.xs","page":"Entropics.jl","title":"Entropics.xs","text":"xs(s::Real)\n\nCheck if standard deviation s is possible for a continuous distribution.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.xv","page":"Entropics.jl","title":"Entropics.xv","text":"xv(v::Real)\n\nCheck if variance v is possible for a continuous distribution.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.xsv","page":"Entropics.jl","title":"Entropics.xsv","text":"xsv(s::Real, v::Real)\n\nCheck if standard deviation s and variance v is consistent for a  distribution. If so, the variance is returned.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.xabuv","page":"Entropics.jl","title":"Entropics.xabuv","text":"xabuv(a::Real, b::Real, u::Union{Real, Nothing}, v::Union{Real, Nothing})\n\nCheck if mean u and variance v is simultaneously possible for a continuous  distribution bounded on a b. See Theorem 001 (Popoviciu(1935)'s  inequality) and Theorem 011 (Bhatia & Davis (2000)'s inequality) in the  article.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.xabmu","page":"Entropics.jl","title":"Entropics.xabmu","text":"xabmu(a::Real, b::Real, m::Union{Real, Nothing}, u::Union{Real, Nothing})\n\nCheck if median m and mean u is simultaneously possible for a continuous  distribution bounded on a b. See Theorem 110 in the article.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.xabmuv","page":"Entropics.jl","title":"Entropics.xabmuv","text":"xabmuv(a::Real, b::Real, m::Union{Real, Nothing}, \n\tu::Union{Real, Nothing}, v::Union{Real, Nothing})\n\nCheck if median m, mean u, and variance v is simultaneously possible  for a continuous distribution bounded on a b. See Theorem 111a and  Theorem 111b in the article.\n\n\n\n\n\n","category":"function"},{"location":"#Sampling","page":"Entropics.jl","title":"Sampling","text":"","category":"section"},{"location":"","page":"Entropics.jl","title":"Entropics.jl","text":"Entropics.sample\nEntropics.helrot","category":"page"},{"location":"#Entropics.sample","page":"Entropics.jl","title":"Entropics.sample","text":"sample(v::Vector{<:Real})\n\nCreate a sample as a distribution.\n\n\n\n\n\n","category":"function"},{"location":"#Entropics.helrot","page":"Entropics.jl","title":"Entropics.helrot","text":"helrot(d::Sample{T}; useiqr::Bool=true) where {T<:AbstractFloat}\n\nCompute the optimal bandwidth using Silverman (1986)'s rule of thumb for  smoothing with a Gaussian kernal. If useiqr is true, then the version  using interquartile range (IQR) is applied.\n\n\n\n\n\n","category":"function"}]
}