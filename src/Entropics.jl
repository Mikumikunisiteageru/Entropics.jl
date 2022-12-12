# src/Entropics.jl

module Entropics

export maxendist
export median, mean, var, std, entropy, quantile, support, pdf, cdf
export PDF, CDF # old-fashioned
export smooth, kldiverg

using Base.Math: @horner

using Cuba
using NLsolve: nlsolve, converged
using Optim: optimize, minimizer, Fminbox, Options

using SpecialFunctions # erf, erfc, erfi, erfinv
import Statistics # median, mean, var, std, quantile

include("maxendist.jl")
include("smoothing.jl")

end # module Entropics
