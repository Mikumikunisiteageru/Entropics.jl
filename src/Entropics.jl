# src/Entropics.jl

module Entropics
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end Entropics

export support
export pdf, cdf
export mean, moment2, var, std
export quantile, median
export entropy
export skewness, kurtosis
export rescale
export sample
export maxendist
export bound
export smooth
# export kldiverg

using Base.Math: @horner
using Base: splat

using Cuba
using NLsolve: nlsolve, converged
using Optim: optimize, minimizer, Fminbox, Options

using SpecialFunctions # erf, erfc, erfi, erfinv

include("math.jl")
include("types.jl")
include("maxendist.jl")
include("sample.jl")

end # module Entropics
