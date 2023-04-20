# Entropics.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://Mikumikunisiteageru.github.io/Entropics.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://Mikumikunisiteageru.github.io/Entropics.jl/dev)
[![CI](https://github.com/Mikumikunisiteageru/Entropics.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Mikumikunisiteageru/Entropics.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/Mikumikunisiteageru/Entropics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Mikumikunisiteageru/Entropics.jl)
[![Aqua.jl Quality Assurance](https://img.shields.io/badge/Aquajl-%F0%9F%8C%A2-aqua.svg)](https://github.com/JuliaTesting/Aqua.jl)

Entropics.jl is a Julia package for computing the maximum-entropy distribution bounded on a close interval with given median, mean, and variance values.

The package in still under development.

An article introducing this package and its interesting applications is also in preparation.

## Examples

The most important function provided in Entropics.jl is `maxendist`, which computes a maximum-entropy distribution with given conditions. For example, we may find the maximum-entropy distribution on interval ``[0,1]`` with median ``0.3`` and mean ``0.35``, and then examine its median, mean, variance, skewness, kurtosis, and entropy.

```julia
julia> using Entropics

julia> d = maxendist(0, 1; median=0.3, mean=0.35)
Entropics.MED110{Float64}(0.0, 1.0, 0.3, -2.134967865645346, 0.8140360297855087, 0.9598690412098198)

julia> median(d)
0.3

julia> mean(d)
0.35

julia> var(d)
0.06861280728114773

julia> skewness(d)
0.6318688292278568

julia> kurtosis(d)
2.405231674579595

julia> entropy(d)
-0.13971378252179312
```
