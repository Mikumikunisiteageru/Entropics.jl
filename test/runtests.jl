# runtests.jl
# 20220225

using MaximumEntropy
using Test

@test !isnothing(maxent(min=0, max=1))
@test !isnothing(maxent(min=0, max=1, mean=0.5))
@test !isnothing(maxent(min=0, max=1, mean=0.3))
@test !isnothing(maxent(min=0, max=1, median=0.2))
@test !isnothing(maxent(min=0, max=1, median=0.2, mean=0.5))
@test !isnothing(maxent(min=0, max=1, median=0.2, mean=0.3))
@test !isnothing(maxent(min=0, max=1, mean=0.5, std=0.1))
@test !isnothing(maxent(min=0, max=1, mean=0.3, std=0.1))
