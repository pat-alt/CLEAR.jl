module Evaluation

using ..CounterfactualExplanations
using DataFrames
using ..Generators
using ..Models
using LinearAlgebra: LinearAlgebra
using Statistics

abstract type AbstractDivergenceMetric <: AbstractMeasure end

include("serialization.jl")
include("divergence/divergence.jl")

export MMD

include("measures.jl")
include("benchmark.jl")
include("evaluate.jl")

export global_serializer, Serializer, NullSerializer, _serialization_state
export global_output_identifier, DefaultOutputIdentifier, _output_id, get_global_output_id
export ExplicitOutputIdentifier
export get_global_ce_transform,
    global_ce_transform, IdentityTransformer, ExplicitCETransformer
export Benchmark, benchmark, evaluate, default_measures
export validity, redundancy
export plausibility
export plausibility_energy_differential,
    plausibility_cosine, plausibility_distance_from_target
export faithfulness
export plausibility_measures, default_measures, distance_measures, all_measures
export concatenate_benchmarks
export compute_divergence

"Available plausibility measures."
const plausibility_measures = [
    plausibility_energy_differential, plausibility_cosine, plausibility_distance_from_target
]

"The default evaluation measures."
const default_measures = [
    validity, CounterfactualExplanations.Objectives.distance, redundancy
]

"All distance measures."
const distance_measures = [
    CounterfactualExplanations.Objectives.distance_l0,
    CounterfactualExplanations.Objectives.distance_l1,
    CounterfactualExplanations.Objectives.distance_l2,
    CounterfactualExplanations.Objectives.distance_linf,
]

"All measures."
const all_measures = [
    validity,
    redundancy,
    collect(values(CounterfactualExplanations.Objectives.penalties_catalogue))...,
    plausibility_measures...,
    faithfulness,
]

end
