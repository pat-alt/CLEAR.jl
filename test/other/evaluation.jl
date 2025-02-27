using CounterfactualExplanations.Convergence
using CounterfactualExplanations.Evaluation:
    Benchmark,
    evaluate,
    validity,
    distance_measures,
    concatenate_benchmarks,
    compute_divergence
using CounterfactualExplanations.Objectives: distance
using Serialization: serialize
using TaijaData: load_moons, load_circles
using TaijaParallel: ThreadsParallelizer

# Dataset
data = TaijaData.load_overlapping()
counterfactual_data = CounterfactualExplanations.DataPreprocessing.CounterfactualData(
    data[1], data[2]
)

# Factual and target:
n_individuals = 5
M = fit_model(counterfactual_data, :Linear)
target = 2
factual = 1
ids = rand(findall(predict_label(M, counterfactual_data) .== factual), n_individuals)
xs = select_factual(counterfactual_data, ids)
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
x = select_factual(counterfactual_data, chosen)
generator = GenericGenerator()
ces = generate_counterfactual(
    xs, target, counterfactual_data, M, generator; num_counterfactuals=5
)
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
# Meta data:
meta_data = Dict(:generator => "Generic", :model => "MLP")
meta_data = [meta_data for i in 1:length(ces)]
# Pre-trained models:
models = Dict(
    :MLP => fit_model(counterfactual_data, :MLP),
    :Linear => fit_model(counterfactual_data, :Linear),
)
# Generators:
generators = Dict(
    :Generic => GenericGenerator(),
    :Gravitational => GravitationalGenerator(),
    :Wachter => WachterGenerator(),
    :ClaPROAR => ClaPROARGenerator(),
)

@testset "Evaluation" begin
    @test typeof(evaluate(ce; measure=validity)) <: Vector
    @test typeof(evaluate(ce; measure=distance)) <: Vector
    @test typeof(evaluate(ce; measure=distance_measures)) <: Vector
    @test typeof(evaluate(ce)) <: Vector
    @test typeof(evaluate.(ces)) <: Vector
    @test typeof(evaluate.(ces; report_each=true)) <: Vector
    @test typeof(evaluate.(ces; output_format=:Dict, report_each=true)) <: Vector{<:Dict}
    @test typeof(evaluate.(ces; output_format=:DataFrame, report_each=true)) <:
        Vector{<:DataFrame}

    # Faithfulness and plausibility:
    faith = Evaluation.faithfulness(ce)
    faith = Evaluation.faithfulness(ce; choose_lowest_energy=true)
    faith = Evaluation.faithfulness(ce; choose_random=true)
    faith = Evaluation.faithfulness(ce; cosine=true)
    delete!(ce.search, :energy_sampler)
    delete!(ce.M.fitresult.other, :energy_sampler)
    faith = Evaluation.faithfulness(ce; nwarmup=100)
    plaus = Evaluation.plausibility(ce)
    plaus = Evaluation.plausibility(ce; choose_random=true)
    plaus = Evaluation.plausibility_distance_from_target(ce)
    plaus = Evaluation.plausibility_cosine(ce)
    plaus = Evaluation.plausibility_energy_differential(ce)
    @test true

    @testset "Divergence Metrics" begin
        @test isnan(evaluate(ce; measure=MMD())[1][1])
    end
end

@testset "Benchmarking" begin
    bmk = Evaluation.benchmark(counterfactual_data; convergence=:generator_conditions)

    @testset "Parallelization" begin
        @testset "Threads" begin
            parallelizer = ThreadsParallelizer()
            bmk = benchmark(
                counterfactual_data;
                convergence=:generator_conditions,
                parallelizer=parallelizer,
            )
        end
    end

    @testset "Basics" begin
        @test typeof(bmk()) <: DataFrame
        @test typeof(bmk(; agg=nothing)) <: DataFrame
        @test typeof(vcat(bmk, bmk)) <: Benchmark
    end

    @testset "Different methods" begin
        @test typeof(benchmark(ces)) <: Benchmark
        @test typeof(benchmark(ces; meta_data=meta_data)) <: Benchmark
        @test typeof(
            benchmark(x, target, counterfactual_data; models=models, generators=generators)
        ) <: Benchmark
    end

    @testset "Full one" begin
        # Data:
        datasets = Dict(
            :moons => CounterfactualData(load_moons()...),
            :circles => CounterfactualData(load_circles()...),
        )

        # Models:
        models = Dict(:MLP => MLP, :Linear => Linear)

        # Generators:
        generators = Dict(:Generic => GenericGenerator(), :Greedy => GreedyGenerator())

        using CounterfactualExplanations.Evaluation: distance_measures
        bmks = []
        storage_dir = tempdir()
        for (i, (dataname, dataset)) in enumerate(datasets)
            bmk = benchmark(
                dataset; models=models, generators=generators, measure=distance_measures
            )
            serialize(joinpath(storage_dir, "run_1", "output_$(i).jls"), bmk)
            push!(bmks, bmk)
        end

        _bmks = concatenate_benchmarks(storage_dir)

        bmk = vcat(bmks[1], bmks[2]; ids=collect(keys(datasets)))
        @test typeof(bmk) <: Benchmark
    end

    @testset "Divergence" begin
        @test all(isnan.(benchmark(ces; measure=MMD())().value))
    end
end

@testset "Serialization" begin
    global_serializer(Serializer())
    @test _serialization_state == true
    global_serializer(NullSerializer())
    @test _serialization_state == false
    global_output_identifier(ExplicitOutputIdentifier("myid"))
    @test get_global_output_id() == "myid"
    global_output_identifier(DefaultOutputIdentifier())
    @test get_global_output_id() == ""
end

@testset "CE transform" begin
    @test_throws AssertionError ExplicitCETransformer(logits)
    transformer = ExplicitCETransformer(CounterfactualExplanations.counterfactual)
    global_ce_transform(transformer)
    @test get_global_ce_transform() == transformer.fun
    global_ce_transform(IdentityTransformer())
    x = 1
    @test get_global_ce_transform()(x) == x     # identity function
end

@testset "Divergence" begin
    n_individuals = 100
    ids = rand(findall(predict_label(M, counterfactual_data) .== factual), n_individuals)
    xs = select_factual(counterfactual_data, ids)
    conv = MaxIterConvergence(10)

    # Generic counterfactuals:
    ces = generate_counterfactual(
        xs,
        target,
        counterfactual_data,
        M,
        generator;
        initialization=:identity,
        convergence=conv,
    )

    @testset "MMD" begin
        using CounterfactualExplanations.Evaluation: kernelsum

        mmd = MMD()
        @test kernelsum(mmd.kernel, counterfactual_data.X[:, 1]) == 0.0
        @test mmd(counterfactual_data.X, counterfactual_data.X)[2] > 0.5

        mmd_generic = mmd(ces, counterfactual_data, n_individuals)

        bmk =
            benchmark(ces; measure=[validity, MMD()]) |>
            bmk -> compute_divergence(
                bmk, [validity, MMD(; compute_p=nothing)], counterfactual_data
            )

        @test all(.!isnan.(bmk.evaluation.value))
    end
end
