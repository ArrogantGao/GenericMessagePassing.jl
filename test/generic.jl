using GenericMessagePassing

using Test
using Random
Random.seed!(1234)

using GenericMessagePassing: uniform, normalize!, damp!, abs_error

@testset "unified operations, Float types" begin
    for T in [Float32, Float64]
        for s in [2, 3, 4]
            @test uniform(T, s) == ones(T, s) ./ prod(s)
            t = rand(T, s)
            normalize!(t)
            @test sum(t) ≈ T(1.0)
            t1 = rand(T, s)
            t1_old = copy(t1)
            t2 = rand(T, s)
            damp!(t1, t2, T(0.2))
            @test t1 == t1_old .* T(0.2) .+ t2 .* T(0.8)
            @test abs_error(t1, t2) > 0.0
        end

        for s in [[2, 3], [2, 3, 4]]
            @test uniform(T, s) == ones(T, s...) ./ prod(s)
            t = rand(T, s...)
            normalize!(t)
            @test sum(t) ≈ T(1.0)
            t1 = rand(T, s...)
            t1_old = copy(t1)
            t2 = rand(T, s...)
            damp!(t1, t2, T(0.2))
            @test t1 == t1_old .* T(0.2) .+ t2 .* T(0.8)
            @test abs_error(t1, t2) > 0.0
        end
    end
end
