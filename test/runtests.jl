using julia_optym
import julia_optym: steepestDescent
using Test

"Testing set of fuctions of two variables with single global minimum (f => [x, y, minimum])"
double = Dict(
    ((x, y) -> (x-2)^2 + (2y-3)^2) => [2, 1.5, 0],
    ((x, y) -> (x + 2y - 7)^2 + (2x + y - 5)^2) => [1, 3, 0],
    ((x, y) -> 0.26(x^2 + y^2) - 0.48x*y) => [0, 0, 0]
)
"Testing set of fuctions of three variables with single global minimum (f => [x, y, z, minimum])"
tripple = Dict(
    ((x, y, z) -> 2(x-1)^2 + y^2 + z^2) => [1, 0, 0, 0],
    ((x, y, z) -> 1/9 * x^4 + 1/16 * y^2 + z^2 - 1) => [0, 0, 0, -1]
)

@testset "Deepest descent testing" begin
    @testset "General test for functions of two variables with single global minimum" begin
        for (f, result) in double
            starting_values = [result[1] - 2, result[2] + 2]
            minimum = result[3]
            @testset "Epsilon tests" begin
                for tolerance in [1e-2, 1e-4, 1e-6, 1e-8]
                    @test isapprox(steepestDescent(f, starting_values, tolerance)[2], minimum, atol=tolerance)
                end
            end
            @testset "Test functions" begin
                @test isapprox(steepestDescent(f, starting_values, 1e-6)[2], minimum, atol=1e-6)
            end
        end
    end
    @testset "General test for functions of three variables with single global minimum" begin
        for (f, result) in tripple
            starting_values = [result[1] - 2, result[2] + 2, result[3] - 5]
            minimum = result[4]
            @testset "Epsilon tests" begin
                for tolerance in [1e-2, 1e-4, 1e-6, 1e-8]
                    @test isapprox(steepestDescent(f, starting_values, tolerance)[2], minimum, atol=tolerance)
                end
            end
            @testset "Test functions" begin
                @test isapprox(steepestDescent(f, starting_values, 1e-6)[2], minimum, atol=1e-6)
            end
        end
    end
end
