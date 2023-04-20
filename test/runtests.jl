using SparsePCA
using Test

using RDatasets
using LinearAlgebra

iris = dataset("datasets", "iris")
describe(iris)
X = Array(iris[:, 1:4])
y = Vector(iris[:, 5])

@testset "SparsePCA.jl" begin
    w, z = SparsePCA.SPCA(X, 2, [0.1, 0.26])
    @test z == [0.7628055537099959 0.0; 0.011448860867247397 0.0; 0.016680144792570066 -0.5890917106637228; 0.14750795241577716 -0.22120294750550107]
end
