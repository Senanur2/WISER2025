using CUDA
using Statistics
using BenchmarkTools
using LinearAlgebra

if length(ARGS) < 2
    println("Usage: julia gpu_tile_benchmark.jl <matrix_size> <tile_size> [gpu_index]")
    exit(1)
end

n = parse(Int, ARGS[1])
tile_size = parse(Int, ARGS[2])
gpu_index = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0

CUDA.device!(gpu_index)
println("Using GPU: ", CUDA.device())

A = rand(Float64, n, n)
B = rand(Float64, n, n)
A = rand(Float32, n, n)
B = rand(Float32, n, n)

dA = CuArray(A)
dB = CuArray(B)
dC_tile = CuArray(zeros(Float64, n, n))
dC_blas = CuArray(zeros(Float64, n, n))
dC_tile = CuArray(zeros(Float32, n, n))
dC_blas = CuArray(zeros(Float32, n, n))

function gpu_tile_kernel(C, A, B, N, tile_size)
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
@@ -61,26 +61,26 @@
tile_gflops = gflops(n, tile_time)

r_blas = @benchmark begin
    CUDA.CUBLAS.gemm!('N', 'N', 1.0, dA, dB, 0.0, dC_blas)
    CUDA.CUBLAS.gemm!('N', 'N', Float32(1.0), dA, dB, Float32(0.0), dC_blas)
    synchronize()
end samples=5 evals=1

blas_time = minimum(r_blas).time / 1e9
blas_gflops = gflops(n, blas_time)

diff = maximum(abs.(Array(dC_tile) .- Array(dC_blas)))

println("  GPU Matrix Multiplication Benchmark  ")
println("Matrix size: $n x $n")
println("Tile size: $tile_size\n")

println("Tile-Based GPU Multiply:")
println("  Time: $(round(tile_time * 1000, digits=2)) ms")
println("  Performance: $(round(tile_gflops, digits=2)) GFLOP/s\n")

println("CUDA CUBLAS.gemm!:")
println("  Time: $(round(blas_time * 1000, digits=2)) ms")
println("  Performance: $(round(blas_gflops, digits=2)) GFLOP/s\n")

println("Accuracy Check:")
println("  Max absolute difference: $diff")
