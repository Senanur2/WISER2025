using BenchmarkTools
using Statistics
using CUDA

if length(ARGS) < 2
    println("Usage: julia gpu_tile_benchmark.jl <matrix_size> <tile_size> [gpu_index]")
    exit(1)
end

n = parse(Int, ARGS[1])
tile_size = parse(Int, ARGS[2])
gpu_index = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0

CUDA.device!(gpu_index)
println("Using GPU: ", CUDA.device())

# Matrix initialization
A = CUDA.randn(Float32, n, n)
B = CUDA.randn(Float32, n, n)

C_blas = CUDA.zeros(Float32, n, n)
dC_blas = similar(C_blas)

function gflops(n, time_s)
    flops = 2 * n^3
    return flops / (time_s * 1e9)
end

fill!(dC_blas, 0.0f0)
    
r_blas = @benchmark CUDA.CUBLAS.gemm!('N', 'N', 1.0f0, $dA, $dB, 0.0f0, $dC_blas) samples=5 evals=1
r_blas = @benchmark begin
    CUDA.CUBLAS.gemm!('N', 'N', 1.0f0, $A, $B, 0.0f0, $dC_blas)
    synchronize()
end samples=5 evals=1

blas_time = minimum(r_blas).time / 1e9
blas_gflops = gflops(n, blas_time)


println("  Matrix Multiplication Comparison  ")
println("Matrix size: $n x $n")
println("Tile size: $tile_size")

println("BLAS gemm! (on GPU):")
println("  Time: $(round(blas_time * 1000, digits=2)) ms")
println("  Performance: $(round(blas_gflops, digits=2)) GFLOP/s\n")
