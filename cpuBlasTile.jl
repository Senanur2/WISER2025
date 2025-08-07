# blas_only.jl
using BenchmarkTools
using LinearAlgebra.BLAS
using Statistics

if length(ARGS) < 2
    println("Usage: julia blas_only.jl <matrix_size> <thread_count>")
    exit(1)
end

n = parse(Int, ARGS[1])
thread_count = parse(Int, ARGS[2])

# Set BLAS thread count
BLAS.set_num_threads(thread_count)

# Matrix initialization
A = randn(n, n)
B = randn(n, n)
C_blas = zeros(n, n)

function gflops(n, time_s)
    flops = 2 * n^3
    return flops / (time_s * 1e9)
end

# Benchmark BLAS gemm!
r_blas = @benchmark gemm!('N', 'N', 1.0, A, B, 0.0, C_blas) samples=5 evals=1
blas_time = minimum(r_blas).time / 1e9
blas_gflops = gflops(n, blas_time)

# Output
println("BLAS gemm!(C, A, B):")
println("  Matrix size: $n x $n")
println("  Threads used: $thread_count")
println("  Time: $(round(blas_time * 1000, digits=2)) ms")
println("  Performance: $(round(blas_gflops, digits=2)) GFLOP/s")
