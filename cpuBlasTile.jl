# blas_only.jl
using BenchmarkTools
using LinearAlgebra.BLAS
using Statistics

if length(ARGS) < 3
    println("Usage: julia blas_only.jl <matrix_size> <tile_size> <thread_count>")
    exit(1)
end

n = parse(Int, ARGS[1])
tile_size = parse(Int, ARGS[2])  # not used, but accepted
thread_count = parse(Int, ARGS[3])

BLAS.set_num_threads(thread_count)

A = randn(n, n)
B = randn(n, n)
C_blas = zeros(n, n)

function gflops(n, time_s)
    flops = 2 * n^3
    return flops / (time_s * 1e9)
end

r_blas = @benchmark gemm!('N', 'N', 1.0, A, B, 0.0, C_blas) samples=5 evals=1
blas_time = minimum(r_blas).time / 1e9
blas_gflops = gflops(n, blas_time)

println("BLAS gemm!(C, A, B):")
println("  Matrix size: $n x $n")
println("  Tile size (unused): $tile_size")
println("  Threads used: $thread_count")
println("  Time: $(round(blas_time * 1000, digits=2)) ms")
println("  Performance: $(round(blas_gflops, digits=2)) GFLOP/s")
