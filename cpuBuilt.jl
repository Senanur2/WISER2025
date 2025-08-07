# builtin_only.jl
using BenchmarkTools
using Statistics
using LinearAlgebra.BLAS

if length(ARGS) < 3
    println("Usage: julia builtin_only.jl <matrix_size> <tile_size> <thread_count>")
    exit(1)
end

n = parse(Int, ARGS[1])
tile_size = parse(Int, ARGS[2])  # not used, just accepted
thread_count = parse(Int, ARGS[3])

BLAS.set_num_threads(thread_count)

A = randn(n, n)
B = randn(n, n)
C_builtin = zeros(n, n)

function gflops(n, time_s)
    flops = 2 * n^3
    return flops / (time_s * 1e9)
end

r_builtin = @benchmark $C_builtin = $A * $B samples=5 evals=1
builtin_time = minimum(r_builtin).time / 1e9
builtin_gflops = gflops(n, builtin_time)

println("Built-in A * B:")
println("  Matrix size: $n x $n")
println("  Tile size (unused): $tile_size")
println("  Threads used (BLAS): $thread_count")
println("  Time: $(round(builtin_time * 1000, digits=2)) ms")
println("  Performance: $(round(builtin_gflops, digits=2)) GFLOP/s")
