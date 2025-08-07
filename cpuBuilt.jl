# builtin_only.jl
using BenchmarkTools
using Statistics
using LinearAlgebra.BLAS

if length(ARGS) < 2
    println("Usage: julia builtin_only.jl <matrix_size> <thread_count>")
    exit(1)
end

n = parse(Int, ARGS[1])
thread_count = parse(Int, ARGS[2])

# Set BLAS threads (some BLAS backends may affect A*B performance)
BLAS.set_num_threads(thread_count)

# Matrix initialization
A = randn(n, n)
B = randn(n, n)
C_builtin = zeros(n, n)

function gflops(n, time_s)
    flops = 2 * n^3
    return flops / (time_s * 1e9)
end

# Benchmark built-in A * B
r_builtin = @benchmark $C_builtin = $A * $B samples=5 evals=1
builtin_time = minimum(r_builtin).time / 1e9
builtin_gflops = gflops(n, builtin_time)

# Output
println("Built-in A * B:")
println("  Matrix size: $n x $n")
println("  Threads used (BLAS): $thread_count")
println("  Time: $(round(builtin_time * 1000, digits=2)) ms")
println("  Performance: $(round(builtin_gflops, digits=2)) GFLOP/s")
