# blas_vs_builtin.jl
using BenchmarkTools
using LinearAlgebra.BLAS
using Statistics

if length(ARGS) < 2
    println("Usage: julia blas_vs_builtin.jl <matrix_size> <thread_count> [tile_size]")
    exit(1)
end

n = parse(Int, ARGS[1])
thread_count = parse(Int, ARGS[2])
tile_size = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0  # optional tile size

# Set BLAS threads
BLAS.set_num_threads(thread_count)

# Matrix initialization
A = randn(n, n)
B = randn(n, n)
C_blas = zeros(n, n)
C_builtin = zeros(n, n)

function gflops(n, time_s)
    flops = 2 * n^3
    return flops / (time_s * 1e9)
end

# ───── Benchmarks ─────
r_blas = @benchmark gemm!('N', 'N', 1.0, A, B, 0.0, C_blas) samples=5 evals=1
blas_time = minimum(r_blas).time / 1e9
blas_gflops = gflops(n, blas_time)

r_builtin = @benchmark $C_builtin = $A * $B samples=5 evals=1
builtin_time = minimum(r_builtin).time / 1e9
builtin_gflops = gflops(n, builtin_time)

# ───── Accuracy Check ─────
diff_builtin_blas = maximum(abs.(C_builtin .- C_blas))

# ───── Output ─────
println("Matrix size: $n x $n")
println("Threads used: $thread_count")
if tile_size > 0
    println("Tile size (info only): $tile_size")
end
println()

println("BLAS gemm!(C, A, B):")
println("  Time: $(round(blas_time * 1000, digits=2)) ms")
println("  Performance: $(round(blas_gflops, digits=2)) GFLOP/s\n")

println("Built-in A * B:")
println("  Time: $(round(builtin_time * 1000, digits=2)) ms")
println("  Performance: $(round(builtin_gflops, digits=2)) GFLOP/s\n")

println("Accuracy Check:")
println("  Max difference (Built-in vs BLAS): $diff_builtin_blas")
