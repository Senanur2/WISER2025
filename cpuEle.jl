using BenchmarkTools
using Base.Threads
using LinearAlgebra.BLAS

if length(ARGS) < 2
    println("Usage: julia benchmark.jl <matrix_size> <thread_count>")
    exit(1)
end

n = parse(Int, ARGS[1])
thread_count = parse(Int, ARGS[2])

println("Requested threads (from command line): $thread_count")

# Make sure Julia uses exactly thread_count threads:
# This is a bit tricky because JULIA_NUM_THREADS must be set before Julia launch.
# But we can *inform* user to launch Julia with that env var.
println("Note: To ensure Julia uses this thread count, launch with JULIA_NUM_THREADS=$thread_count")

# Set BLAS threads accordingly
BLAS.set_num_threads(thread_count)
println("BLAS threads set to: ", BLAS.get_num_threads())

# Check Julia threads (runtime Threads.nthreads() is fixed at launch)
println("Julia threads detected at runtime: ", Threads.nthreads())
if Threads.nthreads() != thread_count
    println("Warning: Julia runtime threads != requested threads! JULIA_NUM_THREADS should match command line argument.")
end

# Initialize matrices
A = randn(Float64, n, n)
B = randn(Float64, n, n)

# Manual threaded element-wise multiply function
function manual_elementwise_multiply(A, B)
    C = zeros(Float64, size(A))
    Threads.@threads for i in 1:size(A,1)
        for j in 1:size(A,2)
            C[i,j] = A[i,j] * B[i,j]
        end
    end
    return C
end

# GFLOPS calculation (only multiplications count)
function gflops_elementwise(n, time_s)
    flops = n^2
    return flops / (time_s * 1e9)
end

# Benchmark manual threaded multiply
r_manual = @benchmark manual_elementwise_multiply($A, $B) samples=5 evals=1
manual_time = minimum(r_manual).time / 1e9
manual_gflops = gflops_elementwise(n, manual_time)
C_manual = manual_elementwise_multiply(A, B)

# Benchmark built-in element-wise multiply
r_builtin = @benchmark $A .* $B samples=5 evals=1
builtin_time = minimum(r_builtin).time / 1e9
builtin_gflops = gflops_elementwise(n, builtin_time)
C_builtin = A .* B

# Accuracy check
max_diff = maximum(abs.(C_manual .- C_builtin))

println("\nElement-wise Multiplication Benchmark")
println("Matrix size: $n x $n")
println("Threads requested: $thread_count")
println("Julia threads detected at runtime: ", Threads.nthreads())
println("BLAS threads: ", BLAS.get_num_threads(), "\n")

println("Manual Threaded Element-wise Multiply:")
println("  Time: $(round(manual_time*1000, digits=2)) ms")
println("  Performance: $(round(manual_gflops, digits=2)) GFLOP/s\n")

println("Built-in Element-wise Multiply:")
println("  Time: $(round(builtin_time*1000, digits=2)) ms")
println("  Performance: $(round(builtin_gflops, digits=2)) GFLOP/s\n")

println("Accuracy Check:")
println("  Max difference (manual vs built-in): $max_diff")
