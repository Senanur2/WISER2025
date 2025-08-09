using BenchmarkTools
using Base.Threads
using LinearAlgebra.BLAS

if length(ARGS) < 1
    println("Usage: julia benchmark.jl <matrix_size>")
    exit(1)
end

n = parse(Int, ARGS[1])
thread_count = Threads.nthreads()
println("Threads detected from JULIA_NUM_THREADS: $thread_count")

# Set BLAS threads to match Julia threads to avoid oversubscription
BLAS.set_num_threads(thread_count)
println("BLAS threads set to: ", BLAS.get_num_threads())

# Initialize matrices as Float64 explicitly
A = randn(Float64, n, n)
B = randn(Float64, n, n)
C_elem = zeros(Float64, n, n)
C_builtin = zeros(Float64, n, n)

# Threaded element-wise multiply
function threaded_elementwise_multiply!(C, A, B)
    Threads.@threads for i in 1:size(A,1)
        for j in 1:size(A,2)
            C[i,j] = A[i,j] * B[i,j]
        end
    end
end

# GFLOPS calculation for element-wise multiply
function gflops_elementwise(n, time_s)
    flops = n^2  # only multiplications
    return flops / (time_s * 1e9)
end

# Ensure C_elem is zeroed before each manual run to avoid stale data
function run_manual_elementwise!(C, A, B)
    fill!(C, 0.0)
    threaded_elementwise_multiply!(C, A, B)
end

# Run manual threaded multiply once to validate accuracy without benchmark noise
run_manual_elementwise!(C_elem, A, B)
max_diff_pre = maximum(abs.(C_elem .- (A .* B)))
println("Initial accuracy check (before benchmarking): max diff = $max_diff_pre")

# Benchmark manual threaded element-wise multiply
r_elem = @benchmark run_manual_elementwise!($C_elem, $A, $B) samples=5 evals=1
elem_time = minimum(r_elem).time / 1e9
elem_gflops = gflops_elementwise(n, elem_time)

# Benchmark built-in element-wise multiply
r_builtin_elem = @benchmark $C_builtin = $A .* $B samples=5 evals=1
builtin_elem_time = minimum(r_builtin_elem).time / 1e9
builtin_elem_gflops = gflops_elementwise(n, builtin_elem_time)

# Accuracy check after benchmarking manual threaded multiply
max_diff_post = maximum(abs.(C_elem .- C_builtin))

println("\nElement-wise Multiplication Benchmark")
println("Matrix size: $n x $n")
println("Threads used: $thread_count\n")

println("Manual Threaded Element-wise Multiply:")
println("  Time: $(round(elem_time*1000, digits=2)) ms")
println("  Performance: $(round(elem_gflops, digits=2)) GFLOP/s\n")

println("Built-in Element-wise Multiply:")
println("  Time: $(round(builtin_elem_time*1000, digits=2)) ms")
println("  Performance: $(round(builtin_elem_gflops, digits=2)) GFLOP/s\n")

println("Accuracy Checks:")
println("  Initial max difference (manual vs built-in): $max_diff_pre")
println("  Post-benchmark max difference (manual vs built-in): $max_diff_post")
