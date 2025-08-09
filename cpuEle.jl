using BenchmarkTools
using Base.Threads

if length(ARGS) < 2
    println("Usage: julia benchmark.jl <matrix_size> <thread_count>")
    exit(1)
end

n = parse(Int, ARGS[1])
thread_count = parse(Int, ARGS[2])

# Make sure to run Julia with correct threads:
# JULIA_NUM_THREADS=thread_count julia benchmark.jl n thread_count

A = randn(n, n)
B = randn(n, n)
C_builtin = zeros(n, n)
C_elementwise = zeros(n, n)

# Element-wise for-loop multiply with threading
function elementwise_forloop!(C, A, B)
    @inbounds @threads for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            C[i, j] = A[i, j] * B[i, j]
        end
    end
end

# GFLOPS for matrix multiply (2 * n^3 FLOPs)
function gflops_matrix(n, time_s)
    flops = 2 * n^3
    return flops / (time_s * 1e9)
end

# GFLOPS for element-wise multiply (n^2 FLOPs)
function gflops_elementwise(n, time_s)
    flops = n^2
    return flops / (time_s * 1e9)
end

# Benchmarks

r_builtin = @benchmark $C_builtin = $A * $B samples=5 evals=1
builtin_time = minimum(r_builtin).time / 1e9
builtin_gflops = gflops_matrix(n, builtin_time)

r_elementwise_loop = @benchmark elementwise_forloop!($C_elementwise, $A, $B) samples=5 evals=1
elementwise_time = minimum(r_elementwise_loop).time / 1e9
elementwise_gflops = gflops_elementwise(n, elementwise_time)

# Accuracy check
diff_elementwise = maximum(abs.(C_elementwise .- (A .* B)))

# Output
println("Matrix size: $n x $n")
println("Threads used: $thread_count\n")

println("Built-in A * B:")
println("  Time: $(round(builtin_time * 1000, digits=2)) ms")
println("  Performance: $(round(builtin_gflops, digits=2)) GFLOP/s\n")

println("Element-wise for-loop multiply:")
println("  Time: $(round(elementwise_time * 1000, digits=2)) ms")
println("  Performance: $(round(elementwise_gflops, digits=2)) GFLOP/s\n")

println("Accuracy Check:")
println("  Max difference (Element-wise vs .*): $diff_elementwise")
