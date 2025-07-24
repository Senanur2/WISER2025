using BenchmarkTools
using Base.Threads
using LinearAlgebra
using Statistics

if length(ARGS) < 2
    println("Usage: julia vector_matrix_mul.jl <m> <n>")
    exit(1)
end

m = parse(Int, ARGS[1])
n = parse(Int, ARGS[2])

A = [i + j for i in 1:m, j in 1:n]
x = collect(1.0:n)

# -------------------------------
# Manual Multithreaded Version
# -------------------------------
function threaded_vector_matrix_multiply(A, x)
    m, n = size(A)
    if length(x) != n
        error("Matrix columns and vector size must match")
    end

    result = zeros(Float64, m)

    @threads for i in 1:m
        sum = 0.0
        for j in 1:n
            sum += A[i, j] * x[j]
        end
        result[i] = sum
    end

    return result
end

# -------------------------------
# Benchmark Manual Threaded
# -------------------------------
t_manual = @benchmark threaded_vector_matrix_multiply($A, $x) samples=10
avg_manual = mean(t_manual).time / 1e9
flops = 2 * m * n
gflops_manual = flops / (avg_manual * 1e9)
manual_result = threaded_vector_matrix_multiply(A, x)

println("\n🧠 Threaded Manual Vector-Matrix Multiply")
println("Threads used: ", nthreads())
println("Matrix size: $m × $n")
println("Time taken: ", round(avg_manual * 1000, digits=3), " ms")
println("Performance: ", round(gflops_manual, digits=3), " GFLOP/s")

# -------------------------------
# Benchmark Built-in A * x
# -------------------------------
t_builtin = @benchmark $A * $x samples=10
avg_builtin = mean(t_builtin).time / 1e9
gflops_builtin = flops / (avg_builtin * 1e9)
builtin_result = A * x

println("\n⚙️  Built-in A * x")
println("Time taken: ", round(avg_builtin * 1000, digits=3), " ms")
println("Performance: ", round(gflops_builtin, digits=3), " GFLOP/s")

# -------------------------------
# Correctness Check
# -------------------------------
println("\n✅ Results match? ", isapprox(manual_result, builtin_result))