using BenchmarkTools
using LinearAlgebra
using Statistics
using Base.Threads

if length(ARGS) < 2
    println("Usage: julia vector_matrix_mul.jl <m> <n>")
    exit(1)
end

m = parse(Int, ARGS[1])
n = parse(Int, ARGS[2])

A = [i + j for i in 1:m, j in 1:n]
x = collect(1.0:n)
flops = 2 * m * n

function manual_vector_matrix_multiply(A, x)
    m, n = size(A)
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

# Run once to get the results for comparison
result_manual = manual_vector_matrix_multiply(A, x)
result_builtin = A * x
result_blas = mul!(zeros(m), A, x)

# Benchmark manual implementation
t_manual = @benchmark manual_vector_matrix_multiply($A, $x) samples=10
avg_manual = mean(t_manual).time / 1e9
gflops_manual = flops / (avg_manual * 1e9)

println("\nManual Implementation:")
println("Time taken: ", round(avg_manual, digits=6), " s")
println("Performance: ", round(gflops_manual, digits=3), " GFLOP/s")

# Benchmark built-in implementation
t_builtin = @benchmark $A * $x samples=10
avg_builtin = mean(t_builtin).time / 1e9
gflops_builtin = flops / (avg_builtin * 1e9)

println("\nBuilt-in Implementation:")
println("Time taken: ", round(avg_builtin, digits=6), " s")
println("Performance: ", round(gflops_builtin, digits=3), " GFLOP/s")

# Benchmark BLAS implementation
t_blas = @benchmark mul!($(zeros(m)), $A, $x) samples=10
avg_blas = mean(t_blas).time / 1e9
gflops_blas = flops / (avg_blas * 1e9)

println("\nBLAS Implementation:")
println("Time taken: ", round(avg_blas, digits=6), " s")
println("Performance: ", round(gflops_blas, digits=3), " GFLOP/s")

# Check for approximate equality
equal = isapprox(result_manual, result_builtin; atol=1e-8)
println("\nManual ≈ Built-in? ", equal ? "true" : "false")
