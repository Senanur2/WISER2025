using BenchmarkTools
using Statistics
using LinearAlgebra
using Base.Threads

if length(ARGS) != 3
    println("Usage: julia matrixBench.jl <rowsA> <sharedDim> <colsB>")
    exit(1)
end

rowsA = parse(Int, ARGS[1])
sharedDim = parse(Int, ARGS[2])
colsB = parse(Int, ARGS[3])

function matrix_multiply(A, B)
    m, nA = size(A)
    nB, p = size(B)

    if nA != nB
        println("Error: Number of columns in A must equal number of rows in B.")
        return nothing
    end

    C = zeros(m, p)

    @threads for i in 1:m
        for j in 1:p
            sum = 0.0
            for k in 1:nA
                sum += A[i, k] * B[k, j]
            end
            C[i, j] = sum
        end
    end

    return C
end

# Generate random matrices
A = rand(Float64, rowsA, sharedDim)
B = rand(Float64, sharedDim, colsB)

# Total FLOPs = 2 * m * n * p
flops = 2 * rowsA * sharedDim * colsB

# Benchmark custom threaded implementation
t_custom = @benchmark matrix_multiply($A, $B) samples=10
avg_custom = mean(t_custom).time / 1e9  # Convert ns to s
gflops_custom = flops / (avg_custom * 1e9)

println("\nManual (Threaded) Implementation:")
println("Time taken: ", round(avg_custom, digits=6), " s")
println("Performance: ", round(gflops_custom, digits=3), " GFLOP/s")

# Benchmark built-in implementation
t_builtin = @benchmark $A * $B samples=10
avg_builtin = mean(t_builtin).time / 1e9  # Convert ns to s
gflops_builtin = flops / (avg_builtin * 1e9)

println("\nBuilt-in Implementation:")
println("Time taken: ", round(avg_builtin, digits=6), " s")
println("Performance: ", round(gflops_builtin, digits=3), " GFLOP/s")

# Benchmark BLAS implementation
t_blas = @benchmark BLAS.gemm!('N', 'N', 1.0, $A, $B, 0.0, zeros(rowsA, colsB)) samples=10
avg_blas = mean(t_blas).time / 1e9  # Convert ns to s
gflops_blas = flops / (avg_blas * 1e9)

println("\nBLAS Implementation:")
println("Time taken: ", round(avg_blas, digits=6), " s")
println("Performance: ", round(gflops_blas, digits=3), " GFLOP/s")
