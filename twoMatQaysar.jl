using BenchmarkTools
using Statistics
using Base.Threads
using LinearAlgebra.BLAS

if length(ARGS) != 3
    println("Usage: julia matrixBench.jl <rowsA> <sharedDim> <colsB>")
    exit(1)
end

rowsA = parse(Int, ARGS[1])
sharedDim = parse(Int, ARGS[2])
colsB = parse(Int, ARGS[3])

# -------------------------------
# Manual Threaded Implementation
# -------------------------------
function threaded_matrix_multiply(A, B)
    m, nA = size(A)
    nB, p = size(B)

    if nA != nB
        println("Error: Incompatible matrix sizes.")
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

# -------------------------------
# BLAS (gemm!)
# -------------------------------
function blas_multiply(A, B)
    m, n = size(A)
    n2, p = size(B)
    C = zeros(m, p)
    alpha = 1.0
    beta = 0.0
    gemm!('N', 'N', alpha, A, B, beta, C)  # C = alpha*A*B + beta*C
    return C
end

# -------------------------------
# Generate Random Input
# -------------------------------
A = rand(Float64, rowsA, sharedDim)
B = rand(Float64, sharedDim, colsB)

# Total FLOPs
flops = 2 * rowsA * sharedDim * colsB

# -------------------------------
# Benchmark Manual Threaded
# -------------------------------
t_manual = @benchmark threaded_matrix_multiply($A, $B) samples=10
avg_manual = mean(t_manual).time / 1e9  # ns → s
gflops_manual = flops / (avg_manual * 1e9)

println("\n Manual Threaded Implementation")
println("Threads used: ", nthreads())
println("Time taken: ", round(avg_manual, digits=6), " s")
println("Performance: ", round(gflops_manual, digits=3), " GFLOP/s")

# -------------------------------
# Benchmark Built-in (*)
# -------------------------------
t_builtin = @benchmark $A * $B samples=10
avg_builtin = mean(t_builtin).time / 1e9
gflops_builtin = flops / (avg_builtin * 1e9)

println("\n Built-in Julia (A * B)")
println("Time taken: ", round(avg_builtin, digits=6), " s")
println("Performance: ", round(gflops_builtin, digits=3), " GFLOP/s")

# -------------------------------
# Benchmark BLAS gemm!
# -------------------------------
t_blas = @benchmark blas_multiply($A, $B) samples=10
avg_blas = mean(t_blas).time / 1e9
gflops_blas = flops / (avg_blas * 1e9)

println("\n BLAS gemm! (C = A * B)")
println("Time taken: ", round(avg_blas, digits=6), " s")
println("Performance: ", round(gflops_blas, digits=3), " GFLOP/s")