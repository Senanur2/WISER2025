using BenchmarkTools
using Statistics
using Base.Threads
using LinearAlgebra
using LinearAlgebra.BLAS

if length(ARGS) != 3
    println("Usage: julia matrixBench.jl <rowsA> <sharedDim> <colsB>")
    exit(1)
end

rowsA = parse(Int, ARGS[1])
sharedDim = parse(Int, ARGS[2])
colsB = parse(Int, ARGS[3])

A = rand(Float64, rowsA, sharedDim)
B = rand(Float64, sharedDim, colsB)
flops = 2 * rowsA * sharedDim * colsB

function threaded_matrix_multiply(A, B)
    m, nA = size(A)
    nB, p = size(B)
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

function blas_multiply(A, B)
    m, n = size(A)
    n2, p = size(B)
    C = zeros(m, p)
    gemm!('N', 'N', 1.0, A, B, 0.0, C)
    return C
end

function print_stats(name, t, flops)
    avg = mean(t).time / 1e9
    gflops = flops / (avg * 1e9)
    println("$name:\tTime = $(round(avg * 1000, digits=3)) ms\tPerformance = $(round(gflops, digits=3)) GFLOP/s")
end

t_manual = @benchmark threaded_matrix_multiply($A, $B) samples=10
t_builtin = @benchmark $A * $B samples=10
t_blas = @benchmark blas_multiply($A, $B) samples=10

println("\nMatrix-Matrix Multiplication ($rowsA x $sharedDim) × ($sharedDim x $colsB)")
print_stats("Manual", t_manual, flops)
print_stats("Built-in", t_builtin, flops)
print_stats("BLAS", t_blas, flops)
