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
flops = 2 * m * n

function threaded_vector_matrix_multiply(A, x)
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

function print_stats(name, t, flops)
    avg = mean(t).time / 1e9
    gflops = flops / (avg * 1e9)
    println("$name:\tTime = $(round(avg * 1000, digits=3)) ms\tPerformance = $(round(gflops, digits=3)) GFLOP/s")
end

t_manual = @benchmark threaded_vector_matrix_multiply($A, $x) samples=10
t_builtin = @benchmark $A * $x samples=10
t_blas = @benchmark mul!($(zeros(m)), $A, $x) samples=10  # BLAS under the hood

println("\nVector-Matrix Multiplication ($m x $n)")
print_stats("Manual", t_manual, flops)
print_stats("Built-in", t_builtin, flops)
print_stats("BLAS", t_blas, flops)
