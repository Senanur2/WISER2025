using BenchmarkTools
using Base.Threads
using LinearAlgebra
using Statistics

if length(ARGS) < 1
    println("Usage: julia dotProduct.jl <array_size>")
    exit(1)
end

n = parse(Int, ARGS[1])
u = collect(1:n)
v = collect(1:n)
flops = 2 * n

function threaded_dot_product(u, v)
    partial_sums = zeros(Float64, nthreads())
    @threads for i in 1:length(u)
        partial_sums[threadid()] += u[i] * v[i]
    end
    return sum(partial_sums)
end

function print_stats(name, t, flops)
    avg = mean(t).time / 1e9
    gflops = flops / (avg * 1e9)
    println("$name:\tTime = $(round(avg * 1000, digits=3)) ms\tPerformance = $(round(gflops, digits=3)) GFLOP/s")
end

t_manual = @benchmark threaded_dot_product($u, $v) samples=10
t_builtin = @benchmark sum($u .* $v) samples=10
t_blas = @benchmark dot($u, $v) samples=10

println("\nDot Product (size: $n)")
print_stats("Manual", t_manual, flops)
print_stats("Built-in", t_builtin, flops)
print_stats("BLAS", t_blas, flops)
