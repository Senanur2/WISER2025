using BenchmarkTools
using LinearAlgebra
using Statistics
using Base.Threads

if length(ARGS) < 1
    println("Usage: julia dotProduct.jl <array_size>")
    exit(1)
end

n = parse(Int, ARGS[1])
u = collect(1:n)
v = collect(1:n)
flops = 2 * n

function manual_dot_product(u, v)
    partial_sums = zeros(Float64, nthreads())
    @threads for i in 1:length(u)
        partial_sums[threadid()] += u[i] * v[i]
    end
    return sum(partial_sums)
end

# Benchmark manual implementation
t_manual = @benchmark manual_dot_product($u, $v) samples=10
avg_manual = mean(t_manual).time / 1e9  # Convert ns to s
gflops_manual = flops / (avg_manual * 1e9)

println("\nManual Implementation:")
println("Time taken: ", round(avg_manual, digits=6), " s")
println("Performance: ", round(gflops_manual, digits=3), " GFLOP/s")

# Benchmark built-in element-wise implementation
t_builtin = @benchmark sum($u .* $v) samples=10
avg_builtin = mean(t_builtin).time / 1e9
gflops_builtin = flops / (avg_builtin * 1e9)

println("\nBuilt-in Implementation:")
println("Time taken: ", round(avg_builtin, digits=6), " s")
println("Performance: ", round(gflops_builtin, digits=3), " GFLOP/s")

# Benchmark BLAS implementation
t_blas = @benchmark dot($u, $v) samples=10
avg_blas = mean(t_blas).time / 1e9
gflops_blas = flops / (avg_blas * 1e9)

println("\nBLAS Implementation:")
println("Time taken: ", round(avg_blas, digits=6), " s")
println("Performance: ", round(gflops_blas, digits=3), " GFLOP/s")
