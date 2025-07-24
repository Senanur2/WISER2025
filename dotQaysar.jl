using BenchmarkTools
using Base.Threads
using LinearAlgebra
using LinearAlgebra.BLAS
using Statistics

if length(ARGS) < 1
    println("Usage: julia dotProduct.jl <array_size>")
    exit(1)
end

n = parse(Int, ARGS[1])
u = collect(1:n)
v = collect(1:n)

# -------------------------------
# Threaded Dot Product
# -------------------------------
function threaded_dot_product(u, v)
    partial_sums = zeros(Float64, nthreads())

    @threads for i in 1:length(u)
        partial_sums[threadid()] += u[i] * v[i]
    end

    return sum(partial_sums)
end

# -------------------------------
# Benchmark Threaded Dot Product
# -------------------------------
t_threaded = @benchmark threaded_dot_product($u, $v) samples=10
avg_threaded = mean(t_threaded).time / 1e9
gflops_threaded = (2 * n) / (avg_threaded * 1e9)

println("\n🧠 Threaded Dot Product")
println("Threads used: ", nthreads())
println("Time taken: ", round(avg_threaded, digits=6), " s")
println("Performance: ", round(gflops_threaded, digits=3), " GFLOP/s")
println("Result: ", threaded_dot_product(u, v))

# -------------------------------
# Benchmark Built-in `sum(u .* v)`
# -------------------------------
t_builtin = @benchmark sum($u .* $v) samples=10
avg_builtin = mean(t_builtin).time / 1e9
gflops_builtin = (2 * n) / (avg_builtin * 1e9)

println("\n⚙️  Built-in `sum(u .* v)`")
println("Time taken: ", round(avg_builtin, digits=6), " s")
println("Performance: ", round(gflops_builtin, digits=3), " GFLOP/s")
println("Result: ", sum(u .* v))

# -------------------------------
# Benchmark BLAS `dot(u, v)`
# -------------------------------
t_blas = @benchmark dot($u, $v) samples=10
avg_blas = mean(t_blas).time / 1e9
gflops_blas = (2 * n) / (avg_blas * 1e9)

println("\n🚀 BLAS `dot(u, v)`")
println("Time taken: ", round(avg_blas, digits=6), " s")
println("Performance: ", round(gflops_blas, digits=3), " GFLOP/s")
println("Result: ", dot(u, v))