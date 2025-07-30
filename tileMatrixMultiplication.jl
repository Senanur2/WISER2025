using LinearAlgebra
using BenchmarkTools
using Base.Threads
using Statistics

if length(ARGS) < 2
    println("Usage: julia tile_mult.jl <matrix_size> <tile_size>")
    exit(1)
end

n = parse(Int, ARGS[1])
tile_size = parse(Int, ARGS[2])

A = randn(n, n)
B = randn(n, n)
C_manual = zeros(n, n)
C_builtin = zeros(n, n)
C_blas = zeros(n, n)

# Manual threaded tile multiplication
function threaded_tile_multiply!(C, A, B, tile_size)
  fill!(C, 0.0)
    n = size(A, 1)
    Threads.@threads for ii in 1:tile_size:n
        for jj in 1:tile_size:n
            for kk in 1:tile_size:n
                i_max = min(ii+tile_size-1, n)
                j_max = min(jj+tile_size-1, n)
                k_max = min(kk+tile_size-1, n)

                for i in ii:i_max
                    for j in jj:j_max
                        for k in kk:k_max
                            C[i, j] += A[i, k] * B[k, j]
                        end
                    end
                end
            end
        end
    end
end

# Benchmark Manual Threaded
LinearAlgebra.BLAS.set_num_threads(1)
t_manual = @benchmark threaded_tile_multiply!($C_manual, $A, $B, $tile_size) samples=10
avg_time_manual = mean(t_manual).time / 1e9

# Benchmark Built-in (A * B)
t_builtin = @benchmark $C_builtin .= $A * $B samples=10
avg_time_builtin = mean(t_builtin).time / 1e9

# Benchmark BLAS (mul!)
t_blas = @benchmark mul!($C_blas, $A, $B) samples=10
avg_time_blas = mean(t_blas).time / 1e9

# Performance: FLOPs = 2n³
flops = 2 * n^3
gflops_manual = flops / (avg_time_manual * 1e9)
gflops_builtin = flops / (avg_time_builtin * 1e9)
gflops_blas = flops / (avg_time_blas * 1e9)

# Results
println("\n--- Matrix Multiplication Comparison ---")
println("Matrix size: $n x $n")
println("Tile size: $tile_size")
println("Threads used: $(nthreads())\n")

println("Manual Threaded:")
println("  Time: $(round(avg_time_manual * 1000, digits=2)) ms")
println("  Performance: $(round(gflops_manual, digits=2)) GFLOP/s")

println("Built-in A * B:")
println("  Time: $(round(avg_time_builtin * 1000, digits=2)) ms")
println("  Performance: $(round(gflops_builtin, digits=2)) GFLOP/s")

println("BLAS mul!(C, A, B):")
println("  Time: $(round(avg_time_blas * 1000, digits=2)) ms")
println("  Performance: $(round(gflops_blas, digits=2)) GFLOP/s")

# Result comparison using isapprox
println("\n--- Result Accuracy Check (isapprox) ---")
println("Manual ≈ Built-in? ", isapprox(C_manual, C_builtin; rtol=1e-5, atol=1e-8))
println("Manual ≈ BLAS?     ", isapprox(C_manual, C_blas; rtol=1e-5, atol=1e-8))
println("Built-in ≈ BLAS?   ", isapprox(C_builtin, C_blas; rtol=1e-5, atol=1e-8))
