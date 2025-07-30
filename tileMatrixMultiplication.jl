using LinearAlgebra
using BenchmarkTools
using Base.Threads
using Statistics

if length(ARGS) < 3
    println("Usage: julia benchmark.jl <matrix_size> <tile_size> <thread_count>")
    exit(1)
end

n = parse(Int, ARGS[1])
tile_size = parse(Int, ARGS[2])
thread_count = parse(Int, ARGS[3])

# Set BLAS threads
BLAS.set_num_threads(thread_count)

# Warning if JULIA_NUM_THREADS mismatch
if nthreads() != thread_count
    println("Warning: Julia threads $(nthreads()) != requested $thread_count. Set JULIA_NUM_THREADS=$thread_count.")
end

# Generate matrices
A = randn(n, n)
B = randn(n, n)
C_tile = zeros(n, n)
C_blas = zeros(n, n)

function threaded_tile_blas_multiply!(C, A, B, tile_size)
    fill!(C, 0.0)
    n = size(A, 1)
    Threads.@threads for ii in 1:tile_size:n
        for jj in 1:tile_size:n
            for kk in 1:tile_size:n
                i_max = min(ii + tile_size - 1, n)
                j_max = min(jj + tile_size - 1, n)
                k_max = min(kk + tile_size - 1, n)

                A_sub = view(A, ii:i_max, kk:k_max)
                B_sub = view(B, kk:k_max, jj:j_max)
                C_sub = view(C, ii:i_max, jj:j_max)

                BLAS.gemm!('N', 'N', 1.0, A_sub, B_sub, 1.0, C_sub)
            end
        end
    end
end

# Benchmarking
t_tile = @benchmark threaded_tile_blas_multiply!($C_tile, $A, $B, $tile_size) samples=5
avg_time_tile = mean(t_tile).time / 1e6   

t_blas = @benchmark mul!($C_blas, $A, $B) samples=5
avg_time_blas = mean(t_blas).time / 1e6    

flops = 2 * n^3
gflops_tile = flops / (avg_time_tile * 1e6)
gflops_blas = flops / (avg_time_blas * 1e6)

#results
println("-->Matrix Multiplication Comparison ")
println("Matrix size: $n x $n")
println("Tile size: $tile_size")
println("Threads used: $thread_count\n")

println("Tile BLAS:")
println("  Time: $(round(avg_time_tile, digits=2)) ms")
println("  Performance: $(round(gflops_tile, digits=2)) GFLOP/s\n")

println("Matrix BLAS: ")
println("  Time: $(round(avg_time_blas, digits=2)) ms")
println("  Performance: $(round(gflops_blas, digits=2)) GFLOP/s\n")

println("--- Result Accuracy Check (isapprox) ---")
println("Manual ≈ BLAS?     ", isapprox(C_tile, C_blas; rtol=1e-5, atol=1e-8))
