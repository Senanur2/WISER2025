using BenchmarkTools
using Base.Threads
using Statistics
using LinearAlgebra.BLAS

if length(ARGS) < 3
    println("Usage: julia benchmark.jl <matrix_size> <tile_size> <thread_count>")
    exit(1)
end

n = parse(Int, ARGS[1])
tile_size = parse(Int, ARGS[2])
thread_count = parse(Int, ARGS[3])

# Set BLAS threads
BLAS.set_num_threads(thread_count)

A = randn(n, n)
B = randn(n, n)
C_tile = zeros(n, n)
C_blas = zeros(n, n)

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

function gflops(n, time_s)
    flops = 2 * n^3
    return flops / (time_s * 1e9)
end

# Benchmark tile
r_tile = @benchmark threaded_tile_multiply!($C_tile, $A, $B, $tile_size) samples=5 evals=1
tile_time = minimum(r_tile).time / 1e9  # ns to sec
tile_gflops = gflops(n, tile_time)

# Benchmark BLAS
r_blas = @benchmark gemm!('N', 'N', 1.0, A, B, 0.0, C_blas) samples=5 evals=1
blas_time = minimum(r_blas).time / 1e9
blas_gflops = gflops(n, blas_time)

# Accuracy check
max_diff = maximum(abs.(C_tile .- C_blas))

# Results
println("  Matrix Multiplication Comparison  ")
println("Matrix size: $n x $n")
println("Tile size: $tile_size")
println("Threads used: $thread_count\n")

println("Threaded Tile Multiply:")
println("  Time: $(round(tile_time * 1000, digits=2)) ms")
println("  Performance: $(round(tile_gflops, digits=2)) GFLOP/s\n")

println("BLAS mul!(C, A, B):")
  println("  Time: $(round(blas_time * 1000, digits=2)) ms")
println("  Performance: $(round(blas_gflops, digits=2)) GFLOP/s\n")

println("Accuracy Check")
println("Max absolute difference: $max_diff")
