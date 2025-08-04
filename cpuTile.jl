

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

tile_time = @elapsed threaded_tile_multiply!(C_tile, A, B, tile_size)
tile_gflops = gflops(n, tile_time)

blas_time = @elapsed mul!(C_blas, A, B)
blas_gflops = gflops(n, blas_time)

max_diff = maximum(abs.(C_tile .- C_blas))



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

