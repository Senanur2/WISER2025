using BenchmarkTools
using Base.Threads
using Statistics

if length(ARGS) < 2
    println("Usage: julia benchmark.jl <matrix_size> <thread_count>")
    exit(1)
end

n = parse(Int, ARGS[1])
thread_count = parse(Int, ARGS[2])

# Make sure to start Julia with JULIA_NUM_THREADS=thread_count
println("Using $thread_count threads (JULIA_NUM_THREADS environment variable)")

A = randn(n, n)
B = randn(n, n)
C_tile = zeros(n, n)

# You can keep tile_size fixed or pick a default inside the code
const tile_size = 256  # or any fixed number you want

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

println("  Matrix Multiplication (Tile-wise)  ")
println("Matrix size: $n x $n")
println("Tile size: $tile_size")
println("Threads used (from JULIA_NUM_THREADS): $thread_count\n")

println("Threaded Tile Multiply:")
println("  Time: $(round(tile_time * 1000, digits=2)) ms")
println("  Performance: $(round(tile_gflops, digits=2)) GFLOP/s")
