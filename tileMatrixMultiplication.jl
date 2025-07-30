using LinearAlgebra
using BenchmarkTools
using Base.Threads
using Statistics
#get size from command line
if length(ARGS) < 2
    println("Usage: julia threaded_tile_blas_benchmark.jl <matrix_size> <tile_size>")
    exit(1)
end

n = parse(Int, ARGS[1])         # Matrix size
tile_size = parse(Int, ARGS[2]) # Tile size

# Matrix setup 
A = randn(n, n)
B = randn(n, n)
C = zeros(n, n)

# Tile-wise threaded BLAS multiply 
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


println("Matrix size: $n x $n | Tile size: $tile_size")

    # Benchmark
    t = @benchmark threaded_tile_blas_multiply!($C, $A, $B, $tile_size) samples=10
    avg_time = mean(t).time / 1e9   # seconds

    flops = 2 * n^3
    gflops = flops / (avg_time * 1e9)

    println("  Avg time: $(round(avg_time * 1000, digits=2)) ms")
    println("  GFLOP/s:  $(round(gflops, digits=2))")
end
