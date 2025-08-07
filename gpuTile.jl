using BenchmarkTools
using Statistics
using CUDA

if length(ARGS) < 2
    println("Usage: julia gpu_tile_benchmark.jl <matrix_size> <tile_size> [gpu_index]")
    exit(1)
end

n = parse(Int, ARGS[1])
tile_size = parse(Int, ARGS[2])
gpu_index = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0

CUDA.device!(gpu_index)
println("Using GPU: ", CUDA.device())

# Matrix initialization
A = randn(Float32, n, n)
B = randn(Float32, n, n)

C_tile = CUDA.zeros(Float32, n, n)
C_builtin = CUDA.zeros(Float32, n, n)
C_blas = CUDA.zeros(Float32, n, n)

dA = CuArray(A)
dB = CuArray(B)
dC_blas = similar(C_blas)

# Tiled multiply using CUBLAS
function tile_multiply!(C, A, B, tile_size)
    fill!(C, 0.0f0)
    n = size(A, 1)

    for ii in 1:tile_size:n
        for jj in 1:tile_size:n
            for kk in 1:tile_size:n
                i_max = min(ii + tile_size - 1, n)
                j_max = min(jj + tile_size - 1, n)
                k_max = min(kk + tile_size - 1, n)

                A_tile = @view A[ii:i_max, kk:k_max]
                B_tile = @view B[kk:k_max, jj:j_max]
                C_tile_view = @view C[ii:i_max, jj:j_max]

                CUDA.CUBLAS.gemm!(
                    'N', 'N',
                    1.0f0,
                    A_tile, B_tile,
                    1.0f0,
                    C_tile_view
                )
            end
        end
    end
end

function gflops(n, time_s)
    flops = 2 * n^3
    return flops / (time_s * 1e9)
end

# Threads used (for reference)
thread_count = Threads.nthreads()

# ───── Benchmarks ─────

# 1. Tile
r_tile = @benchmark tile_multiply!($C_tile, $dA, $dB, $tile_size) samples=5 evals=1
tile_time = minimum(r_tile).time / 1e9
tile_gflops = gflops(n, tile_time)

# 2. BLAS
r_blas = @benchmark CUDA.CUBLAS.gemm!('N', 'N', 1.0f0, $dA, $dB, 0.0f0, $dC_blas) samples=5 evals=1
blas_time = minimum(r_blas).time / 1e9
blas_gflops = gflops(n, blas_time)

# 3. Built-in
r_builtin = @benchmark $C_builtin .= $dA * $dB samples=5 evals=1
builtin_time = minimum(r_builtin).time / 1e9
builtin_gflops = gflops(n, builtin_time)

# ───── Accuracy Checks ─────
diff_tile_blas = maximum(abs.(Array(C_tile) .- Array(dC_blas)))
diff_builtin_blas = maximum(abs.(Array(C_builtin) .- Array(dC_blas)))

# ───── Output ─────
println("  Matrix Multiplication Comparison  ")
println("Matrix size: $n x $n")
println("Tile size: $tile_size")
println("Threads used: $thread_count\n")

println("Tiled Multiply (on GPU):")
println("  Time: $(round(tile_time * 1000, digits=2)) ms")
println("  Performance: $(round(tile_gflops, digits=2)) GFLOP/s\n")

println("BLAS gemm! (on GPU):")
println("  Time: $(round(blas_time * 1000, digits=2)) ms")
println("  Performance: $(round(blas_gflops, digits=2)) GFLOP/s\n")

println("Built-in A * B (on GPU):")
println("  Time: $(round(builtin_time * 1000, digits=2)) ms")
println("  Performance: $(round(builtin_gflops, digits=2)) GFLOP/s\n")

println("Accuracy Check (vs BLAS):")
println("  Max difference (Tile vs BLAS): $diff_tile_blas")
println("  Max difference (Built-in vs BLAS): $diff_builtin_blas")
