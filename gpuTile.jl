using CUDA
using Statistics
using BenchmarkTools
using LinearAlgebra

# Command-line Arguments
if length(ARGS) < 2
    println("Usage: julia gpu_tile_benchmark.jl <matrix_size> <tile_size> [gpu_index]")
    exit(1)
end

n = parse(Int, ARGS[1])
tile_size = parse(Int, ARGS[2])
gpu_index = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0

# Select GPU
CUDA.device!(gpu_index)
println("Using GPU: ", CUDA.device())

# Data Setup
setprecision(113)  # 34-bit ~ 9 decimal digits
A = rand(BigFloat, n, n)
B = rand(BigFloat, n, n)

# Note: CuArray does not support BigFloat. Simulating precision only for tile kernel on CPU.
# For GPU compatibility, you’d need custom Float34 emulation — not practical on CUDA.

# Tile-based Kernel (Runs on CPU for BigFloat simulation)
function cpu_tile_kernel(C, A, B, N, tile_size)
    for row in 1:N, col in 1:N
        acc = BigFloat(0.0)
        for kk in 1:tile_size:N
            k_max = min(kk + tile_size - 1, N)
            for k in kk:k_max
                acc += A[row, k] * B[k, col]
            end
        end
        C[row, col] = acc
    end
end

function gflops(n, time_s)
    flops = 2 * n^3
    return flops / (time_s * 1e9)
end

C_tile = zeros(BigFloat, n, n)
r_tile = @benchmark begin
    cpu_tile_kernel($C_tile, $A, $B, $n, $tile_size)
end samples=1 evals=1

tile_time = minimum(r_tile).time / 1e9
tile_gflops = gflops(n, tile_time)

# Now use cuBLAS gemm with Float32 as usual
A32 = Float32.(A)
B32 = Float32.(B)

dA = CuArray(A32)
dB = CuArray(B32)
dC_blas = CuArray(zeros(Float32, n, n))

r_blas = @benchmark begin
    CUDA.CUBLAS.gemm!('N', 'N', Float32(1.0), dA, dB, Float32(0.0), dC_blas)
    synchronize()
end samples=5 evals=1

blas_time = minimum(r_blas).time / 1e9
blas_gflops = gflops(n, blas_time)

diff = maximum(abs.(Float32.(C_tile) .- Array(dC_blas)))

println("  GPU Matrix Multiplication Benchmark  ")
println("Matrix size: $n x $n")
println("Tile size: $tile_size\n")

println("Tile-Based CPU Multiply (Simulated Float34):")
println("  Time: $(round(tile_time * 1000, digits=2)) ms")
println("  Performance: $(round(tile_gflops, digits=2)) GFLOP/s\n")

println("CUDA CUBLAS.gemm! (Float32):")
println("  Time: $(round(blas_time * 1000, digits=2)) ms")
println("  Performance: $(round(blas_gflops, digits=2)) GFLOP/s\n")

println("Accuracy Check:")
println("  Max absolute difference: $diff")
