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
A = rand(Float32, n, n)
B = rand(Float32, n, n)

dA = CuArray(A)
dB = CuArray(B)
dC_tile = CuArray(zeros(Float32, n, n))
dC_blas = CuArray(zeros(Float32, n, n))


# Tile-based GPU Kernel

function gpu_tile_kernel(C, A, B, N, tile_size)
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if row <= N && col <= N
        acc = 0.0f0
        for kk in 1:tile_size:N
            k_max = min(kk + tile_size - 1, N)
            for k in kk:k_max
                acc += A[row, k] * B[k, col]
            end
        end
        C[row, col] = acc
    end

    return nothing 
end

# GFLOPS calculation

function gflops(n, time_s)
    flops = 2 * n^3
    return flops / (time_s * 1e9)
end


# Run tile-based GPU kernel

threads = (16, 16)
blocks = (cld(n, threads[1]), cld(n, threads[2]))

# Warmup
@cuda threads=threads blocks=blocks gpu_tile_kernel(dC_tile, dA, dB, n, tile_size)
synchronize()

# Benchmark
r_tile = @benchmark begin
    @cuda threads=$threads blocks=$blocks gpu_tile_kernel($dC_tile, $dA, $dB, $n, $tile_size)
    synchronize()
end samples=5 evals=1

tile_time = minimum(r_tile).time / 1e9
tile_gflops = gflops(n, tile_time)


# Run cuBLAS gemm

r_blas = @benchmark begin
    CUDA.CUBLAS.gemm!('N', 'N', Float32(1.0), dA, dB, Float32(0.0), dC)
    synchronize()
end samples=5 evals=1

blas_time = minimum(r_blas).time / 1e9
blas_gflops = gflops(n, blas_time)

# Accuracy

diff = maximum(abs.(Array(dC_tile) .- Array(dC_blas)))

# Output
println("  GPU Matrix Multiplication Benchmark  ")
println("Matrix size: $n x $n")
println("Tile size: $tile_size\n")

println("Tile-Based GPU Multiply:")
println("  Time: $(round(tile_time * 1000, digits=2)) ms")
println("  Performance: $(round(tile_gflops, digits=2)) GFLOP/s\n")

println("CUDA CUBLAS.gemm!:")
println("  Time: $(round(blas_time * 1000, digits=2)) ms")
println("  Performance: $(round(blas_gflops, digits=2)) GFLOP/s\n")

println("Accuracy Check:")
println("  Max absolute difference: $diff")
