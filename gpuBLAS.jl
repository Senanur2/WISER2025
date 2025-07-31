using CUDA
using BenchmarkTools
using Statistics

if length(ARGS) < 2
    println("Usage: julia gpuBLAS.jl <matrix_size> <tile_size>")
    exit(1)
end

n = parse(Int, ARGS[1])
tile_size = parse(Int, ARGS[2])

A = CUDA.randn(Float32, n, n)
B = CUDA.randn(Float32, n, n)
C_tile = CUDA.zeros(Float32, n, n)
C_gpu = CUDA.zeros(Float32, n, n)

function tile_gpu_multiply!(C, A, B, tile_size)
    fill!(C, 0.0f0)  # Optional if you set β = 0.0 below
    n = size(A, 1)
    for ii in 1:tile_size:n, jj in 1:tile_size:n, kk in 1:tile_size:n
        i_max = min(ii + tile_size - 1, n)
        j_max = min(jj + tile_size - 1, n)
        k_max = min(kk + tile_size - 1, n)

        A_sub = @view A[ii:i_max, kk:k_max]
        B_sub = @view B[kk:k_max, jj:j_max]
        C_sub = @view C[ii:i_max, jj:j_max]

        mul!(C_sub, A_sub, B_sub, 1.0f0, 1.0f0)
    end
end

# Benchmark
t_tile = @benchmark CUDA.@sync tile_gpu_multiply!($C_tile, $A, $B, $tile_size) samples=5
t_gpu = @benchmark CUDA.@sync mul!($C_gpu, $A, $B) samples=5

# Metrics
avg_time_tile = mean(t_tile).time / 1e6
avg_time_gpu = mean(t_gpu).time / 1e6
flops = 2 * n^3
gflops_tile = flops / (avg_time_tile * 1e6)
gflops_gpu = flops / (avg_time_gpu * 1e6)

# Output
println("GPU Matrix Multiplication Comparison")
println("Matrix size: $n x $n")
println("Tile size: $tile_size\n")

println("Tile GPU:")
println("  Time: $(round(avg_time_tile, digits=2)) ms")
println("  Performance: $(round(gflops_tile, digits=2)) GFLOP/s\n")

println("GPU Tile Built-in: ")
println("  Time: $(round(avg_time_gpu, digits=2)) ms")
println("  Performance: $(round(gflops_gpu, digits=2)) GFLOP/s\n")

println("--- Result Accuracy Check (isapprox) ---")
println("Manual Tile ≈ TileBuilt-in?     ", isapprox(C_tile, C_gpu; rtol=1e-4, atol=1e-6))
