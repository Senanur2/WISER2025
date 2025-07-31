using CUDA
using BenchmarkTools

# Custom matrix multiplication kernel (element-wise)
function matmul_kernel(A, B, C, M, N, K)
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if row <= M && col <= K
        sum = 0.0f0
        for n = 1:N
            sum += A[row, n] * B[n, col]
        end
        C[row, col] = sum
    end
    return
end

# Matrix dimensions
M, N, K = 512, 512, 512

# Allocate matrices on GPU
A = CUDA.rand(Float32, M, N)
B = CUDA.rand(Float32, N, K)
C_custom = CUDA.zeros(Float32, M, K)
C_builtin = similar(C_custom)

# Threads and blocks
threads = (16, 16)
blocks = (cld(K, threads[1]), cld(M, threads[2]))

# Warm up GPU
@cuda threads=threads blocks=blocks matmul_kernel(A, B, C_custom, M, N, K)
CUDA.@sync


# Benchmarking Custom GPU Kernel 
function benchmark_custom()
    CUDA.@sync
    t = @elapsed begin
        @cuda threads=threads blocks=blocks matmul_kernel(A, B, C_custom, M, N, K)
        CUDA.@sync
    end
    gflops = 2.0 * M * N * K / t / 1e9
    println("--- Custom GPU Kernel ---")
    println("Time: $(round(t * 1000, digits=2)) ms")
    println("Performance: $(round(gflops, digits=2)) GFLOP/s")
end

#Benchmarking for Built-in
function benchmark_builtin()
    CUDA.@sync
    t = @elapsed begin
        mul!(C_builtin, A, B)  # C_builtin = A * B
        CUDA.@sync
    end
    gflops = 2.0 * M * N * K / t / 1e9
    println("--- Built-in mul! ---")
    println("Time: $(round(t * 1000, digits=2)) ms")
    println("Performance: $(round(gflops, digits=2)) GFLOP/s")
end


# Accuracy Check
function check_accuracy()
    max_error = maximum(abs.(Array(C_custom) .- Array(C_builtin)))
    println("--- Result Accuracy Check ---")
    println("Max absolute error: $max_error")
end

# Run benchmarks
benchmark_custom()
benchmark_builtin()
check_accuracy()
