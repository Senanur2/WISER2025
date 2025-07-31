using CUDA
using BenchmarkTools

# Naive element-wise CUDA kernel for matrix multiplication
function matmul_naive_kernel(C, A, B, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= N && j <= N
        sum = zero(eltype(C))
        for k in 1:N
            sum += A[i,k] * B[k,j]
        end
        C[i,j] = sum
    end
    return
end

function gpu_matmul_naive(A_d, B_d, N)
    C_d = CUDA.zeros(Float32, N, N)
    threads = (16, 16)
    blocks = (cld(N, threads[1]), cld(N, threads[2]))
    @cuda threads=threads blocks=blocks matmul_naive_kernel(C_d, A_d, B_d, N)
    return C_d
end

function main()
    N = 1024  # matrix size NxN (adjust as needed)

    # Generate random Float32 matrices on CPU
    A_h = rand(Float32, N, N)
    B_h = rand(Float32, N, N)

    # Transfer to GPU
    A_d = CuArray(A_h)
    B_d = CuArray(B_h)

    # Warmup
    C_built_in = A_d * B_d
    C_naive = gpu_matmul_naive(A_d, B_d, N)
    synchronize()

    # Benchmark built-in
    built_in_time = @belapsed begin
        C_built_in = $A_d * $B_d
        synchronize()
    end

    # Benchmark naive kernel
    naive_time = @belapsed begin
        C_naive = gpu_matmul_naive($A_d, $B_d, $N)
        synchronize()
    end

    # Calculate GFLOPS
    flops = 2 * N^3
    built_in_gflops = flops / (built_in_time * 1e9)
    naive_gflops = flops / (naive_time * 1e9)

    println("Matrix size: $N x $N")
    println("Built-in cuBLAS multiply:")
    println("  Time: $(round(built_in_time*1000, digits=3)) ms")
    println("  Performance: $(round(built_in_gflops, digits=2)) GFLOP/s")
    println("  element-wise kernel multiply:")
    println("  Time: $(round(naive_time*1000, digits=3)) ms")
    println("  Performance: $(round(naive_gflops, digits=2)) GFLOP/s")


    max_error = maximum(abs.(Array(C_built_in) .- Array(C_naive)))
    println("Max absolute error between results: $max_error")
    println("Using GPU: ", CUDA.device())
end

main()
