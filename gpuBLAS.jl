using CUDA

function vector_add_kernel(a, b, c)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(c)
        c[i] = a[i] + b[i]
    end
    return
end

N = 1000
a = CUDA.fill(1.0f0, N)
b = CUDA.fill(2.0f0, N)
c = CUDA.zeros(Float32, N)

threads_per_block = 256
blocks = cld(N, threads_per_block)

@cuda threads=threads_per_block blocks=blocks vector_add_kernel(a, b, c)

result = Array(c)
println(result[1:10])
