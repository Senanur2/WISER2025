using CUDA

function vector_add_kernel(a, b, c)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(c)
        c[i] = a[i] + b[i]
    end
    return
end
