using BenchmarkTools
using Base.Threads

if length(ARGS) < 1
    println("Usage: julia benchmark.jl <matrix_size>")
    exit(1)
end

n = parse(Int, ARGS[1])

thread_count = Threads.nthreads()
println("Threads detected from JULIA_NUM_THREADS: $thread_count")

# Initialize matrices
A = randn(n, n)
B = randn(n, n)
C_elem = zeros(n, n)

# Threaded element-wise multiply
function threaded_elementwise_multiply!(C, A, B)
    Threads.@threads for i in 1:size(A,1)
        for j in 1:size(A,2)
            C[i,j] = A[i,j] * B[i,j]
        end
    end
end

# GFLOPS calculation for element-wise multiply
function gflops_elementwise(n, time_s)
    flops = n^2  # only multiplications
    return flops / (time_s * 1e9)
end

# Benchmark element-wise multiply
r_elem = @benchmark threaded_elementwise_multiply!($C_elem, $A, $B) samples=5 evals=1
elem_time = minimum(r_elem).time / 1e9
elem_gflops = gflops_elementwise(n, elem_time)

# Accuracy check vs built-in element-wise multiply
diff_elem = maximum(abs.(C_elem .- (A .* B)))

# Output
println("Element-wise Multiplication Benchmark")
println("Matrix size: $n x $n")
println("Threads used: $thread_count")
println("Time: $(round(elem_time*1000, digits=2)) ms")
println("Performance: $(round(elem_gflops, digits=2)) GFLOP/s")
println("Max difference vs built-in .*: $diff_elem")
