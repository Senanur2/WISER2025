using BenchmarkTools
using Base.Threads
using Statistics
using LinearAlgebra.BLAS

if length(ARGS) < 3
    println("Usage: julia benchmark.jl <matrix_size> <tile_size> <thread_count>")
    exit(1)
end

n = parse(Int, ARGS[1])
tile_size = parse(Int, ARGS[2])
thread_count = parse(Int, ARGS[3])

# Set BLAS threads
BLAS.set_num_threads(thread_count)

# Matrix initialization
A = randn(n, n)
B = randn(n, n)
C_tile = zeros(n, n)
C_blas = zeros(n, n)
C_builtin = zeros(n, n)

# Tiled multiply using BLAS.gemm!
function threaded_tile_multiply!(C, A, B, tile_size)
    fill!(C, 0.0)
    n = size(A, 1)

    Threads.@threads for ii in 1:tile_size:n
        for jj in 1:tile_size:n
            for kk in 1:tile_size:n
                i_max = min(ii + tile_size - 1, n)
                j_max = min(jj + tile_size - 1, n)
                k_max = min(kk + tile_size - 1, n)

                A_tile = @view A[ii:i_max, kk:k_max]
                B_tile = @view B[kk:k_max, jj:j_max]
                C_tile = @view C[ii:i_max, jj:j_max]

                BLAS.gemm!('N', 'N', 1.0, A_tile, B_tile, 1.0, C_tile)
            end
        end
    end
end

function threaded_elementwise_multiply!(C, A, B)
    n = size(A, 1)
    Threads.@threads for i in 1:n
        for j in 1:n
            C[i, j] = A[i, j] * B[i, j]
        end
    end
end

function gflops(n, time_s)
    flops = 2 * n^3
    return flops / (time_s * 1e9)
end

function gflops_elementwise(n, time_s)
    flops = n^2   # just multiplications, no adds
    return flops / (time_s * 1e9)
end

# ───── Benchmarks ─────

# 1. Tile
r_tile = @benchmark threaded_tile_multiply!($C_tile, $A, $B, $tile_size) samples=5 evals=1
tile_time = minimum(r_tile).time / 1e9
tile_gflops = gflops(n, tile_time)

# 2. BLAS
r_blas = @benchmark gemm!('N', 'N', 1.0, A, B, 0.0, C_blas) samples=5 evals=1
blas_time = minimum(r_blas).time / 1e9
blas_gflops = gflops(n, blas_time)

# 3. Built-in
r_builtin = @benchmark $C_builtin = $A * $B samples=5 evals=1
builtin_time = minimum(r_builtin).time / 1e9
builtin_gflops = gflops(n, builtin_time)

C_elem = zeros(n, n)
r_elem = @benchmark threaded_elementwise_multiply!($C_elem, $A, $B) samples=5 evals=1
elem_time = minimum(r_elem).time / 1e9
elem_gflops = gflops_elementwise(n, elem_time)

# ───── Accuracy Checks ─────
diff_tile_blas = maximum(abs.(C_tile .- C_blas))
diff_builtin_blas = maximum(abs.(C_builtin .- C_blas))
diff_elem = maximum(abs.(C_elem .- (A .* B)))


# ───── Output ─────
println("  Matrix Multiplication Comparison  ")
println("Matrix size: $n x $n")
println("Tile size: $tile_size")
println("Threads used: $thread_count\n")

println("Threaded Tile Multiply:")
println("  Time: $(round(tile_time * 1000, digits=2)) ms")
println("  Performance: $(round(tile_gflops, digits=2)) GFLOP/s\n")

println("Element-wise Multiply:")
println("  Time: $(round(elem_time * 1000, digits=2)) ms")
println("  Performance: $(round(elem_gflops, digits=2)) GFLOP/s\n")

println("BLAS gemm!(C, A, B):")
println("  Time: $(round(blas_time * 1000, digits=2)) ms")
println("  Performance: $(round(blas_gflops, digits=2)) GFLOP/s\n")

println("Built-in A * B:")
println("  Time: $(round(builtin_time * 1000, digits=2)) ms")
println("  Performance: $(round(builtin_gflops, digits=2)) GFLOP/s\n")

println("Accuracy Check (vs BLAS):")
println("  Max difference (Tile vs BLAS): $diff_tile_blas")
println("  Max difference (Built-in vs BLAS): $diff_builtin_blas")
println("  Max difference  (Element-wise vs Built-in .*): $diff_elem")
