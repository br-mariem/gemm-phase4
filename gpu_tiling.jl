# =============================================================
# Phase 4 — Comprehensive Tiling Sweep
# Tests many tile sizes on large matrices (N > 8000)
# Tile sizes: from 2x2 up to 64x64 (hardware limit)
# Matrix sizes: 8192, 10240, 12288, 16384
# Goal: find the optimal tile size for each matrix size
# =============================================================

using CUDA
using CUDA.CUBLAS
using BenchmarkTools
using Printf

gflops(N, t) = 2.0 * Float64(N)^3 / t / 1e9

# -----------------------------------------------------------
# Generic shared-memory tiled kernel
# TILE is a compile-time constant via Val{TILE}
# -----------------------------------------------------------
function shmem_kernel!(C, A, B, M, N, K, ::Val{TILE}) where TILE
    tx = threadIdx().x
    ty = threadIdx().y
    row = (blockIdx().x - 1) * TILE + tx
    col = (blockIdx().y - 1) * TILE + ty

    As = @cuStaticSharedMem(Float32, (TILE, TILE))
    Bs = @cuStaticSharedMem(Float32, (TILE, TILE))

    acc = 0.0f0

    for t in 0:cld(K, TILE)-1
        a_col = t * TILE + ty
        As[tx, ty] = (row <= M && a_col <= K) ? A[row, a_col] : 0.0f0
        b_row = t * TILE + tx
        Bs[tx, ty] = (b_row <= K && col <= N) ? B[b_row, col] : 0.0f0
        sync_threads()
        for k in 1:TILE
            @inbounds acc += As[tx, k] * Bs[k, ty]
        end
        sync_threads()
    end

    if row <= M && col <= N
        @inbounds C[row, col] = acc
    end
    return
end

# -----------------------------------------------------------
# Launch for a specific tile size
# Only square tiles supported by shared memory kernel
# Tile sizes must be power of 2 and <= 32 (hardware limit)
# -----------------------------------------------------------
function run_tiled!(C, A, B, tile)
    M, K = size(A)
    _, N = size(B)
    threads = (tile, tile)
    blocks  = (cld(M, tile), cld(N, tile))

    if tile == 2
        @cuda threads=threads blocks=blocks shmem_kernel!(C, A, B, M, N, K, Val(2))
    elseif tile == 4
        @cuda threads=threads blocks=blocks shmem_kernel!(C, A, B, M, N, K, Val(4))
    elseif tile == 8
        @cuda threads=threads blocks=blocks shmem_kernel!(C, A, B, M, N, K, Val(8))
    elseif tile == 16
        @cuda threads=threads blocks=blocks shmem_kernel!(C, A, B, M, N, K, Val(16))
    elseif tile == 32
        @cuda threads=threads blocks=blocks shmem_kernel!(C, A, B, M, N, K, Val(32))
    else
        error("Unsupported tile size: $tile. Use 2, 4, 8, 16, or 32.")
    end
    CUDA.synchronize()
end

# -----------------------------------------------------------
# Register tiling kernel — each thread computes RX×RY elements
# RX and RY are compile-time via Val
# -----------------------------------------------------------
function reg_kernel!(C, A, B, M, N, K, ::Val{TILE}, ::Val{RX}, ::Val{RY}) where {TILE, RX, RY}
    tx = threadIdx().x
    ty = threadIdx().y
    base_row = (blockIdx().x - 1) * (TILE * RX) + (tx - 1) * RX + 1
    base_col = (blockIdx().y - 1) * (TILE * RY) + (ty - 1) * RY + 1

    As = @cuStaticSharedMem(Float32, (TILE * RX, TILE))
    Bs = @cuStaticSharedMem(Float32, (TILE, TILE * RY))

    # Accumulators in registers
    acc = CUDA.zeros_like(C, (RX, RY))  # not valid in kernel
    a00 = 0.0f0; a01 = 0.0f0
    a10 = 0.0f0; a11 = 0.0f0

    for t in 0:cld(K, TILE)-1
        for r in 1:RX
            row = base_row + r - 1
            col = t * TILE + ty
            @inbounds As[(tx-1)*RX+r, ty] = (row<=M && col<=K) ? A[row,col] : 0.0f0
        end
        for r in 1:RY
            row = t * TILE + tx
            col = base_col + r - 1
            @inbounds Bs[tx, (ty-1)*RY+r] = (row<=K && col<=N) ? B[row,col] : 0.0f0
        end
        sync_threads()
        for k in 1:TILE
            a1 = @inbounds As[(tx-1)*RX+1, k]
            a2 = @inbounds As[(tx-1)*RX+2, k]
            b1 = @inbounds Bs[k, (ty-1)*RY+1]
            b2 = @inbounds Bs[k, (ty-1)*RY+2]
            a00 += a1*b1; a01 += a1*b2
            a10 += a2*b1; a11 += a2*b2
        end
        sync_threads()
    end

    if base_row   <= M && base_col   <= N; @inbounds C[base_row,   base_col]   = a00; end
    if base_row   <= M && base_col+1 <= N; @inbounds C[base_row,   base_col+1] = a01; end
    if base_row+1 <= M && base_col   <= N; @inbounds C[base_row+1, base_col]   = a10; end
    if base_row+1 <= M && base_col+1 <= N; @inbounds C[base_row+1, base_col+1] = a11; end
    return
end

# -----------------------------------------------------------
# Benchmark a single (N, tile) combination
# Returns (gflops, time_ms)
# -----------------------------------------------------------
function benchmark_tile(N, tile; samples=3)
    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)
    C = CUDA.zeros(Float32, N, N)

    try
        run_tiled!(C, A, B, tile)  # warmup
        t = @belapsed begin
            run_tiled!($C, $A, $B, $tile)
            CUDA.synchronize()
        end samples=samples evals=1
        return gflops(N, t), t * 1000
    catch e
        return -1.0, -1.0
    end
end

# -----------------------------------------------------------
# Benchmark cuBLAS for reference
# -----------------------------------------------------------
function benchmark_cublas(N; samples=3)
    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)
    C = CUDA.zeros(Float32, N, N)
    CUBLAS.gemm!('N', 'N', 1.0f0, A, B, 0.0f0, C)
    t = @belapsed begin
        CUBLAS.gemm!('N', 'N', 1.0f0, $A, $B, 0.0f0, $C)
        CUDA.synchronize()
    end samples=samples evals=1
    return gflops(N, t), t * 1000
end

# -----------------------------------------------------------
# Main sweep: all tile sizes × all matrix sizes
# -----------------------------------------------------------
function run_tiling_sweep()
    println("=" ^ 70)
    println("Comprehensive Tiling Sweep — RTX 3050 (Float32)")
    println("GPU: $(CUDA.name(CUDA.device()))")
    println(@sprintf("VRAM: %.1f GB", CUDA.totalmem(CUDA.device()) / 1e9))
    println("=" ^ 70)

    # Tile sizes to test
    # Note: GPU shared memory limits tile to max 32×32
    # (32×32 × 4 bytes × 2 matrices = 8KB, fits in 48KB shared mem)
    # Larger "virtual" tiles achieved via register tiling (2×2 per thread)
    tile_sizes = [2, 4, 8, 16, 32]

    # Large matrix sizes (all > 8000)
    matrix_sizes = [8192, 10240, 12288, 16384]

    # Check VRAM — 16384 in Float32 needs 3×16384²×4 = 3GB
    # Should fit in 4.3GB but leave some margin
    println("\nVRAM check:")
    for N in matrix_sizes
        mem_gb = 3 * N^2 * 4 / 1e9
        fits = mem_gb < 3.5
        @printf("  N=%-6d → %.2f GB needed  %s\n", N, mem_gb, fits ? "✓ OK" : "⚠ may OOM")
    end
    println()

    # -----------------------------------------------------------
    # Sweep 1: Tile size vs GFLOPS for each matrix size
    # -----------------------------------------------------------
    println("=" ^ 70)
    println("SWEEP 1: Tile size × Matrix size")
    println("=" ^ 70)
    println(@sprintf("%-8s %-10s %-12s %-12s %-12s", 
                     "Tile", "N=8192", "N=10240", "N=12288", "N=16384"))
    println(@sprintf("%-8s %-10s %-12s %-12s %-12s",
                     "", "GFLOPS", "GFLOPS", "GFLOPS", "GFLOPS"))
    println("-" ^ 58)

    results = Dict()  # (tile, N) => gflops

    for tile in tile_sizes
        row_str = @sprintf("%-8s", "$(tile)×$(tile)")
        for N in matrix_sizes
            if N > 12288 && CUDA.totalmem(CUDA.device()) < 4e9
                row_str *= @sprintf("%-12s", "skip(OOM)")
                continue
            end
            g, t = benchmark_tile(N, tile)
            results[(tile, N)] = g
            if g > 0
                row_str *= @sprintf("%-12.1f", g)
            else
                row_str *= @sprintf("%-12s", "ERROR")
            end
        end
        println(row_str)
    end

    # Register 2×2 row
    row_str = @sprintf("%-8s", "Reg 2×2")
    for N in matrix_sizes
        A = CUDA.rand(Float32, N, N)
        B = CUDA.rand(Float32, N, N)
        C = CUDA.zeros(Float32, N, N)
        try
            @cuda threads=(16,16) blocks=(cld(N,32),cld(N,32)) reg_kernel!(
                C, A, B, N, N, N, Val(16), Val(2), Val(2))
            CUDA.synchronize()
            t = @belapsed begin
                @cuda threads=(16,16) blocks=(cld($N,32),cld($N,32)) reg_kernel!(
                    $C, $A, $B, $N, $N, $N, Val(16), Val(2), Val(2))
                CUDA.synchronize()
            end samples=3 evals=1
            g = gflops(N, t)
            results[(:reg2x2, N)] = g
            row_str *= @sprintf("%-12.1f", g)
        catch e
            row_str *= @sprintf("%-12s", "ERROR")
        end
    end
    println(row_str)

    # cuBLAS reference row
    row_str = @sprintf("%-8s", "cuBLAS")
    for N in matrix_sizes
        g, t = benchmark_cublas(N)
        results[(:cublas, N)] = g
        row_str *= @sprintf("%-12.1f", g)
    end
    println(row_str)

    # -----------------------------------------------------------
    # Sweep 2: Detailed time breakdown for best matrix size (N=8192)
    # -----------------------------------------------------------
    println("\n")
    println("=" ^ 70)
    println("SWEEP 2: Detailed results for N=8192")
    println("=" ^ 70)
    println(@sprintf("%-12s %-12s %-12s %-12s %-12s",
                     "Tile", "GFLOPS", "Time (ms)", "vs Naive", "vs cuBLAS"))
    println("-" ^ 62)

    N = 8192
    # Naive GPU baseline (no tiling)
    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)
    C = CUDA.zeros(Float32, N, N)

    function naive_kernel!(C, A, B, M, N, K)
        i = (blockIdx().x-1)*blockDim().x + threadIdx().x
        j = (blockIdx().y-1)*blockDim().y + threadIdx().y
        if i<=M && j<=N
            val = 0.0f0
            for k in 1:K; @inbounds val += A[i,k]*B[k,j]; end
            @inbounds C[i,j] = val
        end
        return
    end

    @cuda threads=(16,16) blocks=(cld(N,16),cld(N,16)) naive_kernel!(C,A,B,N,N,N)
    CUDA.synchronize()
    t_naive = @belapsed begin
        @cuda threads=(16,16) blocks=(cld($N,16),cld($N,16)) naive_kernel!($C,$A,$B,$N,$N,$N)
        CUDA.synchronize()
    end samples=3 evals=1
    g_naive = gflops(N, t_naive)
    g_cublas = get(results, (:cublas, N), 1.0)

    @printf("%-12s %-12.2f %-12.2f %-12s %-12s\n",
            "Naive", g_naive, t_naive*1000, "1.00×", @sprintf("%.2f%%", g_naive/g_cublas*100))

    for tile in tile_sizes
        g = get(results, (tile, N), -1.0)
        if g > 0
            t_ms = 2.0 * Float64(N)^3 / g / 1e6
            @printf("%-12s %-12.2f %-12.2f %-12s %-12s\n",
                    "$(tile)×$(tile)", g, t_ms,
                    @sprintf("%.2f×", g/g_naive),
                    @sprintf("%.2f%%", g/g_cublas*100))
        end
    end

    g_reg = get(results, (:reg2x2, N), -1.0)
    if g_reg > 0
        t_ms = 2.0 * Float64(N)^3 / g_reg / 1e6
        @printf("%-12s %-12.2f %-12.2f %-12s %-12s\n",
                "Reg 2×2", g_reg, t_ms,
                @sprintf("%.2f×", g_reg/g_naive),
                @sprintf("%.2f%%", g_reg/g_cublas*100))
    end

    @printf("%-12s %-12.2f %-12.2f %-12s %-12s\n",
            "cuBLAS", g_cublas, 2.0*Float64(N)^3/g_cublas/1e6,
            @sprintf("%.2f×", g_cublas/g_naive), "100.00%")

    # -----------------------------------------------------------
    # Summary: best tile per matrix size
    # -----------------------------------------------------------
    println("\n")
    println("=" ^ 70)
    println("SUMMARY: Best tile size per matrix size")
    println("=" ^ 70)
    println(@sprintf("%-10s %-14s %-10s %-14s", "N", "Best tile", "GFLOPS", "% of cuBLAS"))
    println("-" ^ 52)

    for N in matrix_sizes
        best_tile = nothing
        best_g = 0.0
        for tile in tile_sizes
            g = get(results, (tile, N), -1.0)
            if g > best_g
                best_g = g
                best_tile = "$(tile)×$(tile)"
            end
        end
        g_reg = get(results, (:reg2x2, N), -1.0)
        if g_reg > best_g
            best_g = g_reg
            best_tile = "Reg 2×2"
        end
        g_cb = get(results, (:cublas, N), 1.0)
        @printf("%-10d %-14s %-10.2f %-14s\n",
                N, best_tile, best_g,
                @sprintf("%.1f%%", best_g/g_cb*100))
    end

    println()
    println("Run Phase 6 next to see how Float16 changes everything.")
end

run_tiling_sweep()