# =============================================================
# Tiling Sweep — Version rapide avec détails pour toutes les tailles
# Matrices: 8192, 10240, 12288, 16384
# Tiles: 8×8, 16×16, 32×32
# =============================================================

using CUDA
using CUDA.CUBLAS
using BenchmarkTools
using Printf

gflops(N, t) = 2.0 * Float64(N)^3 / t / 1e9

# Kernel tile 8
function kernel_8!(C, A, B, M, N, K)
    TILE = 8
    tx = threadIdx().x; ty = threadIdx().y
    row = (blockIdx().x-1)*TILE + tx
    col = (blockIdx().y-1)*TILE + ty
    As = @cuStaticSharedMem(Float32, (8,8))
    Bs = @cuStaticSharedMem(Float32, (8,8))
    acc = 0.0f0
    for t in 0:cld(K,TILE)-1
        As[tx,ty] = (row<=M && t*TILE+ty<=K) ? A[row,t*TILE+ty] : 0.0f0
        Bs[tx,ty] = (t*TILE+tx<=K && col<=N) ? B[t*TILE+tx,col] : 0.0f0
        sync_threads()
        for k in 1:TILE; @inbounds acc += As[tx,k]*Bs[k,ty]; end
        sync_threads()
    end
    if row<=M && col<=N; @inbounds C[row,col]=acc; end
    return
end

# Kernel tile 16
function kernel_16!(C, A, B, M, N, K)
    TILE = 16
    tx = threadIdx().x; ty = threadIdx().y
    row = (blockIdx().x-1)*TILE + tx
    col = (blockIdx().y-1)*TILE + ty
    As = @cuStaticSharedMem(Float32, (16,16))
    Bs = @cuStaticSharedMem(Float32, (16,16))
    acc = 0.0f0
    for t in 0:cld(K,TILE)-1
        As[tx,ty] = (row<=M && t*TILE+ty<=K) ? A[row,t*TILE+ty] : 0.0f0
        Bs[tx,ty] = (t*TILE+tx<=K && col<=N) ? B[t*TILE+tx,col] : 0.0f0
        sync_threads()
        for k in 1:TILE; @inbounds acc += As[tx,k]*Bs[k,ty]; end
        sync_threads()
    end
    if row<=M && col<=N; @inbounds C[row,col]=acc; end
    return
end

# Kernel tile 32
function kernel_32!(C, A, B, M, N, K)
    TILE = 32
    tx = threadIdx().x; ty = threadIdx().y
    row = (blockIdx().x-1)*TILE + tx
    col = (blockIdx().y-1)*TILE + ty
    As = @cuStaticSharedMem(Float32, (32,32))
    Bs = @cuStaticSharedMem(Float32, (32,32))
    acc = 0.0f0
    for t in 0:cld(K,TILE)-1
        As[tx,ty] = (row<=M && t*TILE+ty<=K) ? A[row,t*TILE+ty] : 0.0f0
        Bs[tx,ty] = (t*TILE+tx<=K && col<=N) ? B[t*TILE+tx,col] : 0.0f0
        sync_threads()
        for k in 1:TILE; @inbounds acc += As[tx,k]*Bs[k,ty]; end
        sync_threads()
    end
    if row<=M && col<=N; @inbounds C[row,col]=acc; end
    return
end

function run_kernel!(C, A, B, tile)
    M, K = size(A); _, N = size(B)
    threads = (tile, tile)
    blocks  = (cld(M,tile), cld(N,tile))
    if tile == 8
        @cuda threads=threads blocks=blocks kernel_8!(C,A,B,M,N,K)
    elseif tile == 16
        @cuda threads=threads blocks=blocks kernel_16!(C,A,B,M,N,K)
    elseif tile == 32
        @cuda threads=threads blocks=blocks kernel_32!(C,A,B,M,N,K)
    end
    CUDA.synchronize()
end

function run()
    println("=" ^ 65)
    println("Tiling Sweep — GPU RTX 3050 (Float32)")
    println("GPU: $(CUDA.name(CUDA.device()))")
    println("=" ^ 65)

    sizes = [8192, 10240, 12288, 16384]
    tiles = [8, 16, 32]

    # Compile tous les kernels d'abord sur petite matrice
    println("\nCompilation des kernels...")
    A_w = CUDA.rand(Float32, 512, 512)
    B_w = CUDA.rand(Float32, 512, 512)
    C_w = CUDA.zeros(Float32, 512, 512)
    for tile in tiles
        run_kernel!(C_w, A_w, B_w, tile)
        println("  Kernel $(tile)×$(tile) compilé ✓")
    end
    CUBLAS.gemm!('N','N',1.0f0,A_w,B_w,0.0f0,C_w)
    CUDA.synchronize()
    println("  cuBLAS compilé ✓")
    println()

    # -----------------------------------------------------------
    # Sweep 1 — Résumé: une ligne par matrice
    # -----------------------------------------------------------
    println("=" ^ 65)
    println("SWEEP 1: Résumé — GFLOPS par tile et par matrice")
    println("=" ^ 65)
    println(@sprintf("%-10s %-12s %-12s %-12s %-12s %-10s",
                     "N", "8×8", "16×16", "32×32", "cuBLAS", "Meilleur"))
    println(@sprintf("%-10s %-12s %-12s %-12s %-12s %-10s",
                     "", "GFLOPS", "GFLOPS", "GFLOPS", "GFLOPS", "tile"))
    println("-" ^ 70)

    all_results = Dict()

    for N in sizes
        A = CUDA.rand(Float32, N, N)
        B = CUDA.rand(Float32, N, N)
        C = CUDA.zeros(Float32, N, N)

        row_vals = Float64[]
        for tile in tiles
            run_kernel!(C, A, B, tile)
            t = @belapsed begin
                run_kernel!($C, $A, $B, $tile)
                CUDA.synchronize()
            end samples=2 evals=1
            g = gflops(N, t)
            push!(row_vals, g)
            all_results[(N, tile)] = (g, t*1000)
        end

        CUBLAS.gemm!('N','N',1.0f0,A,B,0.0f0,C); CUDA.synchronize()
        t_cb = @belapsed begin
            CUBLAS.gemm!('N','N',1.0f0,$A,$B,0.0f0,$C)
            CUDA.synchronize()
        end samples=2 evals=1
        g_cb = gflops(N, t_cb)
        all_results[(N, :cublas)] = (g_cb, t_cb*1000)

        best_idx = argmax(row_vals)
        best_name = "$(tiles[best_idx])×$(tiles[best_idx])"

        @printf("%-10d %-12.1f %-12.1f %-12.1f %-12.1f %-10s\n",
                N, row_vals[1], row_vals[2], row_vals[3], g_cb, best_name)
    end

    # -----------------------------------------------------------
    # Sweep 2 — Détails pour TOUTES les tailles
    # -----------------------------------------------------------
    println()
    println("=" ^ 65)
    println("SWEEP 2: Détails pour toutes les tailles de matrices")
    println("=" ^ 65)

    for N in sizes
        println()
        println("--- N = $N ---")
        println(@sprintf("%-10s %-12s %-14s %-14s %-12s",
                         "Tile", "GFLOPS", "Time (ms)", "Blocs GPU", "% cuBLAS"))
        println("-" ^ 65)

        g_cb, t_cb = all_results[(N, :cublas)]
        n_blocs_cb = "N/A"

        for tile in tiles
            g, t = all_results[(N, tile)]
            n_blocs = (N ÷ tile)^2  # nombre de blocs pour cette matrice
            @printf("%-10s %-12.2f %-14.2f %-14d %-12s\n",
                    "$(tile)×$(tile)", g, t,
                    n_blocs,
                    @sprintf("%.1f%%", g/g_cb*100))
        end
        @printf("%-10s %-12.2f %-14.2f %-14s %-12s\n",
                "cuBLAS", g_cb, t_cb, "auto", "100.0%")
    end

    # -----------------------------------------------------------
    # Résumé final — meilleure tile par matrice
    # -----------------------------------------------------------
    println()
    println("=" ^ 65)
    println("RÉSUMÉ FINAL: Meilleure tile par taille de matrice")
    println("=" ^ 65)
    println(@sprintf("%-10s %-14s %-12s %-14s %-12s",
                     "N", "Meilleure tile", "GFLOPS", "Time (ms)", "% cuBLAS"))
    println("-" ^ 65)

    for N in sizes
        best_tile = 0
        best_g = 0.0
        best_t = 0.0
        for tile in tiles
            g, t = all_results[(N, tile)]
            if g > best_g
                best_g = g
                best_t = t
                best_tile = tile
            end
        end
        g_cb, _ = all_results[(N, :cublas)]
        @printf("%-10d %-14s %-12.2f %-14.2f %-12s\n",
                N, "$(best_tile)×$(best_tile)", best_g, best_t,
                @sprintf("%.1f%%", best_g/g_cb*100))
    end
end

run()