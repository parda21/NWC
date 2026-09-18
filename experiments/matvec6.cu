/* Absicherung: fusionierter rANS-Matvec (Formatversion 4) gegen zwei Baselines,
   mehrere K, Median ueber viele Laeufe.
     - nativ (handgeschrieben, Warp je Zeile, uint4)
     - cuBLAS GemmEx BF16 -> FP32
   nutzung: matvec6 <W.raw> <W4.nwc> <K1> [K2 ...]      (n muss Vielfaches jedes K sein)      */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define SCALE_BITS 12
#define TOT        (1u << SCALE_BITS)
#define RANS_L16   (1u << 16)
#define ITER       60
#define WARM       10

static void pruef(cudaError_t e, int zeile) {
    if (e != cudaSuccess) { printf("CUDA-Fehler @ %d: %s\n", zeile, cudaGetErrorString(e)); exit(1); }
}
#define PRUEF(x) pruef((x), __LINE__)

__device__ __forceinline__ float bf16_zu_f32(uint32_t bits16) { return __uint_as_float(bits16 << 16); }
__device__ __forceinline__ float warp_summe(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__global__ void __launch_bounds__(128)
kern_nativ_schnell(const uint16_t *__restrict__ W, const float *__restrict__ x, float *__restrict__ y,
                   uint32_t M, uint32_t K)
{
    uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    uint32_t lane = threadIdx.x & 31;
    if (warp >= M) return;
    const uint4 *zeile = (const uint4*)(W + (uint64_t)warp * K);
    float acc = 0.f;
    for (uint32_t k = lane * 8; k < K; k += 256) {
        uint4 v = __ldg(zeile + (k >> 3));
        float4 xa = __ldg((const float4*)(x + k)), xb = __ldg((const float4*)(x + k + 4));
        acc = fmaf(bf16_zu_f32(v.x & 0xFFFF), xa.x, acc); acc = fmaf(bf16_zu_f32(v.x >> 16), xa.y, acc);
        acc = fmaf(bf16_zu_f32(v.y & 0xFFFF), xa.z, acc); acc = fmaf(bf16_zu_f32(v.y >> 16), xa.w, acc);
        acc = fmaf(bf16_zu_f32(v.z & 0xFFFF), xb.x, acc); acc = fmaf(bf16_zu_f32(v.z >> 16), xb.y, acc);
        acc = fmaf(bf16_zu_f32(v.w & 0xFFFF), xb.z, acc); acc = fmaf(bf16_zu_f32(v.w >> 16), xb.w, acc);
    }
    acc = warp_summe(acc);
    if (lane == 0) y[warp] = acc;
}

__device__ __forceinline__ uint32_t u4_byte(const uint4 &c, int j) {
    uint32_t w = (j < 4) ? c.x : (j < 8) ? c.y : (j < 12) ? c.z : c.w;
    return (w >> (8 * (j & 3))) & 0xFFu;
}
__device__ __forceinline__ float f4_el(const float4 &v, int j) {
    return (j == 0) ? v.x : (j == 1) ? v.y : (j == 2) ? v.z : v.w;
}

__global__ void __launch_bounds__(128, 12)
kern_fusioniert(const uint8_t  *__restrict__ daten, const uint32_t *__restrict__ basen,
                const uint16_t *__restrict__ subrel, const uint32_t *__restrict__ tab,
                const uint8_t  *__restrict__ low, const float *__restrict__ x, float *__restrict__ y,
                uint64_t n, uint32_t block, uint32_t K)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    if (basis >= n) return;
    uint32_t n_chunks = block / 512;

    const uint8_t *start = daten + basen[b] + subrel[(uint64_t)b * 32 + lane];
    uint32_t xs = *(const uint32_t*)start;
    uint64_t wb = *(const uint32_t*)(start + 4); int nb = 32;
    const uint2 *p = (const uint2*)(start + 8);
    uint2 nxt = __ldg(p++);

    uint32_t m  = (uint32_t)(basis / K);
    uint32_t kk = (uint32_t)(basis % K) + lane * 16;
    const uint4 *lo4 = (const uint4*)(low + basis) + lane;
    uint4 chunk_nxt = __ldg(lo4);
    float acc = 0.f;

    for (uint32_t c = 0; c < n_chunks; c++) {
        uint4 chunk = chunk_nxt;
        if (c + 1 < n_chunks) chunk_nxt = __ldg(lo4 + 32 * (c + 1));
        const float4 *xp = (const float4*)(x + kk);
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            float4 xq;
            if ((j & 3) == 0) xq = __ldg(xp + (j >> 2));
            uint32_t e = __ldg(tab + (xs & (TOT - 1)));
            uint32_t hi = e & 0xFFu;
            xs = (((e >> 8) & 0xFFFu) + 1u) * (xs >> SCALE_BITS) + (e >> 20);
            if (xs < RANS_L16) {
                if (nb == 0) { wb = (uint64_t)nxt.x | ((uint64_t)nxt.y << 32); nb = 64; nxt = __ldg(p++); }
                xs = (xs << 16) | (uint32_t)(wb & 0xFFFFu); wb >>= 16; nb -= 16;
            }
            float w = bf16_zu_f32((hi << 8) | u4_byte(chunk, j));
            acc = fmaf(w, f4_el(xq, j & 3), acc);
        }
        kk += 512;
        if (kk >= K) {
            kk -= K;
            float s = warp_summe(acc);
            if (lane == 0) atomicAdd(y + m, s);
            acc = 0.f; m++;
        }
    }
    float s = warp_summe(acc);
    if (lane == 0 && s != 0.f) atomicAdd(y + m, s);
}

static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char **argv) {
    if (argc < 4) { printf("nutzung: matvec6 <W.raw> <W4.nwc> <K...>\n"); return 2; }

    FILE *fw = fopen(argv[1], "rb"); if (!fw) { perror("W.raw"); return 1; }
    fseek(fw, 0, SEEK_END); long wg = ftell(fw); fseek(fw, 0, SEEK_SET);
    uint8_t *h_W = (uint8_t*)malloc(wg);
    if (fread(h_W, 1, wg, fw) != (size_t)wg) { printf("read W\n"); return 1; }
    fclose(fw);
    uint64_t n = (uint64_t)wg / 2;

    FILE *fn = fopen(argv[2], "rb"); if (!fn) { perror("W.nwc"); return 1; }
    fseek(fn, 0, SEEK_END); long fg = ftell(fn); fseek(fn, 0, SEEK_SET);
    uint8_t *h_buf = (uint8_t*)malloc(fg);
    if (fread(h_buf, 1, fg, fn) != (size_t)fg) { printf("read nwc\n"); return 1; }
    fclose(fn);
    if (memcmp(h_buf, "NWC2", 4) || h_buf[4] != 4) { printf("brauche Formatversion 4\n"); return 1; }
    uint64_t n_high; uint32_t stride, block, n_bloecke;
    memcpy(&n_high, h_buf + 8, 8); memcpy(&stride, h_buf + 16, 4);
    memcpy(&block,  h_buf + 20, 4); memcpy(&n_bloecke, h_buf + 24, 4);
    if (n_high != n || stride != 0 || n % block) { printf("nwc passt nicht\n"); return 1; }
    size_t off_basen = 28 + 512, off_sub = off_basen + 4ull * n_bloecke;
    size_t off_daten = off_sub + 2ull * n_bloecke * 32, off_low = (size_t)fg - n_high;
    size_t daten_len = off_low - off_daten;
    const uint16_t *freq16 = (const uint16_t*)(h_buf + 28);
    uint32_t kum = 0, *h_tab = (uint32_t*)malloc(TOT * 4);
    for (uint32_t s = 0; s < 256; s++) { uint32_t f = freq16[s];
        for (uint32_t j = 0; j < f; j++) h_tab[kum + j] = s | ((f - 1) << 8) | (j << 20); kum += f; }

    uint16_t *d_W; uint8_t *d_daten, *d_low; uint32_t *d_basen, *d_tab; uint16_t *d_sub;
    PRUEF(cudaMalloc(&d_W, wg));
    PRUEF(cudaMalloc(&d_daten, daten_len + 64));
    PRUEF(cudaMalloc(&d_low, n_high + 1024));
    PRUEF(cudaMalloc(&d_basen, 4ull * n_bloecke));
    PRUEF(cudaMalloc(&d_sub, 2ull * n_bloecke * 32));
    PRUEF(cudaMalloc(&d_tab, TOT * 4));
    PRUEF(cudaMemcpy(d_W, h_W, wg, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_daten, h_buf + off_daten, daten_len, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_low, h_buf + off_low, n_high, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_basen, h_buf + off_basen, 4ull * n_bloecke, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_sub, h_buf + off_sub, 2ull * n_bloecke * 32, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_tab, h_tab, TOT * 4, cudaMemcpyHostToDevice));
    PRUEF(cudaFuncSetAttribute(kern_fusioniert, cudaFuncAttributePreferredSharedMemoryCarveout, 0));

    cublasHandle_t cb; cublasCreate(&cb);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("W: %.1f MB BF16, komprimiert %.1f MB (Rate %.4f), %d Laeufe, Median\n\n", wg / 1e6, fg / 1e6, (double)fg / wg, ITER);
    printf("  %6s | %10s %10s %10s | %9s %9s | %9s\n", "K", "cuBLAS", "nativ", "NWC", "vs cuBLAS", "vs nativ", "Fehler");
    printf("  -------+----------------------------------+---------------------+----------\n");

    for (int a = 3; a < argc; a++) {
        uint32_t K = (uint32_t)atoi(argv[a]);
        if (K % 512 || n % K) { printf("  %6u | uebersprungen (K%%512 oder n%%K)\n", K); continue; }
        uint32_t M = (uint32_t)(n / K);

        float *h_x = (float*)malloc(4ull * K); uint16_t *h_xb = (uint16_t*)malloc(2ull * K);
        uint32_t seed = 12345;
        for (uint32_t k = 0; k < K; k++) {
            seed = seed * 1664525u + 1013904223u; h_x[k] = ((seed >> 8) & 0xFFFF) / 65536.f - 0.5f;
            uint32_t bits; memcpy(&bits, &h_x[k], 4); h_xb[k] = (uint16_t)(bits >> 16);   /* bf16-Kopie fuer cuBLAS */
        }
        float *d_x, *d_y1, *d_y2, *d_y3; uint16_t *d_xb;
        PRUEF(cudaMalloc(&d_x, 4ull * K)); PRUEF(cudaMalloc(&d_xb, 2ull * K));
        PRUEF(cudaMalloc(&d_y1, 4ull * M)); PRUEF(cudaMalloc(&d_y2, 4ull * M)); PRUEF(cudaMalloc(&d_y3, 4ull * M));
        PRUEF(cudaMemcpy(d_x, h_x, 4ull * K, cudaMemcpyHostToDevice));
        PRUEF(cudaMemcpy(d_xb, h_xb, 2ull * K, cudaMemcpyHostToDevice));

        int bl_schnell = (int)(((uint64_t)M * 32 + 127) / 128);
        int bl_fus     = (int)(((uint64_t)n_bloecke * 32 + 127) / 128);
        const float eins = 1.f, null = 0.f;
        std::vector<float> tc, tn, tf;

        for (int w = 0; w < ITER + WARM; w++) {
            float ms;
            /* cuBLAS: y = W(MxK, row-major) * x  ==  col-major W^T (KxM) mit op=T */
            cudaEventRecord(e0);
            cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, K, &eins,
                         d_W, CUDA_R_16BF, K, d_xb, CUDA_R_16BF, K, &null,
                         d_y2, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            cudaEventRecord(e1); PRUEF(cudaEventSynchronize(e1));
            cudaEventElapsedTime(&ms, e0, e1); if (w >= WARM) tc.push_back(ms);

            cudaEventRecord(e0);
            kern_nativ_schnell<<<bl_schnell, 128>>>(d_W, d_x, d_y1, M, K);
            cudaEventRecord(e1); PRUEF(cudaEventSynchronize(e1));
            cudaEventElapsedTime(&ms, e0, e1); if (w >= WARM) tn.push_back(ms);

            PRUEF(cudaMemsetAsync(d_y3, 0, 4ull * M));
            cudaEventRecord(e0);
            kern_fusioniert<<<bl_fus, 128>>>(d_daten, d_basen, d_sub, d_tab, d_low, d_x, d_y3, n, block, K);
            cudaEventRecord(e1); PRUEF(cudaEventSynchronize(e1));
            cudaEventElapsedTime(&ms, e0, e1); if (w >= WARM) tf.push_back(ms);
        }
        PRUEF(cudaGetLastError());

        float *y3 = (float*)malloc(4ull * M);
        PRUEF(cudaMemcpy(y3, d_y3, 4ull * M, cudaMemcpyDeviceToHost));
        double max_ref = 0;
        for (uint32_t m = 0; m < M; m += (M / 200 ? M / 200 : 1)) {
            double ref = 0;
            for (uint32_t k = 0; k < K; k++) {
                uint32_t bits = (uint32_t)(((const uint16_t*)h_W)[(uint64_t)m * K + k]) << 16;
                float wv; memcpy(&wv, &bits, 4); ref += (double)wv * h_x[k];
            }
            double d = fabs(ref - y3[m]) / fmax(fabs(ref), 1e-3);
            if (d > max_ref) max_ref = d;
        }
        float mc = median(tc), mn = median(tn), mf = median(tf);
        printf("  %6u | %7.3f ms %7.3f ms %7.3f ms | %8.3fx %8.3fx | %8.1e\n", K, mc, mn, mf, mc / mf, mn / mf, max_ref);

        cudaFree(d_x); cudaFree(d_xb); cudaFree(d_y1); cudaFree(d_y2); cudaFree(d_y3); free(h_x); free(h_xb); free(y3);
    }
    cublasDestroy(cb);
    return 0;
}
