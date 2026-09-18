/* Heureka-Test: y = W*x mit W im NWC-Format, dekodiert IN REGISTERN waehrend der Rechnung.
   Kein entpackter Gewichtswert beruehrt je den Speicher.

   Kernel:
     nativ_schnell  BF16 direkt, ein Warp je Zeile, 16-Byte-Laden      -> Tempo-Baseline
     nativ_flach    BF16 direkt, gleiche Struktur wie fusioniert        -> Korrektheits-Baseline
     fusioniert     NWC: rANS-Dekodierung + Mantisse roh + FMA          -> der Kandidat

   nutzung: matvec <W.raw> <W.nwc> <K>                                                   */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#define SCALE_BITS 12
#define TOT        (1u << SCALE_BITS)
#define RANS_L16   (1u << 16)

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

/* ---------- Baseline 1: natives BF16, ein Warp je Zeile, uint4-Laden ---------- */
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

/* ---------- Baseline 2: natives BF16 in der Struktur des fusionierten Kernels ---------- */
__global__ void __launch_bounds__(128)
kern_nativ_flach(const uint16_t *__restrict__ W, const float *__restrict__ x, float *__restrict__ y,
                 uint64_t n, uint32_t block, uint32_t K)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    if (basis >= n) return;
    uint32_t n_lane = block / 32;

    uint64_t p0 = basis + lane;
    uint32_t m = (uint32_t)(p0 / K), kk = (uint32_t)(p0 % K);
    float acc = 0.f;
    for (uint32_t i = 0; i < n_lane; i++) {
        float w = bf16_zu_f32(__ldg(W + p0 + (uint64_t)i * 32));
        acc = fmaf(w, __ldg(x + kk), acc);
        kk += 32;
        if (kk >= K) {                                   /* warp-uniform, da K % 32 == 0 */
            kk -= K;
            float s = warp_summe(acc);
            if (lane == 0) atomicAdd(y + m, s);
            acc = 0.f; m++;
        }
    }
    float s = warp_summe(acc);
    if (lane == 0 && s != 0.f) atomicAdd(y + m, s);
}

/* ---------- Der Kandidat: fusionierte rANS-Dekodierung + Matvec ---------- */
__global__ void __launch_bounds__(128)
kern_fusioniert(const uint8_t  *__restrict__ daten,
                const uint32_t *__restrict__ basen,
                const uint16_t *__restrict__ subrel,
                const uint32_t *__restrict__ tab,
                const uint8_t  *__restrict__ low,
                const float    *__restrict__ x,
                float          *__restrict__ y,
                uint64_t n, uint32_t block, uint32_t K)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    if (basis >= n) return;
    uint32_t n_lane = block / 32;

    const uint8_t *start = daten + basen[b] + subrel[(uint64_t)b * 32 + lane];
    uint32_t xs = *(const uint32_t*)start;
    uint64_t wb = *(const uint32_t*)(start + 4); int nb = 32;
    const uint2 *p = (const uint2*)(start + 8);

    uint64_t p0 = basis + lane;
    uint32_t m = (uint32_t)(p0 / K), kk = (uint32_t)(p0 % K);
    const uint8_t *lo = low + p0;
    float acc = 0.f;

    #pragma unroll 4
    for (uint32_t i = 0; i < n_lane; i++) {
        uint32_t e = __ldg(tab + (xs & (TOT - 1)));                  /* Exponentbyte dekodieren */
        uint32_t hi = e & 0xFFu;
        xs = (((e >> 8) & 0xFFFu) + 1u) * (xs >> SCALE_BITS) + (e >> 20);
        if (xs < RANS_L16) {
            if (nb == 0) { uint2 v = *p++; wb = (uint64_t)v.x | ((uint64_t)v.y << 32); nb = 64; }
            xs = (xs << 16) | (uint32_t)(wb & 0xFFFFu); wb >>= 16; nb -= 16;
        }
        uint32_t lob = __ldg(lo + (uint64_t)i * 32);                 /* Mantissenbyte roh */
        float w = bf16_zu_f32((hi << 8) | lob);                      /* BF16 nur in Registern */
        acc = fmaf(w, __ldg(x + kk), acc);
        kk += 32;
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

static float zeit_best(cudaEvent_t a, cudaEvent_t b) { float ms; cudaEventElapsedTime(&ms, a, b); return ms; }

int main(int argc, char **argv) {
    if (argc < 4) { printf("nutzung: matvec <W.raw> <W.nwc> <K>\n"); return 2; }
    uint32_t K = (uint32_t)atoi(argv[3]);
    if (K % 32) { printf("K muss durch 32 teilbar sein\n"); return 1; }

    /* --- W roh --- */
    FILE *fw = fopen(argv[1], "rb"); if (!fw) { perror("W.raw"); return 1; }
    fseek(fw, 0, SEEK_END); long wg = ftell(fw); fseek(fw, 0, SEEK_SET);
    uint8_t *h_W = (uint8_t*)malloc(wg);
    if (fread(h_W, 1, wg, fw) != (size_t)wg) { printf("read W\n"); return 1; }
    fclose(fw);
    uint64_t n = (uint64_t)wg / 2; uint32_t M = (uint32_t)(n / K);
    if ((uint64_t)M * K != n) { printf("W.raw ist keine %u-spaltige Matrix\n", K); return 1; }

    /* --- W komprimiert --- */
    FILE *fn = fopen(argv[2], "rb"); if (!fn) { perror("W.nwc"); return 1; }
    fseek(fn, 0, SEEK_END); long fg = ftell(fn); fseek(fn, 0, SEEK_SET);
    uint8_t *h_buf = (uint8_t*)malloc(fg);
    if (fread(h_buf, 1, fg, fn) != (size_t)fg) { printf("read nwc\n"); return 1; }
    fclose(fn);
    if (memcmp(h_buf, "NWC2", 4) || h_buf[4] != 3 || h_buf[5] != 32 || h_buf[7] != 16) {
        printf("brauche NWC2 v3, lanes=32, renorm=16\n"); return 1;
    }
    uint64_t n_high; uint32_t stride, block, n_bloecke;
    memcpy(&n_high, h_buf + 8, 8); memcpy(&stride, h_buf + 16, 4);
    memcpy(&block,  h_buf + 20, 4); memcpy(&n_bloecke, h_buf + 24, 4);
    if (n_high != n || stride != 0 || n % block) { printf("nwc passt nicht zu raw (oder n nicht Vielfaches von block)\n"); return 1; }
    size_t off_basen = 28 + 512, off_sub = off_basen + 4ull * n_bloecke;
    size_t off_daten = off_sub + 2ull * n_bloecke * 32, off_low = (size_t)fg - n_high;
    size_t daten_len = off_low - off_daten;

    const uint16_t *freq16 = (const uint16_t*)(h_buf + 28);
    uint32_t kum = 0, *h_tab = (uint32_t*)malloc(TOT * 4);
    for (uint32_t s = 0; s < 256; s++) { uint32_t f = freq16[s];
        for (uint32_t j = 0; j < f; j++) h_tab[kum + j] = s | ((f - 1) << 8) | (j << 20); kum += f; }

    /* --- x deterministisch --- */
    float *h_x = (float*)malloc(4ull * K);
    uint32_t seed = 12345;
    for (uint32_t k = 0; k < K; k++) { seed = seed * 1664525u + 1013904223u; h_x[k] = ((seed >> 8) & 0xFFFF) / 65536.f - 0.5f; }

    printf("W: %u x %u BF16 = %.1f MB   komprimiert: %.1f MB (Rate %.4f)   %u NWC-Bloecke\n",
           M, K, wg / 1e6, fg / 1e6, (double)fg / wg, n_bloecke);

    uint16_t *d_W; float *d_x, *d_y1, *d_y2, *d_y3; uint8_t *d_daten, *d_low; uint32_t *d_basen, *d_tab; uint16_t *d_sub;
    PRUEF(cudaMalloc(&d_W, wg));
    PRUEF(cudaMalloc(&d_x, 4ull * K));
    PRUEF(cudaMalloc(&d_y1, 4ull * M)); PRUEF(cudaMalloc(&d_y2, 4ull * M)); PRUEF(cudaMalloc(&d_y3, 4ull * M));
    PRUEF(cudaMalloc(&d_daten, daten_len + 16));
    PRUEF(cudaMalloc(&d_low, n_high));
    PRUEF(cudaMalloc(&d_basen, 4ull * n_bloecke));
    PRUEF(cudaMalloc(&d_sub, 2ull * n_bloecke * 32));
    PRUEF(cudaMalloc(&d_tab, TOT * 4));
    PRUEF(cudaMemcpy(d_W, h_W, wg, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_x, h_x, 4ull * K, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_daten, h_buf + off_daten, daten_len, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_low, h_buf + off_low, n_high, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_basen, h_buf + off_basen, 4ull * n_bloecke, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_sub, h_buf + off_sub, 2ull * n_bloecke * 32, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_tab, h_tab, TOT * 4, cudaMemcpyHostToDevice));

    int bl_schnell = (int)(((uint64_t)M * 32 + 127) / 128);
    int bl_flach   = (int)(((uint64_t)n_bloecke * 32 + 127) / 128);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float t1 = 1e30f, t2 = 1e30f, t3 = 1e30f;
    for (int w = 0; w < 12; w++) {
        cudaEventRecord(e0);
        kern_nativ_schnell<<<bl_schnell, 128>>>(d_W, d_x, d_y1, M, K);
        cudaEventRecord(e1); PRUEF(cudaEventSynchronize(e1));
        if (w >= 3) t1 = fminf(t1, zeit_best(e0, e1));

        PRUEF(cudaMemset(d_y2, 0, 4ull * M));
        cudaEventRecord(e0);
        kern_nativ_flach<<<bl_flach, 128>>>(d_W, d_x, d_y2, n, block, K);
        cudaEventRecord(e1); PRUEF(cudaEventSynchronize(e1));
        if (w >= 3) t2 = fminf(t2, zeit_best(e0, e1));

        PRUEF(cudaMemset(d_y3, 0, 4ull * M));
        cudaEventRecord(e0);
        kern_fusioniert<<<bl_flach, 128>>>(d_daten, d_basen, d_sub, d_tab, d_low, d_x, d_y3, n, block, K);
        cudaEventRecord(e1); PRUEF(cudaEventSynchronize(e1));
        if (w >= 3) t3 = fminf(t3, zeit_best(e0, e1));
    }
    PRUEF(cudaGetLastError());

    float *y1 = (float*)malloc(4ull * M), *y2 = (float*)malloc(4ull * M), *y3 = (float*)malloc(4ull * M);
    PRUEF(cudaMemcpy(y1, d_y1, 4ull * M, cudaMemcpyDeviceToHost));
    PRUEF(cudaMemcpy(y2, d_y2, 4ull * M, cudaMemcpyDeviceToHost));
    PRUEF(cudaMemcpy(y3, d_y3, 4ull * M, cudaMemcpyDeviceToHost));

    /* CPU-Referenz fuer einige Zeilen, Fehlermasse */
    double max_rel_32 = 0, max_rel_31 = 0, max_abs_ref = 0; int exakt_32 = 0;
    for (uint32_t m = 0; m < M; m++) {
        double d32 = fabs((double)y3[m] - y2[m]), d31 = fabs((double)y3[m] - y1[m]);
        double sk = fmax(fabs((double)y2[m]), 1e-6);
        if (d32 / sk > max_rel_32) max_rel_32 = d32 / sk;
        if (d31 / sk > max_rel_31) max_rel_31 = d31 / sk;
        if (y3[m] == y2[m]) exakt_32++;
    }
    for (uint32_t m = 0; m < M; m += M / 64) {
        double ref = 0;
        for (uint32_t k = 0; k < K; k++) {
            uint32_t bits = (uint32_t)(((const uint16_t*)h_W)[(uint64_t)m * K + k]) << 16;
            float w; memcpy(&w, &bits, 4); ref += (double)w * h_x[k];
        }
        double d = fabs(ref - y3[m]) / fmax(fabs(ref), 1e-6);
        if (d > max_abs_ref) max_abs_ref = d;
    }

    double mb_nativ = wg / 1e6, mb_fus = (daten_len + n_high) / 1e6;
    printf("\n  %-34s %8.3f ms   %7.1f GB/s gelesen\n", "nativ BF16 (Warp je Zeile, uint4)", t1, mb_nativ / t1);
    printf("  %-34s %8.3f ms   %7.1f GB/s gelesen\n", "nativ BF16 (flache Struktur)", t2, mb_nativ / t2);
    printf("  %-34s %8.3f ms   %7.1f GB/s gelesen  (%.1f GB/s Gewichte-Aequivalent)\n",
           "FUSIONIERT NWC (dekodiert im FMA)", t3, mb_fus / t3, mb_nativ / t3);
    printf("\n  Speedup fusioniert vs nativ-schnell : %.3fx\n", t1 / t3);
    printf("  Speedup fusioniert vs nativ-flach   : %.3fx\n", t2 / t3);
    printf("\n  Korrektheit: max rel. Abw. zu nativ-flach %.2e (%d/%u Zeilen bitidentisch), zu nativ-schnell %.2e,\n"
           "               zu CPU-double-Referenz (64 Zeilen) %.2e\n", max_rel_32, exakt_32, M, max_rel_31, max_abs_ref);
    return 0;
}
