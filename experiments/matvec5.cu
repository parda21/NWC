/* Heureka-Test, Runde 5: tANS/FSE (Formatversion 5).
   Pro Symbol: Tabellenzugriff, Maske, Addition -- keine Multiplikation, keine Schleife.
   Nachladen des Bitpuffers verzweigungsfrei per Auswahl; nur der Ladebefehl selbst ist praediziert.
   nutzung: matvec5 <W.raw> <W5.nwc> <K>                                                    */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#ifndef MINB
#define MINB 12
#endif
#define TLOG 12
#define L    (1u << TLOG)

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

__global__ void __launch_bounds__(128, MINB)
kern_fusioniert5(const uint8_t  *__restrict__ daten,
                 const uint32_t *__restrict__ basen,
                 const uint16_t *__restrict__ subrel,
                 const uint32_t *__restrict__ tab,       /* sym | nb<<8 | neu<<12 */
                 const uint8_t  *__restrict__ low,
                 const float    *__restrict__ x,
                 float          *__restrict__ y,
                 uint64_t n, uint32_t block, uint32_t K)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    if (basis >= n) return;
    uint32_t n_chunks = block / 512;

    const uint8_t *start = daten + basen[b] + subrel[(uint64_t)b * 32 + lane];
    uint32_t u = *(const uint32_t*)start;
    int pos = (int)*(const uint32_t*)(start + 4);       /* Startposition im ersten Wort */
    const uint2 *p = (const uint2*)(start + 8);
    uint2 cur = __ldg(p++);
    uint64_t buf = (uint64_t)cur.x | ((uint64_t)cur.y << 32);
    uint2 nxt = __ldg(p++);
    uint32_t wi = 0;                                    /* naechstes Wort: 0 -> nxt.x, 1 -> nxt.y */

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
            uint32_t e  = __ldg(tab + u);
            uint32_t hi = e & 0xFFu;
            uint32_t nb = (e >> 8) & 0xFu;
            u = (e >> 12) + ((uint32_t)(buf >> pos) & ((1u << nb) - 1u));
            pos += nb;
            /* Nachladen ohne Verzweigung */
            bool voll = pos >= 32;
            uint32_t w = wi ? nxt.y : nxt.x;
            uint64_t buf2 = (buf >> 32) | ((uint64_t)w << 32);
            buf = voll ? buf2 : buf;
            pos = voll ? pos - 32 : pos;
            uint32_t wi2 = wi ^ 1u;
            if (voll && wi2 == 0) nxt = __ldg(p++);
            wi = voll ? wi2 : wi;

            float wgt = bf16_zu_f32((hi << 8) | u4_byte(chunk, j));
            acc = fmaf(wgt, f4_el(xq, j & 3), acc);
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

static int highbit(uint32_t v) { int r = 0; while (v >>= 1) r++; return r; }
static float zeit(cudaEvent_t a, cudaEvent_t b) { float ms; cudaEventElapsedTime(&ms, a, b); return ms; }

int main(int argc, char **argv) {
    if (argc < 4) { printf("nutzung: matvec5 <W.raw> <W5.nwc> <K>\n"); return 2; }
    uint32_t K = (uint32_t)atoi(argv[3]);
    if (K % 512) { printf("K muss durch 512 teilbar sein\n"); return 1; }

    FILE *fw = fopen(argv[1], "rb"); if (!fw) { perror("W.raw"); return 1; }
    fseek(fw, 0, SEEK_END); long wg = ftell(fw); fseek(fw, 0, SEEK_SET);
    uint8_t *h_W = (uint8_t*)malloc(wg);
    if (fread(h_W, 1, wg, fw) != (size_t)wg) { printf("read W\n"); return 1; }
    fclose(fw);
    uint64_t n = (uint64_t)wg / 2; uint32_t M = (uint32_t)(n / K);
    if ((uint64_t)M * K != n) { printf("W.raw ist keine %u-spaltige Matrix\n", K); return 1; }

    FILE *fn = fopen(argv[2], "rb"); if (!fn) { perror("W.nwc"); return 1; }
    fseek(fn, 0, SEEK_END); long fg = ftell(fn); fseek(fn, 0, SEEK_SET);
    uint8_t *h_buf = (uint8_t*)malloc(fg);
    if (fread(h_buf, 1, fg, fn) != (size_t)fg) { printf("read nwc\n"); return 1; }
    fclose(fn);
    if (memcmp(h_buf, "NWC2", 4) || h_buf[4] != 5 || h_buf[5] != 32 || h_buf[6] != TLOG) {
        printf("brauche NWC2 Formatversion 5 (nwcenc_fse)\n"); return 1;
    }
    uint64_t n_high; uint32_t stride, block, n_bloecke;
    memcpy(&n_high, h_buf + 8, 8); memcpy(&stride, h_buf + 16, 4);
    memcpy(&block,  h_buf + 20, 4); memcpy(&n_bloecke, h_buf + 24, 4);
    if (n_high != n || stride != 0 || n % block || block % 512) { printf("nwc passt nicht\n"); return 1; }
    size_t off_basen = 28 + 512, off_sub = off_basen + 4ull * n_bloecke;
    size_t off_daten = off_sub + 2ull * n_bloecke * 32, off_low = (size_t)fg - n_high;
    size_t daten_len = off_low - off_daten;

    /* Dekodertabelle wie im Encoder */
    const uint16_t *freq = (const uint16_t*)(h_buf + 28);
    uint8_t *tsym = (uint8_t*)malloc(L); uint32_t *h_tab = (uint32_t*)malloc(L * 4);
    uint32_t step = (L >> 1) + (L >> 3) + 3, pos = 0;
    for (int s = 0; s < 256; s++) for (uint32_t i = 0; i < freq[s]; i++) { tsym[pos] = (uint8_t)s; pos = (pos + step) & (L - 1); }
    uint32_t next[256]; for (int s = 0; s < 256; s++) next[s] = freq[s];
    for (uint32_t u = 0; u < L; u++) {
        int s = tsym[u]; uint32_t xv = next[s]++; int nb = TLOG - highbit(xv);
        h_tab[u] = (uint32_t)s | ((uint32_t)nb << 8) | (((xv << nb) - L) << 12);
    }

    float *h_x = (float*)malloc(4ull * K);
    uint32_t seed = 12345;
    for (uint32_t k = 0; k < K; k++) { seed = seed * 1664525u + 1013904223u; h_x[k] = ((seed >> 8) & 0xFFFF) / 65536.f - 0.5f; }

    uint16_t *d_W; float *d_x, *d_y1, *d_y3; uint8_t *d_daten, *d_low; uint32_t *d_basen, *d_tab; uint16_t *d_sub;
    PRUEF(cudaMalloc(&d_W, wg));
    PRUEF(cudaMalloc(&d_x, 4ull * K));
    PRUEF(cudaMalloc(&d_y1, 4ull * M)); PRUEF(cudaMalloc(&d_y3, 4ull * M));
    PRUEF(cudaMalloc(&d_daten, daten_len + 64));
    PRUEF(cudaMalloc(&d_low, n_high + 1024));
    PRUEF(cudaMalloc(&d_basen, 4ull * n_bloecke));
    PRUEF(cudaMalloc(&d_sub, 2ull * n_bloecke * 32));
    PRUEF(cudaMalloc(&d_tab, L * 4));
    PRUEF(cudaMemcpy(d_W, h_W, wg, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_x, h_x, 4ull * K, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_daten, h_buf + off_daten, daten_len, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_low, h_buf + off_low, n_high, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_basen, h_buf + off_basen, 4ull * n_bloecke, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_sub, h_buf + off_sub, 2ull * n_bloecke * 32, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_tab, h_tab, L * 4, cudaMemcpyHostToDevice));
    PRUEF(cudaFuncSetAttribute(kern_fusioniert5, cudaFuncAttributePreferredSharedMemoryCarveout, 0));

    int bl_schnell = (int)(((uint64_t)M * 32 + 127) / 128);
    int bl_fus     = (int)(((uint64_t)n_bloecke * 32 + 127) / 128);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float t1 = 1e30f, t3 = 1e30f;
    for (int w = 0; w < 15; w++) {
        cudaEventRecord(e0);
        kern_nativ_schnell<<<bl_schnell, 128>>>(d_W, d_x, d_y1, M, K);
        cudaEventRecord(e1); PRUEF(cudaEventSynchronize(e1));
        if (w >= 3) t1 = fminf(t1, zeit(e0, e1));
        PRUEF(cudaMemset(d_y3, 0, 4ull * M));
        cudaEventRecord(e0);
        kern_fusioniert5<<<bl_fus, 128>>>(d_daten, d_basen, d_sub, d_tab, d_low, d_x, d_y3, n, block, K);
        cudaEventRecord(e1); PRUEF(cudaEventSynchronize(e1));
        if (w >= 3) t3 = fminf(t3, zeit(e0, e1));
    }
    PRUEF(cudaGetLastError());

    float *y1 = (float*)malloc(4ull * M), *y3 = (float*)malloc(4ull * M);
    PRUEF(cudaMemcpy(y1, d_y1, 4ull * M, cudaMemcpyDeviceToHost));
    PRUEF(cudaMemcpy(y3, d_y3, 4ull * M, cudaMemcpyDeviceToHost));
    double max_ref = 0; int schlecht = 0;
    for (uint32_t m = 0; m < M; m += 97) {
        double ref = 0;
        for (uint32_t k = 0; k < K; k++) {
            uint32_t bits = (uint32_t)(((const uint16_t*)h_W)[(uint64_t)m * K + k]) << 16;
            float w; memcpy(&w, &bits, 4); ref += (double)w * h_x[k];
        }
        double d = fabs(ref - y3[m]) / fmax(fabs(ref), 1e-3);
        if (d > max_ref) max_ref = d;
        if (d > 1e-3) schlecht++;
    }
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, kern_fusioniert5);
    double mb_nativ = wg / 1e6, mb_fus = (daten_len + n_high) / 1e6;
    printf("FSE MINB=%-2d regs=%2d | nativ %.3f ms | fusioniert %.3f ms  %.0f GB/s gelesen (%.0f GB/s Aequiv.) | "
           "Speedup %.3fx | Fehler %.1e, Ausreisser %d\n",
           MINB, fa.numRegs, t1, t3, mb_fus / t3, mb_nativ / t3, t1 / t3, max_ref, schlecht);
    return 0;
}
