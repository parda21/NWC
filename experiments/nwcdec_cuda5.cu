/* NWC2-Dekoder v5 (Formatversion 3): wie v4/LOAD64, aber
   - Lookup-Tabelle per __ldg aus globalem Speicher (bleibt im L1, 16 KB) statt Shared Memory
     -> kein Shared-Memory-Limit mehr, Belegung nur noch durch Register begrenzt
   - CUDA-Blockgroesse TH waehlbar (-DTH=64/128/256) gegen Wellen-Quantisierung        */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef TH
#define TH 128
#endif
#define SCALE_BITS 12
#define TOT        (1u << SCALE_BITS)
#define RANS_L16   (1u << 16)

static void pruef(cudaError_t e, int zeile) {
    if (e != cudaSuccess) { printf("CUDA-Fehler @ %d: %s\n", zeile, cudaGetErrorString(e)); exit(1); }
}
#define PRUEF(x) pruef((x), __LINE__)

__global__ void __launch_bounds__(TH)
kern_rans16_l1(const uint8_t  *__restrict__ daten,
               const uint32_t *__restrict__ basen,
               const uint16_t *__restrict__ subrel,
               const uint32_t *__restrict__ tab,
               uint8_t *__restrict__ res,
               uint64_t n_high, uint32_t block, uint32_t lanes, uint32_t n_bloecke)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= (uint64_t)n_bloecke * lanes) return;
    uint32_t b    = (uint32_t)(tid / lanes);
    uint32_t lane = (uint32_t)(tid % lanes);
    uint64_t basis = (uint64_t)b * block;
    uint64_t rest  = n_high - basis;
    uint32_t n_sym = (rest > block) ? block : (uint32_t)rest;
    if (lane >= n_sym) return;
    uint32_t n_lane = (n_sym - lane + lanes - 1) / lanes;

    const uint8_t *start = daten + basen[b] + subrel[(uint64_t)b * lanes + lane];
    uint32_t x = *(const uint32_t*)start;
    uint64_t w = *(const uint32_t*)(start + 4); int nb = 32;
    const uint2 *p = (const uint2*)(start + 8);

    uint8_t *out = res + basis + lane;
    #pragma unroll 4
    for (uint32_t i = 0; i < n_lane; i++) {
        uint32_t e = __ldg(tab + (x & (TOT - 1)));
        out[i * lanes] = (uint8_t)e;
        x = (((e >> 8) & 0xFFFu) + 1u) * (x >> SCALE_BITS) + (e >> 20);
        if (x < RANS_L16) {
            if (nb == 0) { uint2 v = *p++; w = (uint64_t)v.x | ((uint64_t)v.y << 32); nb = 64; }
            x = (x << 16) | (uint32_t)(w & 0xFFFFu); w >>= 16; nb -= 16;
        }
    }
}

__global__ void kern_verschraenken2(const uint16_t *__restrict__ low2, const uint16_t *__restrict__ high2,
                                    uint32_t *__restrict__ aus4, uint64_t n_paare) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t schritt = gridDim.x * (uint64_t)blockDim.x;
    for (; i < n_paare; i += schritt) {
        uint32_t l = low2[i], h = high2[i];
        aus4[i] = (l & 0xFFu) | ((h & 0xFFu) << 8) | ((l & 0xFF00u) << 8) | ((h & 0xFF00u) << 16);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { printf("nutzung: nwcdec_cuda5 <datei.nwc> <referenz.raw>\n"); return 2; }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror("open"); return 1; }
    fseek(fp, 0, SEEK_END); long fg = ftell(fp); fseek(fp, 0, SEEK_SET);
    uint8_t *h_buf = (uint8_t*)malloc(fg);
    if (fread(h_buf, 1, fg, fp) != (size_t)fg) { printf("read\n"); return 1; }
    fclose(fp);
    if (memcmp(h_buf, "NWC2", 4)) { printf("kein NWC2\n"); return 1; }

    uint32_t version = h_buf[4], lanes = h_buf[5], skala = h_buf[6], renorm = h_buf[7];
    uint64_t n_high; uint32_t stride, block, n_bloecke;
    memcpy(&n_high, h_buf + 8, 8); memcpy(&stride, h_buf + 16, 4);
    memcpy(&block,  h_buf + 20, 4); memcpy(&n_bloecke, h_buf + 24, 4);
    if (version != 3 || skala != SCALE_BITS || renorm != 16 || stride != 0) {
        printf("v5 braucht Formatversion 3 (nwcenc), renorm=16, stride=0\n"); return 1;
    }

    size_t off_freq  = 28;
    size_t off_basen = off_freq + 512;
    size_t off_sub   = off_basen + 4ull * n_bloecke;
    size_t off_daten = off_sub + 2ull * n_bloecke * lanes;
    size_t off_low   = (size_t)fg - (size_t)n_high;
    size_t daten_len = off_low - off_daten;

    const uint16_t *freq16 = (const uint16_t*)(h_buf + off_freq);
    uint32_t kum = 0; uint32_t *h_tab = (uint32_t*)malloc(TOT * 4);
    for (uint32_t s = 0; s < 256; s++) {
        uint32_t f = freq16[s];
        for (uint32_t j = 0; j < f; j++) h_tab[kum + j] = s | ((f - 1) << 8) | (j << 20);
        kum += f;
    }
    if (kum != TOT) { printf("Tabelle summiert auf %u statt %u\n", kum, TOT); return 1; }

    uint8_t *d_daten, *d_res, *d_low, *d_aus; uint32_t *d_basen, *d_tab; uint16_t *d_sub;
    PRUEF(cudaMalloc(&d_daten, daten_len + 16));
    PRUEF(cudaMalloc(&d_basen, 4ull * n_bloecke));
    PRUEF(cudaMalloc(&d_sub,   2ull * n_bloecke * lanes));
    PRUEF(cudaMalloc(&d_tab,   TOT * 4));
    PRUEF(cudaMalloc(&d_res,   n_high));
    PRUEF(cudaMalloc(&d_low,   n_high));
    PRUEF(cudaMalloc(&d_aus,   2 * n_high));
    PRUEF(cudaMemcpy(d_daten, h_buf + off_daten, daten_len, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_basen, h_buf + off_basen, 4ull * n_bloecke, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_sub,   h_buf + off_sub,   2ull * n_bloecke * lanes, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_tab,   h_tab, TOT * 4, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_low,   h_buf + off_low,   n_high, cudaMemcpyHostToDevice));

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    uint64_t n_threads = (uint64_t)n_bloecke * lanes;
    int bl_a = (int)((n_threads + TH - 1) / TH);
    int bl_c = p.multiProcessorCount * 32;
    printf("Datei          : %.1f MB, %u Bloecke x %u Lanes = %llu Threads, TH=%d -> %d CUDA-Bloecke\n",
           fg / 1e6, n_bloecke, lanes, (unsigned long long)n_threads, TH, bl_a);

    cudaEvent_t e[3];
    for (int i = 0; i < 3; i++) cudaEventCreate(&e[i]);
    float ms_a = 0, ms_c = 0, ms_ges = 1e30f;
    for (int w = 0; w < 10; w++) {
        cudaEventRecord(e[0]);
        kern_rans16_l1<<<bl_a, TH>>>(d_daten, d_basen, d_sub, d_tab, d_res, n_high, block, lanes, n_bloecke);
        cudaEventRecord(e[1]);
        kern_verschraenken2<<<bl_c, 256>>>((const uint16_t*)d_low, (const uint16_t*)d_res,
                                           (uint32_t*)d_aus, n_high / 2);
        cudaEventRecord(e[2]);
        PRUEF(cudaEventSynchronize(e[2]));
        if (w < 3) continue;
        float a, c, g;
        cudaEventElapsedTime(&a, e[0], e[1]); cudaEventElapsedTime(&c, e[1], e[2]);
        cudaEventElapsedTime(&g, e[0], e[2]);
        if (g < ms_ges) { ms_a = a; ms_c = c; ms_ges = g; }
    }
    PRUEF(cudaGetLastError());

    uint8_t *h_aus = (uint8_t*)malloc(2 * n_high);
    PRUEF(cudaMemcpy(h_aus, d_aus, 2 * n_high, cudaMemcpyDeviceToHost));
    int ok = 0;
    FILE *rf = fopen(argv[2], "rb");
    if (rf) {
        fseek(rf, 0, SEEK_END); long rg = ftell(rf); fseek(rf, 0, SEEK_SET);
        uint8_t *ref = (uint8_t*)malloc(rg);
        if (fread(ref, 1, rg, rf) == (size_t)rg)
            ok = (rg == (long)(2 * n_high)) && !memcmp(ref, h_aus, rg);
        fclose(rf);
    }
    double mb = 2.0 * n_high / 1e6;
    printf("Verifikation   : %s\n", ok ? "bitexakt OK" : "FEHLER");
    printf("  Stufe A rANS16 L1/TH%-3d : %7.3f ms   %8.1f GB/s Ausgabe\n", TH, ms_a, mb / ms_a);
    printf("  Stufe C verschr.        : %7.3f ms   %8.1f GB/s\n", ms_c, mb / ms_c);
    printf("  GESAMT                  : %7.3f ms   %8.1f GB/s\n", ms_ges, mb / ms_ges);
    return ok ? 0 : 1;
}
