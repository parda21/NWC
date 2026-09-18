/* NWC2-Dekoder v2.
   - 16-Bit-Renormalisierung: hoechstens ein Nachladen pro Symbol -> kein Schleifenkopf
   - gepackte Lookup-Tabelle (sym | freq-1 << 8 | bias << 20): ein Shared-Zugriff pro Symbol
   - Tabelle einmal auf dem Host gebaut, je Threadblock nur kopiert
   - 32-Bit-Arithmetik in der inneren Schleife
   - Stufe C mit 16-Bit-Laden / 32-Bit-Schreiben                                          */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define SCALE_BITS 12
#define TOT        (1u << SCALE_BITS)
#define RANS_L16   (1u << 16)

static void pruef(cudaError_t e, int zeile) {
    if (e != cudaSuccess) { printf("CUDA-Fehler @ %d: %s\n", zeile, cudaGetErrorString(e)); exit(1); }
}
#define PRUEF(x) pruef((x), __LINE__)

__global__ void __launch_bounds__(256)
kern_rans16(const uint8_t  *__restrict__ daten,
            const uint32_t *__restrict__ basen,
            const uint16_t *__restrict__ subrel,
            const uint32_t *__restrict__ tab_g,
            uint8_t *__restrict__ res,
            uint64_t n_high, uint32_t block, uint32_t lanes, uint32_t n_bloecke)
{
    __shared__ uint32_t s_tab[TOT];
    for (uint32_t i = threadIdx.x; i < TOT; i += blockDim.x) s_tab[i] = tab_g[i];
    __syncthreads();

    uint64_t tid    = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t gesamt = (uint64_t)n_bloecke * lanes;
    uint64_t schritt = gridDim.x * (uint64_t)blockDim.x;

    for (; tid < gesamt; tid += schritt) {
        uint32_t b    = (uint32_t)(tid / lanes);
        uint32_t lane = (uint32_t)(tid % lanes);
        uint64_t basis = (uint64_t)b * block;
        uint64_t rest  = n_high - basis;
        uint32_t n_sym = (rest > block) ? block : (uint32_t)rest;
        if (lane >= n_sym) continue;
        uint32_t n_lane = (n_sym - lane + lanes - 1) / lanes;

        const uint8_t *p = daten + basen[b] + subrel[(uint64_t)b * lanes + lane];
        uint32_t x = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                     ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
        p += 4;

        uint8_t *out = res + basis + lane;
        #pragma unroll 4
        for (uint32_t i = 0; i < n_lane; i++) {
            uint32_t e = s_tab[x & (TOT - 1)];
            out[i * lanes] = (uint8_t)e;
            x = (((e >> 8) & 0xFFFu) + 1u) * (x >> SCALE_BITS) + (e >> 20);
            if (x < RANS_L16) { x = (x << 16) | ((uint32_t)p[0] << 8) | p[1]; p += 2; }
        }
    }
}

__global__ void kern_destride(uint8_t *__restrict__ high, uint64_t n, uint32_t stride) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= stride) return;
    uint8_t vor = high[r];
    for (uint64_t i = (uint64_t)r + stride; i < n; i += stride) { vor ^= high[i]; high[i] = vor; }
}

/* zwei Gewichte je Thread-Schritt: 2x16 Bit laden, 1x32 Bit schreiben */
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
    if (argc < 3) { printf("nutzung: nwcdec_cuda2 <datei.nwc> <referenz.raw>\n"); return 2; }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror("open"); return 1; }
    fseek(fp, 0, SEEK_END); long fg = ftell(fp); fseek(fp, 0, SEEK_SET);
    uint8_t *h_buf = (uint8_t*)malloc(fg);
    if (fread(h_buf, 1, fg, fp) != (size_t)fg) { printf("read\n"); return 1; }
    fclose(fp);
    if (memcmp(h_buf, "NWC2", 4)) { printf("kein NWC2\n"); return 1; }

    uint32_t lanes = h_buf[5], skala = h_buf[6], renorm = h_buf[7];
    uint64_t n_high; uint32_t stride, block, n_bloecke;
    memcpy(&n_high, h_buf + 8, 8); memcpy(&stride, h_buf + 16, 4);
    memcpy(&block,  h_buf + 20, 4); memcpy(&n_bloecke, h_buf + 24, 4);
    if (skala != SCALE_BITS) { printf("skala falsch\n"); return 1; }
    if (renorm != 16) { printf("v2 braucht renorm=16, Datei hat %u\n", renorm); return 1; }
    if (n_high & 1) { printf("n_high muss gerade sein\n"); return 1; }

    size_t off_freq  = 28;
    size_t off_basen = off_freq + 512;
    size_t off_sub   = off_basen + 4ull * n_bloecke;
    size_t off_daten = off_sub + 2ull * n_bloecke * lanes;
    size_t off_low   = (size_t)fg - (size_t)n_high;
    size_t daten_len = off_low - off_daten;

    /* gepackte Tabelle auf dem Host */
    const uint16_t *freq16 = (const uint16_t*)(h_buf + off_freq);
    uint32_t kum = 0; uint32_t *h_tab = (uint32_t*)malloc(TOT * 4);
    for (uint32_t s = 0; s < 256; s++) {
        uint32_t f = freq16[s];
        for (uint32_t j = 0; j < f; j++) h_tab[kum + j] = s | ((f - 1) << 8) | (j << 20);
        kum += f;
    }
    if (kum != TOT) { printf("Tabelle summiert auf %u statt %u\n", kum, TOT); return 1; }

    printf("Datei          : %.1f MB, %u Bloecke x %u Lanes = %llu Threads, renorm=%u\n",
           fg / 1e6, n_bloecke, lanes, (unsigned long long)n_bloecke * lanes, renorm);
    printf("stride         : %u\n", stride);

    uint8_t *d_daten, *d_res, *d_low, *d_aus; uint32_t *d_basen, *d_tab; uint16_t *d_sub;
    PRUEF(cudaMalloc(&d_daten, daten_len));
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
    int th = 256;
    long long bl_ll = (long long)((n_threads + th - 1) / th);
    int grenze = p.multiProcessorCount * 64;
    int bl_a = (bl_ll > grenze) ? grenze : (int)bl_ll;
    int bl_b = (int)((stride + th - 1) / th); if (bl_b < 1) bl_b = 1;
    int bl_c = p.multiProcessorCount * 32;

    cudaEvent_t e[4];
    for (int i = 0; i < 4; i++) cudaEventCreate(&e[i]);
    float ms_a = 0, ms_b = 0, ms_c = 0, ms_ges = 1e30f;

    for (int w = 0; w < 8; w++) {
        cudaEventRecord(e[0]);
        kern_rans16<<<bl_a, th>>>(d_daten, d_basen, d_sub, d_tab, d_res, n_high, block, lanes, n_bloecke);
        cudaEventRecord(e[1]);
        if (stride) kern_destride<<<bl_b, th>>>(d_res, n_high, stride);
        cudaEventRecord(e[2]);
        kern_verschraenken2<<<bl_c, th>>>((const uint16_t*)d_low, (const uint16_t*)d_res,
                                          (uint32_t*)d_aus, n_high / 2);
        cudaEventRecord(e[3]);
        PRUEF(cudaEventSynchronize(e[3]));
        if (w < 3) continue;
        float a, b, c, g;
        cudaEventElapsedTime(&a, e[0], e[1]); cudaEventElapsedTime(&b, e[1], e[2]);
        cudaEventElapsedTime(&c, e[2], e[3]); cudaEventElapsedTime(&g, e[0], e[3]);
        if (g < ms_ges) { ms_a = a; ms_b = b; ms_c = c; ms_ges = g; }
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
    printf("Verifikation   : %s\n\n", ok ? "bitexakt OK" : "FEHLER");
    printf("  Stufe A rANS16   : %7.3f ms   %8.1f GB/s Ausgabe\n", ms_a, mb / ms_a);
    printf("  Stufe B destride : %7.3f ms   %8.1f GB/s\n", ms_b, ms_b > 0 ? mb / 2 / ms_b : 0.0);
    printf("  Stufe C verschr. : %7.3f ms   %8.1f GB/s\n", ms_c, mb / ms_c);
    printf("  ---------------------------------------------\n");
    printf("  GESAMT           : %7.3f ms   %8.1f GB/s\n", ms_ges, mb / ms_ges);
    return ok ? 0 : 1;
}
