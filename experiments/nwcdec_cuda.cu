/* NWC2-Dekoder auf der GPU.
   Stufe A: ein Thread je Teilstrom (Block x Lane), Tabellen im Shared Memory
   Stufe B: eine XOR-Kette je Restklasse mod stride
   Stufe C: Byte-Ebenen verschraenken                                        */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define SCALE_BITS 12
#define TOT        (1u << SCALE_BITS)
#define RANS_L     (1u << 23)

static void pruef(cudaError_t e, int zeile) {
    if (e != cudaSuccess) { printf("CUDA-Fehler @ %d: %s\n", zeile, cudaGetErrorString(e)); exit(1); }
}
#define PRUEF(x) pruef((x), __LINE__)

__global__ void kern_rans(const uint8_t *__restrict__ daten,
                          const uint32_t *__restrict__ basen,
                          const uint16_t *__restrict__ subrel,
                          const uint16_t *__restrict__ freq_g,
                          uint8_t *__restrict__ res,
                          uint64_t n_high, uint32_t block, uint32_t lanes, uint32_t n_bloecke)
{
    __shared__ uint8_t  s_slot2sym[TOT];
    __shared__ uint32_t s_freq[256];
    __shared__ uint32_t s_kum[257];

    if (threadIdx.x == 0) {
        uint32_t acc = 0;
        for (int i = 0; i < 256; i++) { s_kum[i] = acc; s_freq[i] = freq_g[i]; acc += freq_g[i]; }
        s_kum[256] = acc;
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < TOT; i += blockDim.x) {
        uint32_t lo = 0, hi = 256;
        while (lo + 1 < hi) { uint32_t m = (lo + hi) >> 1; if (s_kum[m] <= i) lo = m; else hi = m; }
        s_slot2sym[i] = (uint8_t)lo;
    }
    __syncthreads();

    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t gesamt = (uint64_t)n_bloecke * lanes;
    uint64_t schritt = gridDim.x * (uint64_t)blockDim.x;

    for (; tid < gesamt; tid += schritt) {
        uint32_t b    = (uint32_t)(tid / lanes);
        uint32_t lane = (uint32_t)(tid % lanes);
        uint64_t basis = (uint64_t)b * block;
        uint64_t n_sym = n_high - basis; if (n_sym > block) n_sym = block;
        if (lane >= n_sym) continue;
        uint64_t n_lane = (n_sym - lane + lanes - 1) / lanes;

        const uint8_t *p = daten + basen[b] + subrel[(uint64_t)b * lanes + lane];
        uint32_t x = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                     ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
        p += 4;

        uint8_t *out = res + basis + lane;
        for (uint64_t i = 0; i < n_lane; i++) {
            uint32_t slot = x & (TOT - 1);
            uint8_t s = s_slot2sym[slot];
            out[i * lanes] = s;                  /* benachbarte Lanes -> benachbarte Adressen */
            x = s_freq[s] * (x >> SCALE_BITS) + slot - s_kum[s];
            while (x < RANS_L) { x = (x << 8) | *p++; }
        }
    }
}

__global__ void kern_destride(uint8_t *__restrict__ high, uint64_t n, uint32_t stride) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= stride) return;
    uint8_t vor = high[r];
    for (uint64_t i = (uint64_t)r + stride; i < n; i += stride) { vor ^= high[i]; high[i] = vor; }
}

__global__ void kern_verschraenken(const uint8_t *__restrict__ low, const uint8_t *__restrict__ high,
                                   uint8_t *__restrict__ aus, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t schritt = gridDim.x * (uint64_t)blockDim.x;
    for (; i < n; i += schritt) { aus[2*i] = low[i]; aus[2*i+1] = high[i]; }
}

int main(int argc, char **argv) {
    if (argc < 3) { printf("nutzung: nwcdec_cuda <datei.nwc> <referenz.raw>\n"); return 2; }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror("open"); return 1; }
    fseek(fp, 0, SEEK_END); long fg = ftell(fp); fseek(fp, 0, SEEK_SET);
    uint8_t *h_buf = (uint8_t*)malloc(fg);
    if (fread(h_buf, 1, fg, fp) != (size_t)fg) { printf("read\n"); return 1; }
    fclose(fp);
    if (memcmp(h_buf, "NWC2", 4)) { printf("kein NWC2\n"); return 1; }

    uint32_t lanes = h_buf[5], skala = h_buf[6];
    uint64_t n_high; uint32_t stride, block, n_bloecke;
    memcpy(&n_high, h_buf + 8, 8); memcpy(&stride, h_buf + 16, 4);
    memcpy(&block,  h_buf + 20, 4); memcpy(&n_bloecke, h_buf + 24, 4);
    if (skala != SCALE_BITS) { printf("skala falsch\n"); return 1; }

    size_t off_freq  = 28;
    size_t off_basen = off_freq + 512;
    size_t off_sub   = off_basen + 4ull * n_bloecke;
    size_t off_daten = off_sub + 2ull * n_bloecke * lanes;
    size_t off_low   = (size_t)fg - (size_t)n_high;
    size_t daten_len = off_low - off_daten;

    printf("Datei          : %.1f MB, %u Bloecke x %u Lanes = %llu Threads\n",
           fg / 1e6, n_bloecke, lanes, (unsigned long long)n_bloecke * lanes);
    printf("stride         : %u\n", stride);

    uint8_t *d_daten, *d_res, *d_low, *d_aus; uint32_t *d_basen; uint16_t *d_sub, *d_freq;
    PRUEF(cudaMalloc(&d_daten, daten_len));
    PRUEF(cudaMalloc(&d_basen, 4ull * n_bloecke));
    PRUEF(cudaMalloc(&d_sub,   2ull * n_bloecke * lanes));
    PRUEF(cudaMalloc(&d_freq,  512));
    PRUEF(cudaMalloc(&d_res,   n_high));
    PRUEF(cudaMalloc(&d_low,   n_high));
    PRUEF(cudaMalloc(&d_aus,   2 * n_high));
    PRUEF(cudaMemcpy(d_daten, h_buf + off_daten, daten_len, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_basen, h_buf + off_basen, 4ull * n_bloecke, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_sub,   h_buf + off_sub,   2ull * n_bloecke * lanes, cudaMemcpyHostToDevice));
    PRUEF(cudaMemcpy(d_freq,  h_buf + off_freq,  512, cudaMemcpyHostToDevice));
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

    for (int w = 0; w < 6; w++) {
        cudaEventRecord(e[0]);
        kern_rans<<<bl_a, th>>>(d_daten, d_basen, d_sub, d_freq, d_res, n_high, block, lanes, n_bloecke);
        cudaEventRecord(e[1]);
        if (stride) kern_destride<<<bl_b, th>>>(d_res, n_high, stride);
        cudaEventRecord(e[2]);
        kern_verschraenken<<<bl_c, th>>>(d_low, d_res, d_aus, n_high);
        cudaEventRecord(e[3]);
        PRUEF(cudaEventSynchronize(e[3]));
        if (w < 2) continue;                     /* Aufwaermrunden verwerfen */
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
    printf("  Stufe A rANS     : %7.3f ms   %8.1f GB/s Ausgabe\n", ms_a, mb / ms_a);
    printf("  Stufe B destride : %7.3f ms   %8.1f GB/s\n", ms_b, ms_b > 0 ? mb / 2 / ms_b : 0.0);
    printf("  Stufe C verschr. : %7.3f ms   %8.1f GB/s\n", ms_c, mb / ms_c);
    printf("  ---------------------------------------------\n");
    printf("  GESAMT           : %7.3f ms   %8.1f GB/s\n", ms_ges, mb / ms_ges);
    return ok ? 0 : 1;
}
