/* gather_bench -- Durchsatz abhaengiger Tabellen-Gathers je SM-Takt (die Kernfrage des rANS-Dekoders).
   MODE 0: u32-Tabelle 16 KB (L1)   1: uint2-Tabelle 32 KB (L1)   2: u32-Tabelle im Shared Memory
        3: u16-Tabelle 8 KB (L1)    4: kein Load (nur ALU)         5: u32 4 KB (L1)
   NCH = unabhaengige Ketten je Thread; Adresse des naechsten Zugriffs haengt vom geladenen Wert ab.
   nvcc -O3 -arch=sm_86 -o build/gather_bench experimente/gather_bench.cu                             */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

template<int NCH, int MODE>
__global__ void __launch_bounds__(128, 8)
k(const uint32_t *__restrict__ tab, uint32_t *__restrict__ out, int iters, uint32_t maske) {
    __shared__ uint32_t stab[4096];
    if (MODE == 2) { for (int i = threadIdx.x; i < 4096; i += blockDim.x) stab[i] = tab[i]; __syncthreads(); }
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x[NCH];
    #pragma unroll
    for (int c = 0; c < NCH; c++) x[c] = (tid + 1) * 2654435761u + c * 0x9E3779B9u;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int c = 0; c < NCH; c++) {
            uint32_t e;
            if (MODE == 0)      e = __ldg(tab + (x[c] & maske));
            else if (MODE == 1) { uint2 v = __ldg((const uint2*)tab + (x[c] & maske)); e = v.x + v.y; }
            else if (MODE == 2) e = stab[x[c] & maske];
            else if (MODE == 3) e = __ldg((const uint16_t*)tab + (x[c] & maske));
            else if (MODE == 5) e = __ldg(tab + (x[c] & 1023));
            else                e = x[c] >> 12;
            x[c] = (x[c] ^ (x[c] >> 15)) * 0x2c1b3c6du + e; x[c] ^= x[c] >> 12;   /* Mischer: untere Bits echt zufaellig (LCG-Low-Bits waren es nicht) */
        }
    }
    uint32_t acc = 0;
    #pragma unroll
    for (int c = 0; c < NCH; c++) acc += x[c];
    out[tid] = acc;
}

template<int NCH, int MODE>
static void lauf(const char *name, const uint32_t *tab, uint32_t *out, int sms, double ghz, int bl_je_sm) {
    int iters = 4000, bl = sms * bl_je_sm;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    k<NCH, MODE><<<bl, 128>>>(tab, out, 100, 4095);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    k<NCH, MODE><<<bl, 128>>>(tab, out, iters, 4095);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    double g = (double)bl * 128 * NCH * iters;
    double je_sm_takt = g / (ms * 1e-3) / (sms * ghz * 1e9);
    printf("%-14s NCH=%d  %2d Bl/SM: %8.3f ms  %6.3f Gathers je SM-Takt  = %5.1f Takte je Warp-Gather\n",
           name, NCH, bl_je_sm, ms, je_sm_takt, 32.0 / je_sm_takt);
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    double ghz = clk * 1e-6;
    printf("%s: %d SMs, %.3f GHz (Attribut), L2 %d KB\n", p.name, p.multiProcessorCount, ghz, p.l2CacheSize >> 10);
    uint32_t *tab, *out; cudaMalloc(&tab, 32768); cudaMalloc(&out, 1 << 24);
    uint32_t h[8192]; for (int i = 0; i < 8192; i++) h[i] = i * 2246822519u; cudaMemcpy(tab, h, 32768, cudaMemcpyHostToDevice);
    int s = p.multiProcessorCount;
    #define ALLE(M, name) lauf<1,M>(name, tab, out, s, ghz, 8); lauf<2,M>(name, tab, out, s, ghz, 8); \
                          lauf<4,M>(name, tab, out, s, ghz, 8); lauf<8,M>(name, tab, out, s, ghz, 8); \
                          lauf<4,M>(name, tab, out, s, ghz, 4); lauf<4,M>(name, tab, out, s, ghz, 2);
    ALLE(4, "nur ALU")
    ALLE(0, "u32 16KB L1")
    ALLE(1, "uint2 32KB L1")
    ALLE(3, "u16 8KB L1")
    ALLE(5, "u32 4KB L1")
    ALLE(2, "u32 16KB smem")
    return 0;
}
