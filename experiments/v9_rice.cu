/* v9_rice -- Experiment: tabellenarmer Praefixcode statt rANS fuer die Exponentbytes.
   Code je Gewicht: Rang r des Exponents (7 Bit, nach Haeufigkeit sortiert) unaer (r Nullen, dann 1), dann das
   Vorzeichenbit roh; r >= 15: Escape = 15 Nullen, 1, dann das ganze Exponentbyte (8 Bit) roh.
   Dekoder: PEEK Bits vom Bitfenster als Index in eine LUT (Shared Memory), Eintrag = [Byte0 | Byte1 | c | n]:
   n = Zahl vollstaendiger Codes (0..2) in den PEEK Bits, c = verbrauchte Bits. Ein Nachschlag je Paar; wenn
   n < 2 (selten), dekodiert ein langsamer Pfad per clz nach. Fenster (hi:lo, MSB zuerst) wird per IMAD mit 2^c
   geschoben (FMA-Pipe), nb zaehlt gueltige Bits, Nachladen 32 Bit, wenn nb < 32.
   Block = 4096 Gewichte = 32 Lanes x 128; Lane L: 16 aufeinanderfolgende Gewichte je 512er-Chunk (wie v8).
   Layout: [32 Lane-Stroeme (4-B-Vielfache)], Header u8 je Lane = Laenge/4, basen[] je Block 8-B-ausgerichtet.
   nvcc -O3 -arch=sm_86 -o build/v9_rice experimente/v9_rice.cu [-DPEEK=14]
   ./build/v9_rice data/W.raw M K            (M*K*2 Bytes werden aus der Datei genommen)                     */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#ifndef PEEK
#define PEEK 12
#endif
#define LUTN   (1u << PEEK)
#define BLOCK  4096u
#define LANES  32u
#define TBP    1024u
#define MAXR   15u                       /* Raenge 0..14 unaer, ab 15 Escape */

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

/* ---------------- Host: Encoder ---------------- */
struct Bits {                                             /* MSB-first Bitschreiber in u32-Worte */
    std::vector<uint32_t> w; uint64_t akk = 0; int n = 0;
    void push(uint32_t v, int nb) { akk = (akk << nb) | v; n += nb; while (n >= 32) { w.push_back((uint32_t)(akk >> (n - 32))); n -= 32; } }
    void ende() { if (n) { w.push_back((uint32_t)(akk << (32 - n))); n = 0; } }   /* Fenster liest ueber das Ende in den naechsten Strom (harmlos); Puffer hat 64 B Reserve */
};
static inline uint32_t pos_im_block(uint32_t L, uint32_t i) { return (i >> 4) * 512 + L * 16 + (i & 15); }

static uint32_t clz32(uint32_t x) { uint32_t u = 0; if (!x) return 32; while (!(x & 0x80000000u)) { x <<= 1; u++; } return u; }
/* LUT: fuer jedes PEEK-Bit-Muster die ersten <= 2 vollstaendigen Codes */
static void lut_bauen(const uint8_t *exp_von_rang, uint32_t *lut) {
    for (uint32_t pat = 0; pat < LUTN; pat++) {
        uint32_t bits = pat << (32 - PEEK), pos = 0, n = 0, c = 0, by[2] = {0, 0};
        for (int k = 0; k < 2; k++) {
            uint32_t rest = PEEK - pos; if (rest == 0) break;
            uint32_t x = bits << pos;                              /* naechste Bits oben */
            uint32_t u = clz32(x);
            if (u >= MAXR || u + 2 > rest) break;                  /* Escape oder unvollstaendig */
            uint32_t s = (x >> (30 - u)) & 1;
            by[k] = (s << 7) | exp_von_rang[u]; pos += u + 2; n++; c = pos;
        }
        lut[pat] = by[0] | (by[1] << 8) | (c << 16) | (n << 24);
    }
}

/* ---------------- Device ---------------- */
struct Fenster {
    uint32_t hi, lo, nb; const uint32_t *p;
    __device__ __forceinline__ void init(const uint32_t *start) { hi = __ldg(start); lo = __ldg(start + 1); nb = 64; p = start + 2; }
    __device__ __forceinline__ void schiebe(uint32_t c) {           /* (hi:lo) <<= c per IMAD, nb -= c */
        uint32_t m = 1u << c;
        uint64_t t = (uint64_t)lo * m;
        lo = (uint32_t)t; hi = hi * m + (uint32_t)(t >> 32); nb -= c;
    }
    __device__ __forceinline__ void nachladen() {                   /* wenn nb < 32: 32 Bit anhaengen */
#ifdef NOREFILL   /* Diagnose: kein Nachladen (Fenster laeuft leer -> Muell) */
        if (nb < 32) { nb += 32; hi |= lo; lo = hi * 3u; }
#else
        if (nb < 32) { uint32_t w = __ldg(p); p++; hi |= w >> nb; lo = (nb == 0) ? w : (w << (32 - nb)); nb += 32; }
#endif
    }
    /* ein Code per clz (langsamer Pfad) */
    __device__ __forceinline__ uint32_t code_langsam(const uint8_t *exp_von_rang) {
        uint32_t u = __clz(hi);
        if (u < MAXR) { uint32_t s = (hi >> (30 - u)) & 1; uint32_t b = (s << 7) | exp_von_rang[u]; schiebe(u + 2); nachladen(); return b; }
        schiebe(16); nachladen(); uint32_t b = hi >> 24; schiebe(8); nachladen(); return b;
    }
};

/* Kernel: MODE 0 = Matvec (y = W x, fp32 Partialsummen je Block-Zeile), MODE 1 = Exponentbytes ausgeben (Bitexakt) */
template<int MODE>
__global__ void __launch_bounds__(TBP, 1)
kern(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen, const uint8_t *__restrict__ hdr,
     const uint32_t *__restrict__ lut, const uint8_t *__restrict__ low, const float *__restrict__ x,
     float *__restrict__ teil, uint8_t *__restrict__ out_hi, uint32_t K, uint32_t P, uint32_t n_items,
     const uint8_t *__restrict__ exp_von_rang_g)
{
    extern __shared__ uint32_t smem[];                          /* [LUT][x][exp_von_rang 16 B] */
    for (uint32_t i = threadIdx.x; i < LUTN; i += TBP) smem[i] = __ldg(lut + i);
    float *xs = (float*)(smem + LUTN);
    if (MODE == 0) for (uint32_t i = threadIdx.x; i < K; i += TBP) xs[i] = __ldg(x + i);
    uint8_t *evr = (uint8_t*)(xs + (MODE == 0 ? K : 0));
    if (threadIdx.x < 16) evr[threadIdx.x] = __ldg(exp_von_rang_g + threadIdx.x);
    __syncthreads();
    const uint32_t sm_lut = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t lane = threadIdx.x & 31;

    for (uint32_t b = blockIdx.x * (TBP / 32) + (threadIdx.x >> 5); b < n_items; b += gridDim.x * (TBP / 32)) {
        uint64_t basis = (uint64_t)b * BLOCK;
        uint32_t l4 = __ldg(hdr + (uint64_t)b * LANES + lane), xx = l4;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { uint32_t y = __shfl_up_sync(0xffffffffu, xx, o); if (lane >= (uint32_t)o) xx += y; }
        Fenster F; F.init((const uint32_t*)(daten + basen[b]) + (xx - l4));
        uint32_t m_cur = (uint32_t)(basis / K);
        uint64_t pl = basis + lane * 16;
        uint32_t m_l = (uint32_t)(pl / K), kk = (uint32_t)(pl - (uint64_t)m_l * K);
        const uint4 *lo4 = (const uint4*)(low + basis) + lane;
        uint4 chunk_nxt = __ldcs(lo4);
        float acc = 0.f;
        for (uint32_t c = 0; c < BLOCK / 512; c++) {
            uint4 chunk = chunk_nxt;
            if (c + 1 < BLOCK / 512) chunk_nxt = __ldcs(lo4 + 32 * (c + 1));
            const float4 *xp = (const float4*)(xs + kk);
            float4 x4[4];
            if (MODE == 0) { _Pragma("unroll") for (int i = 0; i < 4; i++) x4[i] = xp[i]; }
            const float *xw = (const float*)x4;
            float s0 = 0.f, s1 = 0.f;
            uint32_t w16[8];
            #pragma unroll
            for (int q = 0; q < 8; q++) {
                uint32_t e;
#ifdef NOLDS      /* Diagnose: LUT-Eintrag aus Arithmetik (Muell, aber n == 2 und c = 6) */
                e = ((F.hi >> (32 - PEEK)) * 2654435761u & 0xFFFFu) | (6u << 16) | (2u << 24);
#else
                asm("ld.shared.u32 %0, [%1];" : "=r"(e) : "r"(sm_lut + ((F.hi >> (32 - PEEK)) << 2)));
#endif
#ifdef NOSLOW     /* Diagnose: langsamer Pfad weg (falsch, wenn n < 2) */
                uint32_t n = 2;
#else
                uint32_t n = e >> 24;
#endif
                if (n == 2) { F.schiebe(__byte_perm(e, 0u, 0x4442)); }
                else {                                                   /* langsam: fehlende Codes per clz */
                    uint32_t b0, b1;
                    if (n == 1) { b0 = e & 0xFFu; F.schiebe(__byte_perm(e, 0u, 0x4442)); F.nachladen(); }
                    else b0 = F.code_langsam(evr);
                    b1 = F.code_langsam(evr);
                    e = b0 | (b1 << 8);
                }
                F.nachladen();
                e = __byte_perm(e, 0u, 0x4410);                       /* Bytes 2/3 (c, n) nullen: gewicht-PRMT nimmt sie als Mantissen-Nullbytes */
                const int jj = 2 * q; const uint32_t cw = (jj < 4) ? chunk.x : (jj < 8) ? chunk.y : (jj < 12) ? chunk.z : chunk.w;
                if (MODE == 0) {
                    float &sa = (q & 1) ? s1 : s0;
                    sa = fmaf(__uint_as_float(__byte_perm(cw, e, (4 << 12) | ((jj & 3) << 8) | 0x76)), xw[jj], sa);
                    sa = fmaf(__uint_as_float(__byte_perm(cw, e, (5 << 12) | (((jj + 1) & 3) << 8) | 0x76)), xw[jj + 1], sa);
                } else w16[q] = e & 0xFFFFu;
            }
            if (MODE == 1) {
                uint8_t *o = out_hi + basis + (uint64_t)c * 512 + lane * 16;
                #pragma unroll
                for (int q = 0; q < 8; q++) { o[2 * q] = (uint8_t)w16[q]; o[2 * q + 1] = (uint8_t)(w16[q] >> 8); }
            } else {
                float s_l = s0 + s1;
                bool gleich = (m_l == m_cur);
                if (__all_sync(0xffffffffu, gleich)) acc += s_l;
                else {
                    float v = gleich ? acc + s_l : acc;
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                    if (lane == 0) teil[(uint64_t)m_cur * P + (b - (uint32_t)(((uint64_t)m_cur * K) / BLOCK))] = v;
                    acc = gleich ? 0.f : s_l; m_cur++;
                }
                kk += 512; if (kk >= K) { kk -= K; m_l++; }
            }
        }
        if (MODE == 0) {
            float v = acc;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) teil[(uint64_t)m_cur * P + (b - (uint32_t)(((uint64_t)m_cur * K) / BLOCK))] = v;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) { printf("nutzung: v9_rice W.raw M K\n"); return 1; }
    uint32_t M = atoi(argv[2]), K = atoi(argv[3]); uint64_t n = (uint64_t)M * K;
    if (n % BLOCK || K % 16) { printf("M*K muss Vielfaches von 4096 sein, K von 16\n"); return 1; }
    std::vector<uint16_t> w(n);
    FILE *f = fopen(argv[1], "rb"); if (!f || fread(w.data(), 2, n, f) != n) { printf("Datei\n"); return 1; } fclose(f);
    std::vector<uint8_t> low(n), high(n);
    for (uint64_t i = 0; i < n; i++) { low[i] = (uint8_t)w[i]; high[i] = (uint8_t)(w[i] >> 8); }
    /* Raenge der Exponenten (7 Bit) nach Haeufigkeit */
    std::vector<uint64_t> z(128, 0); for (uint64_t i = 0; i < n; i++) z[high[i] & 0x7F]++;
    std::vector<int> ord(128); for (int i = 0; i < 128; i++) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return z[a] > z[b]; });
    uint8_t rang_von_exp[128], exp_von_rang[16] = {0};
    for (int r = 0; r < 128; r++) { rang_von_exp[ord[r]] = (uint8_t)r; if (r < 16) exp_von_rang[r] = (uint8_t)ord[r]; }
    /* Kodieren */
    uint32_t n_bl = (uint32_t)(n / BLOCK);
    std::vector<uint32_t> basen(n_bl); std::vector<uint8_t> hdr((uint64_t)n_bl * LANES);
    std::vector<uint8_t> daten; daten.reserve(n);
    uint64_t bits_ges = 0, esc = 0;
    for (uint32_t b = 0; b < n_bl; b++) {
        while (daten.size() % 8) daten.push_back(0);
        basen[b] = (uint32_t)daten.size();
        for (uint32_t L = 0; L < LANES; L++) {
            Bits bs;
            for (uint32_t i = 0; i < BLOCK / LANES; i++) {
                uint8_t e = high[(uint64_t)b * BLOCK + pos_im_block(L, i)];
                uint32_t r = rang_von_exp[e & 0x7F], s = e >> 7;
                if (r < MAXR) { bs.push(1, r + 1); bs.push(s, 1); bits_ges += r + 2; }
                else { bs.push(1, 16); bs.push(e, 8); bits_ges += 24; esc++; }
            }
            bs.ende();
            if (bs.w.size() > 255) { printf("Strom zu lang\n"); return 1; }
            hdr[(uint64_t)b * LANES + L] = (uint8_t)bs.w.size();
            for (uint32_t v : bs.w) for (int k = 3; k >= 0; k--) daten.push_back((uint8_t)(v >> (8 * k)));   /* big-endian Wort -> ld.u32 liest LE! */
        }
    }
    /* Achtung: der Dekoder laedt u32 little-endian; die Worte oben sind byteweise big-endian abgelegt -> Bytes drehen */
    for (uint64_t i = 0; i + 3 < daten.size(); i += 4) { std::swap(daten[i], daten[i + 3]); std::swap(daten[i + 1], daten[i + 2]); }
    for (int i = 0; i < 64; i++) daten.push_back(0);
    double rate = ((double)daten.size() + (double)hdr.size() + 4.0 * n_bl + (double)n) / (2.0 * n);
    printf("M=%u K=%u  Exponent-Bits je Gewicht %.3f (Escape %.3f %%), Rate gesamt %.4f  (PEEK %d, LUT %u KB)\n",
           M, K, (double)bits_ges / n, 100.0 * esc / n, rate, PEEK, LUTN * 4 / 1024);
    std::vector<uint32_t> lut(LUTN); lut_bauen(exp_von_rang, lut.data());
    uint64_t n2 = 0; for (uint32_t i = 0; i < LUTN; i++) n2 += (lut[i] >> 24) == 2;
    printf("LUT-Muster mit 2 vollstaendigen Codes: %.1f %%\n", 100.0 * n2 / LUTN);

    /* Device */
    uint8_t *d_daten, *d_hdr, *d_low, *d_hi, *d_evr; uint32_t *d_basen, *d_lut; float *d_x, *d_teil;
    CK(cudaMalloc(&d_daten, daten.size())); CK(cudaMemcpy(d_daten, daten.data(), daten.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_hdr, hdr.size())); CK(cudaMemcpy(d_hdr, hdr.data(), hdr.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_basen, 4 * n_bl)); CK(cudaMemcpy(d_basen, basen.data(), 4 * n_bl, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_low, n)); CK(cudaMemcpy(d_low, low.data(), n, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_lut, 4 * LUTN)); CK(cudaMemcpy(d_lut, lut.data(), 4 * LUTN, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_evr, 16)); CK(cudaMemcpy(d_evr, exp_von_rang, 16, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_hi, n)); CK(cudaMalloc(&d_x, 4 * K));
    uint32_t P = K / BLOCK + 2; CK(cudaMalloc(&d_teil, 4ull * M * P)); CK(cudaMemset(d_teil, 0, 4ull * M * P));
    std::vector<float> x(K); srand(1); for (uint32_t i = 0; i < K; i++) x[i] = (rand() / (float)RAND_MAX) - 0.5f;
    CK(cudaMemcpy(d_x, x.data(), 4 * K, cudaMemcpyHostToDevice));
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    size_t sm0 = 4 * LUTN + 16, sm1 = 4 * LUTN + 4 * K + 16;
    CK(cudaFuncSetAttribute(kern<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm0));
    CK(cudaFuncSetAttribute(kern<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm1));
    int grid = std::min<int>(sms, (n_bl + 31) / 32);

    /* Bitexakt */
    kern<1><<<grid, TBP, sm0>>>(d_daten, d_basen, d_hdr, d_lut, d_low, d_x, d_teil, d_hi, K, P, n_bl, d_evr);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> hi2(n); CK(cudaMemcpy(hi2.data(), d_hi, n, cudaMemcpyDeviceToHost));
    uint64_t falsch = 0; for (uint64_t i = 0; i < n; i++) falsch += hi2[i] != high[i];
    printf("Exponentbytes bitexakt: %s (%llu falsch)\n", falsch ? "NEIN" : "ja", (unsigned long long)falsch);

    /* Matvec: Zeit */
    cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
    for (int i = 0; i < 5; i++) kern<0><<<grid, TBP, sm1>>>(d_daten, d_basen, d_hdr, d_lut, d_low, d_x, d_teil, d_hi, K, P, n_bl, d_evr);
    CK(cudaDeviceSynchronize());
    std::vector<float> zeiten;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(a);
        kern<0><<<grid, TBP, sm1>>>(d_daten, d_basen, d_hdr, d_lut, d_low, d_x, d_teil, d_hi, K, P, n_bl, d_evr);
        cudaEventRecord(e); cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms, a, e); zeiten.push_back(ms);
    }
    std::sort(zeiten.begin(), zeiten.end()); float ms = zeiten[zeiten.size() / 2];
    double takte = ms * 1e-3 * sms * (clk * 1e3) / ((double)n / 64.0);
    printf("Matvec: %.3f ms  ->  %.0f GB/s komprimiert, %.0f GB/s BF16-aequivalent, %.1f SM-Takte je Warp-Schritt (32 Paare, %d SMs, %.3f GHz)\n",
           ms, rate * 2 * n / 1e6 / ms, 2 * n / 1e6 / ms, takte, sms, clk * 1e-6);
    /* Korrektheit Matvec: Partialsummen summieren, gegen double-Referenz (erste 2048 Zeilen) */
    std::vector<float> teil((size_t)M * P); CK(cudaMemcpy(teil.data(), d_teil, 4ull * M * P, cudaMemcpyDeviceToHost));
    double maxrel = 0;
    for (uint32_t m = 0; m < std::min<uint32_t>(M, 2048); m++) {
        uint64_t aa = (uint64_t)m * K; uint32_t b0 = (uint32_t)(aa / BLOCK), b1 = (uint32_t)((aa + K - 1) / BLOCK);
        double y = 0; for (uint32_t s = 0; s <= b1 - b0; s++) y += teil[(size_t)m * P + s];
        double r = 0; for (uint32_t k = 0; k < K; k++) { uint32_t u = (uint32_t)w[aa + k] << 16; float wf; memcpy(&wf, &u, 4); r += (double)wf * x[k]; }
        double d = fabs(y - r) / (fabs(r) + 1e-2); if (d > maxrel) maxrel = d;
    }
    printf("Matvec max. rel. Fehler (2048 Zeilen): %.2e %s\n", maxrel, maxrel < 1e-4 ? "OK" : "FEHLER");
    return 0;
}
