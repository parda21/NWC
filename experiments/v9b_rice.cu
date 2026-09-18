/* v9b_rice -- Praefixcode-Dekoder, Stufe 1+2: Blocklayout 8 Zeilen x 512 Spalten (x je Block in 16 Registern,
   8 Zeilenakkumulatoren, keine x-LDS, keine Zeilenwechsel-Logik) und INT-armes Nachladen:
   Fuellstand multiplikativ M = 2^(64-nb) (nb = gueltige Bits im 64-Bit-Fenster hi:lo, MSB zuerst, Invariante
   nb > 32 zu Schrittbeginn). Schritt um c Bits: m = 2^c aus der LUT, (hi:lo) *= m per IMAD, M64 = M*m per
   IMAD.WIDE -> M64.hi = 2^(32-nb') falls nb' <= 32 (Nachladen noetig), sonst 0. Nachladen unbedingt:
   (hi:lo) = w*M64.hi + (hi:lo) (eine IMAD.WIDE; w = vorgeladenes naechstes Wort), M = M64.hi + M64.lo;
   nur die LDG des uebernaechsten Worts ist praediziert (M64.hi != 0).
   LUT-Eintrag (12-Bit-Peek): 2 Codes:  by0 | by1<<8 | m<<16            (m = 2^c, c <= 12)
                              <2 Codes: FLAG | by0 | n<<8 | m1<<16       (n = 0/1, m1 = 2^c1 des 1. Codes)
   FLAG = Bit 30, Byte 3 hat damit MSB 0 -> Gewichts-PRMT liefert die Nullbytes per Vorzeichen-Replikation.
   Code je Gewicht wie v9_rice: Rang unaer (r Nullen, 1) + Vorzeichenbit; Rang >= 15: 15 Nullen, 1, Byte roh.
   nvcc -O3 -arch=sm_86 -o build/v9b_rice experimente/v9b_rice.cu [-DNOSLOW -DNOREFILL -DNOLDS -DSLOWCALL]
   ./build/v9b_rice data/W.raw M K          (K % 512 == 0, M % 8 == 0)                                        */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define PEEK   12
#define LUTN   (1u << PEEK)
#define ZEILEN 8u
#define SPALT  512u
#define BLOCK  (ZEILEN * SPALT)
#define LANES  32u
#define TBP    1024u
#define MAXR   15u
#define FLAG   (1u << 30)
#ifndef UNR
#define UNR 1              /* Ausrollfaktor der Zeilenschleife (1 = kleinster Code) */
#endif
#define PRAGMA_(x) _Pragma(#x)
#define PRAGMA(x) PRAGMA_(x)
#define UNROLL(n) PRAGMA(unroll n)
#ifndef PEEK_SHF
#define PEEKHI             /* Peek per IMAD.HI statt SHF+LOP3 (gemessen 0,2-0,3 Takte besser) */
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

/* ---------------- Host: Encoder ---------------- */
struct Bits {
    std::vector<uint32_t> w; uint64_t akk = 0; int n = 0; std::vector<uint8_t> laengen;
    void push(uint32_t v, int nb) { akk = (akk << nb) | v; n += nb; while (n >= 32) { w.push_back((uint32_t)(akk >> (n - 32))); n -= 32; } }
    void ende() { if (n) { w.push_back((uint32_t)(akk << (32 - n))); n = 0; } }
    uint32_t peek(uint64_t bitpos) const {                     /* 12 Bit ab bitpos (MSB zuerst), hinter dem Ende Nullen */
        uint64_t v = 0; for (int i = 0; i < 2; i++) { uint64_t wi = bitpos / 32 + i; v = (v << 32) | (wi < w.size() ? w[wi] : 0); }
        return (uint32_t)(v >> (64 - PEEK - bitpos % 32)) & (LUTN - 1);
    }
};

static uint32_t clz32(uint32_t x) { uint32_t u = 0; if (!x) return 32; while (!(x & 0x80000000u)) { x <<= 1; u++; } return u; }
static void lut_bauen(const uint8_t *exp_von_rang, uint32_t *lut) {
    for (uint32_t pat = 0; pat < LUTN; pat++) {
        uint32_t bits = pat << (32 - PEEK), pos = 0, n = 0, c1 = 0, by[2] = {0, 0};
        for (int k = 0; k < 2; k++) {
            uint32_t rest = PEEK - pos; if (rest == 0) break;
            uint32_t x = bits << pos, u = clz32(x);
            if (u >= MAXR || u + 2 > rest) break;
            uint32_t s = (x >> (30 - u)) & 1;
            by[k] = (s << 7) | exp_von_rang[u]; pos += u + 2; n++; if (k == 0) c1 = pos;
        }
        if (n == 2) lut[pat] = by[0] | (by[1] << 8) | ((1u << pos) << 16);
        else        lut[pat] = FLAG | by[0] | (n << 8) | ((n ? (1u << c1) : 0u) << 16);
    }
}

/* ---------------- Device ---------------- */
struct Fenster {
    uint64_t hl; uint32_t M, w, cnt; const uint32_t *basis;
    __device__ __forceinline__ void init(const uint32_t *start) {
        hl = ((uint64_t)__ldg(start) << 32) | __ldg(start + 1); w = __ldg(start + 2); cnt = 3; M = 1; basis = start;
    }
    __device__ __forceinline__ uint32_t hi() const { return (uint32_t)(hl >> 32); }
    /* Fenster um c Bits schieben (m = 2^c) und nachladen, alles ohne Praedikat ausser der LDG */
    __device__ __forceinline__ void schiebe(uint32_t m) {
        uint32_t mh, ml;
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(mh) : "r"(M), "r"(m));
        ml = M * m;
        hl = hl * m;
#ifdef NOREFILL   /* Diagnose: kein Nachladen (Muell) */
        M = ml + mh; hl ^= mh;
#else
        hl = (uint64_t)w * mh + hl;
        M = ml + mh;
        if (mh) { w = __ldg(basis + cnt); cnt++; }
#endif
    }
#ifdef PEEKHI
    __device__ __forceinline__ uint32_t peek4() const { return __umulhi(hi(), 1u << PEEK) * 4u; }      /* (hi >> 20) * 4 */
#else
    __device__ __forceinline__ uint32_t peek4() const { return (hi() >> (32 - PEEK)) << 2; }
#endif

    /* ein Code per clz (langsamer Pfad) */
    __device__ __forceinline__ uint32_t code_langsam(const uint8_t *evr) {
        uint32_t h = hi(), u = __clz(h);
        if (u < MAXR) { uint32_t s = (h >> (30 - u)) & 1; uint32_t b = (s << 7) | evr[u]; schiebe(1u << (u + 2)); return b; }
        schiebe(1u << 16); uint32_t b = hi() >> 24; schiebe(1u << 8); return b;
    }
};

/* PRMT mit Vorzeichen-Replikation (Selektor-Nibble 0xF = Byte 7 als Vorzeichenbyte -> 0x00, da Byte 7 MSB 0) */
__device__ __forceinline__ float gewicht(uint32_t cw, uint32_t e, uint32_t sel) {
    uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(cw), "r"(e), "r"(sel)); return __uint_as_float(r);
}

#ifdef SLOWCALL
__device__ __noinline__ uint4 langsam_fn(uint64_t hl, uint32_t M, uint32_t w, uint32_t cnt, const uint32_t *basis, uint32_t e, const uint8_t *evr) {
    Fenster F; F.hl = hl; F.M = M; F.w = w; F.cnt = cnt; F.basis = basis;
    uint32_t n = (e >> 8) & 1, b0 = e & 0xFFu, b1;
    if (n) F.schiebe((e >> 16) & 0x3FFFu);
    if (!n) b0 = F.code_langsam(evr);
    b1 = F.code_langsam(evr);
    return make_uint4(F.hi(), (uint32_t)F.hl, F.M, b0 | (b1 << 8) | (F.cnt << 16));
}
#endif

/* Kernel: MODE 0 = Matvec (Partialsummen teil[zeile*P + cb]), MODE 1 = Exponentbytes ausgeben */
template<int MODE>
__global__ void __launch_bounds__(TBP, 1)
kern(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen, const uint8_t *__restrict__ hdr,
     const uint32_t *__restrict__ lut, const uint8_t *__restrict__ low, const float *__restrict__ x,
     float *__restrict__ teil, uint8_t *__restrict__ out_hi, uint32_t K, uint32_t n_items,
     const uint8_t *__restrict__ exp_von_rang_g)
{
    extern __shared__ uint32_t smem[];                          /* [LUT][exp_von_rang 16 B] */
    for (uint32_t i = threadIdx.x; i < LUTN; i += TBP) smem[i] = __ldg(lut + i);
    uint8_t *evr = (uint8_t*)(smem + LUTN);
    if (threadIdx.x < 16) evr[threadIdx.x] = __ldg(exp_von_rang_g + threadIdx.x);
    __syncthreads();
    const uint32_t sm_lut = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t lane = threadIdx.x & 31, CB = K / SPALT;

    for (uint32_t b = blockIdx.x * (TBP / 32) + (threadIdx.x >> 5); b < n_items; b += gridDim.x * (TBP / 32)) {
        const uint32_t rb = b / CB, cb = b - rb * CB;
        uint32_t l4 = __ldg(hdr + (uint64_t)b * LANES + lane), xx = l4;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { uint32_t y = __shfl_up_sync(0xffffffffu, xx, o); if (lane >= (uint32_t)o) xx += y; }
        Fenster F; F.init((const uint32_t*)(daten + basen[b]) + (xx - l4));
        const uint64_t elem0 = (uint64_t)rb * ZEILEN * K + cb * SPALT + lane * 16;   /* Zeile 0 des Blocks, diese Lane */
        float xw[16];
        if (MODE == 0) {
            const float4 *xp = (const float4*)(x + cb * SPALT + lane * 16);
            #pragma unroll
            for (int i = 0; i < 4; i++) { float4 v = __ldg(xp + i); xw[4 * i] = v.x; xw[4 * i + 1] = v.y; xw[4 * i + 2] = v.z; xw[4 * i + 3] = v.w; }
        }
        float acc[ZEILEN];
        #pragma unroll
        for (int r = 0; r < (int)ZEILEN; r++) acc[r] = 0.f;
        const uint4 *lo4 = (const uint4*)(low + elem0);
        uint4 chunk_nxt = __ldcs(lo4);
        UNROLL(UNR)
        for (int r = 0; r < (int)ZEILEN; r++) {
            uint4 chunk = chunk_nxt;
            chunk_nxt = __ldcs((const uint4*)(low + elem0 + (uint64_t)(r + 1) * K));   /* letzte Zeile: ein Wort zu viel (harmlos, Puffer hat Reserve) */
            uint32_t w16[8]; float sr = 0.f;
            #pragma unroll
            for (int q = 0; q < 8; q++) {
                uint32_t e;
#ifdef NOLDS
                e = (F.peek4() * 2654435761u & 0xFFFFu) | (64u << 16);
#else
                asm("ld.shared.u32 %0, [%1];" : "=r"(e) : "r"(sm_lut + F.peek4()));
#endif
#ifdef NOSLOW
                F.schiebe(__byte_perm(e, 0u, 0x4432));
#else
#ifdef ANY
                bool langsam = e >= FLAG;
                if (!__any_sync(0xffffffffu, langsam)) F.schiebe(__byte_perm(e, 0u, 0x4432));
                else if (!langsam) F.schiebe(__byte_perm(e, 0u, 0x4432));
                else {
#else
                if (e < FLAG) F.schiebe(__byte_perm(e, 0u, 0x4432));
                else {
#endif
#ifdef SLOWCALL
                    uint4 R = langsam_fn(F.hl, F.M, F.w, F.cnt, F.basis, e, evr);
                    F.hl = ((uint64_t)R.x << 32) | R.y; F.M = R.z; F.cnt = R.w >> 16; e = R.w & 0xFFFFu;
                    F.w = __ldg(F.basis + F.cnt - 1);
#else
                    uint32_t n = (e >> 8) & 1, b0 = e & 0xFFu, b1;
                    if (n) F.schiebe((e >> 16) & 0x3FFFu);
                    if (!n) b0 = F.code_langsam(evr);
                    b1 = F.code_langsam(evr);
                    e = b0 | (b1 << 8);
#endif
                }
#endif
                const int jj = 2 * q; const uint32_t cw = (jj < 4) ? chunk.x : (jj < 8) ? chunk.y : (jj < 12) ? chunk.z : chunk.w;
                if (MODE == 0) {
                    sr = fmaf(gewicht(cw, e, (4 << 12) | ((jj & 3) << 8) | 0xFF), xw[jj], sr);
                    sr = fmaf(gewicht(cw, e, (5 << 12) | (((jj + 1) & 3) << 8) | 0xFF), xw[jj + 1], sr);
                } else w16[q] = e & 0xFFFFu;
            }
            if (MODE == 1) {
                uint4 o; o.x = w16[0] | (w16[1] << 16); o.y = w16[2] | (w16[3] << 16); o.z = w16[4] | (w16[5] << 16); o.w = w16[6] | (w16[7] << 16);
                *(uint4*)(out_hi + elem0 + (uint64_t)r * K) = o;
            } else {                                              /* Rotation: nach 8 Zeilen acc[i] = Zeile i */
                #pragma unroll
                for (int i = 0; i < (int)ZEILEN - 1; i++) acc[i] = acc[i + 1];
                acc[ZEILEN - 1] = sr;
            }
        }
        if (MODE == 0) {                                          /* transponierte Reduktion: Lane 4r erhaelt Zeile r */
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float s = (lane & 16) ? acc[i] : acc[i + 4], k = (lane & 16) ? acc[i + 4] : acc[i];
                acc[i] = k + __shfl_xor_sync(0xffffffffu, s, 16);
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                float s = (lane & 8) ? acc[i] : acc[i + 2], k = (lane & 8) ? acc[i + 2] : acc[i];
                acc[i] = k + __shfl_xor_sync(0xffffffffu, s, 8);
            }
            float s = (lane & 4) ? acc[0] : acc[1], k = (lane & 4) ? acc[1] : acc[0];
            float v = k + __shfl_xor_sync(0xffffffffu, s, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            if ((lane & 3) == 0) teil[(uint64_t)(rb * ZEILEN + (lane >> 2)) * CB + cb] = v;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) { printf("nutzung: v9b_rice W.raw M K\n"); return 1; }
    uint32_t M = atoi(argv[2]), K = atoi(argv[3]); uint64_t n = (uint64_t)M * K;
    if (M % ZEILEN || K % SPALT) { printf("M muss Vielfaches von 8 sein, K von 512\n"); return 1; }
    std::vector<uint16_t> w(n);
    FILE *f = fopen(argv[1], "rb"); if (!f || fread(w.data(), 2, n, f) != n) { printf("Datei\n"); return 1; } fclose(f);
    std::vector<uint8_t> low(n), high(n);
    for (uint64_t i = 0; i < n; i++) { low[i] = (uint8_t)w[i]; high[i] = (uint8_t)(w[i] >> 8); }
    std::vector<uint64_t> z(128, 0); for (uint64_t i = 0; i < n; i++) z[high[i] & 0x7F]++;
    std::vector<int> ord(128); for (int i = 0; i < 128; i++) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return z[a] > z[b]; });
    uint8_t rang_von_exp[128], exp_von_rang[16] = {0};
    for (int r = 0; r < 128; r++) { rang_von_exp[ord[r]] = (uint8_t)r; if (r < 16) exp_von_rang[r] = (uint8_t)ord[r]; }
    std::vector<uint32_t> lut(LUTN); lut_bauen(exp_von_rang, lut.data());
    uint64_t n2 = 0; for (uint32_t i = 0; i < LUTN; i++) n2 += lut[i] < FLAG;

    /* Kodieren: Block = 8 Zeilen x 512 Spalten, Lane L: Spalten 16L..16L+15 in jeder der 8 Zeilen */
    const uint32_t CB = K / SPALT, n_bl = (uint32_t)(n / BLOCK);
    std::vector<uint32_t> basen(n_bl); std::vector<uint8_t> hdr((uint64_t)n_bl * LANES);
    std::vector<uint8_t> daten; daten.reserve(n);
    uint64_t bits_ges = 0, esc = 0, langsam_lane = 0, langsam_warp = 0;
    std::vector<uint8_t> flags(LANES * 64);
    for (uint32_t b = 0; b < n_bl; b++) {
        while (daten.size() % 8) daten.push_back(0);
        basen[b] = (uint32_t)daten.size();
        uint32_t rb = b / CB, cb = b % CB;
        for (uint32_t L = 0; L < LANES; L++) {
            Bits bs;
            for (uint32_t i = 0; i < BLOCK / LANES; i++) {
                uint64_t pos = (uint64_t)(rb * ZEILEN + (i >> 4)) * K + cb * SPALT + L * 16 + (i & 15);
                uint8_t e = high[pos];
                uint32_t r = rang_von_exp[e & 0x7F], s = e >> 7;
                if (r < MAXR) { bs.push(1, r + 1); bs.push(s, 1); bits_ges += r + 2; bs.laengen.push_back((uint8_t)(r + 2)); }
                else { bs.push(1, 16); bs.push(e, 8); bits_ges += 24; esc++; bs.laengen.push_back(24); }
            }
            bs.ende();
            /* Statistik: wie oft liefert der 12-Bit-Peek keine 2 Codes? */
            uint64_t bp = 0;
            for (uint32_t q = 0; q < 64; q++) { flags[L * 64 + q] = lut[bs.peek(bp)] >= FLAG; langsam_lane += flags[L * 64 + q]; bp += bs.laengen[2 * q] + bs.laengen[2 * q + 1]; }
            if (bs.w.size() > 255) { printf("Strom zu lang\n"); return 1; }
            hdr[(uint64_t)b * LANES + L] = (uint8_t)bs.w.size();
            for (uint32_t v : bs.w) for (int k = 3; k >= 0; k--) daten.push_back((uint8_t)(v >> (8 * k)));
        }
        for (uint32_t q = 0; q < 64; q++) { int any = 0; for (uint32_t L = 0; L < LANES; L++) any |= flags[L * 64 + q]; langsam_warp += any; }
    }
    for (uint64_t i = 0; i + 3 < daten.size(); i += 4) { std::swap(daten[i], daten[i + 3]); std::swap(daten[i + 1], daten[i + 2]); }
    for (int i = 0; i < 64; i++) daten.push_back(0);
    double rate = ((double)daten.size() + (double)hdr.size() + 4.0 * n_bl + (double)n) / (2.0 * n);
    printf("M=%u K=%u  Exponent-Bits je Gewicht %.3f (Escape %.3f %%), Rate gesamt %.4f, LUT-Muster mit 2 Codes %.1f %%\n",
           M, K, (double)bits_ges / n, 100.0 * esc / n, rate, 100.0 * n2 / LUTN);
    printf("langsamer Pfad: je Lane-Schritt %.2f %%, je Warp-Schritt (irgendeine Lane) %.1f %%\n",
           100.0 * langsam_lane / (n / 2.0), 100.0 * langsam_warp / (n / 64.0));

    uint8_t *d_daten, *d_hdr, *d_low, *d_hi, *d_evr; uint32_t *d_basen, *d_lut; float *d_x, *d_teil;
    CK(cudaMalloc(&d_daten, daten.size())); CK(cudaMemcpy(d_daten, daten.data(), daten.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_hdr, hdr.size())); CK(cudaMemcpy(d_hdr, hdr.data(), hdr.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_basen, 4 * n_bl)); CK(cudaMemcpy(d_basen, basen.data(), 4 * n_bl, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_low, n + K + 64)); CK(cudaMemcpy(d_low, low.data(), n, cudaMemcpyHostToDevice));   /* Reserve: Vorauslesen der Mantissen ueber das Ende */
    CK(cudaMalloc(&d_lut, 4 * LUTN)); CK(cudaMemcpy(d_lut, lut.data(), 4 * LUTN, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_evr, 16)); CK(cudaMemcpy(d_evr, exp_von_rang, 16, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_hi, n)); CK(cudaMalloc(&d_x, 4 * K));
    CK(cudaMalloc(&d_teil, 4ull * M * CB)); CK(cudaMemset(d_teil, 0, 4ull * M * CB));
    std::vector<float> x(K); srand(1); for (uint32_t i = 0; i < K; i++) x[i] = (rand() / (float)RAND_MAX) - 0.5f;
    CK(cudaMemcpy(d_x, x.data(), 4 * K, cudaMemcpyHostToDevice));
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    size_t sm0 = 4 * LUTN + 16;
    CK(cudaFuncSetAttribute(kern<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm0));
    CK(cudaFuncSetAttribute(kern<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm0));
    int grid = std::min<int>(sms, (n_bl + 31) / 32);

    kern<1><<<grid, TBP, sm0>>>(d_daten, d_basen, d_hdr, d_lut, d_low, d_x, d_teil, d_hi, K, n_bl, d_evr);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> hi2(n); CK(cudaMemcpy(hi2.data(), d_hi, n, cudaMemcpyDeviceToHost));
    uint64_t falsch = 0; for (uint64_t i = 0; i < n; i++) falsch += hi2[i] != high[i];
    printf("Exponentbytes bitexakt: %s (%llu falsch)\n", falsch ? "NEIN" : "ja", (unsigned long long)falsch);

    cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
    for (int i = 0; i < 5; i++) kern<0><<<grid, TBP, sm0>>>(d_daten, d_basen, d_hdr, d_lut, d_low, d_x, d_teil, d_hi, K, n_bl, d_evr);
    CK(cudaDeviceSynchronize());
    std::vector<float> zeiten;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(a);
        kern<0><<<grid, TBP, sm0>>>(d_daten, d_basen, d_hdr, d_lut, d_low, d_x, d_teil, d_hi, K, n_bl, d_evr);
        cudaEventRecord(e); cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms, a, e); zeiten.push_back(ms);
    }
    std::sort(zeiten.begin(), zeiten.end()); float ms = zeiten[zeiten.size() / 2];
    double takte = ms * 1e-3 * sms * (clk * 1e3) / ((double)n / 64.0);
    printf("Matvec: %.3f ms  ->  %.0f GB/s komprimiert, %.0f GB/s BF16-aequivalent, %.1f SM-Takte je Warp-Schritt (32 Paare, %d SMs, %.3f GHz)\n",
           ms, rate * 2 * n / 1e6 / ms, 2 * n / 1e6 / ms, takte, sms, clk * 1e-6);
    std::vector<float> teil((size_t)M * CB); CK(cudaMemcpy(teil.data(), d_teil, 4ull * M * CB, cudaMemcpyDeviceToHost));
    double maxrel = 0;
    for (uint32_t m = 0; m < std::min<uint32_t>(M, 2048); m++) {
        double y = 0; for (uint32_t s = 0; s < CB; s++) y += teil[(size_t)m * CB + s];
        double r = 0; for (uint32_t k = 0; k < K; k++) { uint32_t u = (uint32_t)w[(uint64_t)m * K + k] << 16; float wf; memcpy(&wf, &u, 4); r += (double)wf * x[k]; }
        double d = fabs(y - r) / (fabs(r) + 1e-2); if (d > maxrel) maxrel = d;
    }
    printf("Matvec max. rel. Fehler (2048 Zeilen): %.2e %s\n", maxrel, maxrel < 1e-4 ? "OK" : "FEHLER");
    return 0;
}
