/* nwc_ops.dll -- NWC fuer PyTorch per ctypes.
   Blockgroesse (Symbole je Block, Vielfaches von 512) ist Laufzeitparameter: grosse Matrizen 8192
   (beste Rate), kleine 512..2048 (genug Threads fuer die GPU).
   Exporte:
     nwc_encode      Host. BF16 (n Elemente, n % block == 0) -> freq, basen, subrel, daten, low
     nwc_build_tab   Host. freq[256] -> gepackte 4096er-Tabelle
     nwc_matvec      Device, fp32-Pfad (Tests)
     nwc_linear_bf16 Device. y = W x + b, BF16 rein/raus, dekodiert in Registern, K % 16 == 0, K >= 512
     nwc_dequant     Device. komprimiert -> BF16[n]
     nwc_gather      Device. Zeilen ids[] einer [M x K]-Matrix -> BF16[n_ids x K] (Embedding-Lookup)
   Format: NWC2 Version 4 (Chunk-Layout), rANS 12/16, 32 Lanes, 8-Byte-Ausrichtung.                     */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef SCALE_BITS
#define SCALE_BITS 12
#endif
#define TOT        (1u << SCALE_BITS)
#define L16        (1u << 16)
#define LANES      32u
#define ALIGN      8u
#ifndef MINB
#define MINB 10                      /* min. Bloecke je SM -> Registerdeckel 51 (12 -> 40 spillt im bf16-Kernel) */
#endif
#ifndef TB
#define TB 128                       /* Threads je Thread-Block (Warps je CTA = TB/32) */
#endif
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

/* ============================ Host: Encoder ============================ */
static void normiere(const uint64_t *z, uint16_t *freq) {
    uint64_t gesamt = 0; for (int i = 0; i < 256; i++) gesamt += z[i];
    uint32_t f[256]; int idx[256], n_idx = 0;
    for (int i = 0; i < 256; i++) {
        f[i] = 0;
        if (z[i]) { uint64_t v = z[i] * TOT / gesamt; f[i] = v < 1 ? 1 : (uint32_t)v; idx[n_idx++] = i; }
    }
    for (int a = 0; a < n_idx; a++)
        for (int b = a + 1; b < n_idx; b++)
            if (f[idx[b]] > f[idx[a]]) { int t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
    long diff = (long)TOT; for (int i = 0; i < 256; i++) diff -= f[i];
    for (int k = 0; diff != 0; k++) {
        int i = idx[k % n_idx];
        if (diff > 0) { f[i]++; diff--; } else if (f[i] > 1) { f[i]--; diff++; }
    }
    for (int i = 0; i < 256; i++) freq[i] = (uint16_t)f[i];
}
static inline uint32_t pos_im_block(uint32_t lane, uint32_t i) { return (i >> 4) * 512 + lane * 16 + (i & 15); }

static size_t kodiere_lane(const uint8_t *sym, uint32_t n, const uint16_t *freq, const uint32_t *kum, uint8_t *ende) {
    uint8_t *q = ende; uint32_t x = L16;
    for (int64_t i = (int64_t)n - 1; i >= 0; i--) {
        uint32_t s = sym[i], f = freq[s], x_max = f << (32 - SCALE_BITS);
        while (x >= x_max) { q -= 2; q[0] = (uint8_t)x; q[1] = (uint8_t)(x >> 8); x >>= 16; }
        x = ((x / f) << SCALE_BITS) + (x % f) + kum[s];
    }
    q -= 4; q[0] = (uint8_t)x; q[1] = (uint8_t)(x >> 8); q[2] = (uint8_t)(x >> 16); q[3] = (uint8_t)(x >> 24);
    return (size_t)(ende - q);
}

EXPORT int64_t nwc_encode(const uint16_t *w, uint64_t n, uint32_t block, uint16_t *freq, uint32_t *basen,
                          uint8_t *sublen, uint8_t *daten, uint64_t daten_cap, uint8_t *low)
{
    if (block % 512 || n % block) return -1;
    uint8_t *high = (uint8_t*)malloc(n);
#ifdef CHUNKI
    /* Mantisse chunk-verschraenkt: Chunk c aller Bloecke liegt nebeneinander -> benachbarte Warps lesen
       benachbarte 512-B-Stuecke (DRAM-Row-Hits). Index: ((c * n_bloecke) + b) * 512 + off */
    { uint64_t nbl = n / block, nch = block / 512;
      for (uint64_t i = 0; i < n; i++) {
          uint64_t b = i / block, c = (i % block) / 512, off = i % 512;
          low[(c * nbl + b) * 512 + off] = (uint8_t)w[i]; high[i] = (uint8_t)(w[i] >> 8); } }
#else
    for (uint64_t i = 0; i < n; i++) { low[i] = (uint8_t)w[i]; high[i] = (uint8_t)(w[i] >> 8); }
#endif
    uint64_t z[256] = {0}; for (uint64_t i = 0; i < n; i++) z[high[i]]++;
    normiere(z, freq);
    uint32_t kum[257]; kum[0] = 0; for (int i = 0; i < 256; i++) kum[i + 1] = kum[i] + freq[i];

    uint32_t n_bloecke = (uint32_t)(n / block), n_lane = block / LANES;
    uint8_t *tmp = (uint8_t*)malloc(2ull * block + 64), *symtmp = (uint8_t*)malloc(block);
    uint64_t pos = 0;
#ifdef ROWS
    /* Zeilenverschraenkt: je Block [Wort 0 aller 32 Lanes][Wort 1 aller 32 Lanes]... (8-Byte-Woerter,
       Zeilen 256 B, kurze Lanes mit Nullen aufgefuellt). Ein Warp-Load trifft 256 B am Stueck. */
    uint8_t *lb = (uint8_t*)malloc(LANES * (2ull * n_lane + 16)); uint32_t wl[LANES];
    for (uint32_t b = 0; b < n_bloecke; b++) {
        uint64_t basis = (uint64_t)b * block; uint32_t maxw = 1;
        basen[b] = (uint32_t)pos;
        for (uint32_t lane = 0; lane < LANES; lane++) {
            for (uint32_t i = 0; i < n_lane; i++) symtmp[i] = high[basis + pos_im_block(lane, i)];
            size_t len = kodiere_lane(symtmp, n_lane, freq, kum, tmp + 2ull * block + 64);
            wl[lane] = (uint32_t)((len + 7) / 8); if (wl[lane] > maxw) maxw = wl[lane];
            uint8_t *d = lb + lane * (2ull * n_lane + 16);
            memset(d, 0, 8ull * wl[lane]); memcpy(d, tmp + 2ull * block + 64 - len, len);
            sublen[(uint64_t)b * LANES + lane] = (uint8_t)wl[lane];
        }
        if (pos + 8ull * maxw * LANES + 320 > daten_cap) { free(high); free(tmp); free(symtmp); free(lb); return -1; }
        for (uint32_t r = 0; r < maxw; r++)
            for (uint32_t lane = 0; lane < LANES; lane++) {
                uint8_t *dst = daten + pos + (8ull * r * LANES) + 8ull * lane;
                if (r < wl[lane]) memcpy(dst, lb + lane * (2ull * n_lane + 16) + 8ull * r, 8); else memset(dst, 0, 8);
            }
        pos += 8ull * maxw * LANES;
    }
    free(lb); free(high); free(tmp); free(symtmp);
    return (int64_t)(pos + 256 + 16);                   /* Reserve: Prefetch liest eine Zeile voraus */
#else
    /* Version 6: Lane-Stroeme ohne Padding hintereinander (2-Byte-Granularitaet); je Lane 1 Byte Laenge/2,
       Startoffset im Kernel per Warp-Scan. Bloecke 8-B-ausgerichtet. */
    for (uint32_t b = 0; b < n_bloecke; b++) {
        uint64_t basis = (uint64_t)b * block;
        pos = (pos + ALIGN - 1) / ALIGN * ALIGN;
        basen[b] = (uint32_t)pos;
        for (uint32_t lane = 0; lane < LANES; lane++) {
            for (uint32_t i = 0; i < n_lane; i++) symtmp[i] = high[basis + pos_im_block(lane, i)];
            size_t len = kodiere_lane(symtmp, n_lane, freq, kum, tmp + 2ull * block + 64);
            if (len / 2 > 255 || pos + len + 64 > daten_cap) { free(high); free(tmp); free(symtmp); return -1; }
            sublen[(uint64_t)b * LANES + lane] = (uint8_t)(len / 2);
            memcpy(daten + pos, tmp + 2ull * block + 64 - len, len);
            pos += len;
        }
    }
    free(high); free(tmp); free(symtmp);
    return (int64_t)((pos + ALIGN - 1) / ALIGN * ALIGN + 32);
#endif
}

EXPORT void nwc_build_tab(const uint16_t *freq, uint32_t *tab) {
    uint32_t kum = 0;
#ifdef TAB2
    uint8_t *t8 = (uint8_t*)tab;
    for (uint32_t s = 0; s < 256; s++) { uint32_t f = freq[s];
        for (uint32_t j = 0; j < f; j++) t8[kum + j] = (uint8_t)s;
        tab[TOT / 4 + s] = (kum << 16) | f; kum += f; }
#else
    for (uint32_t s = 0; s < 256; s++) { uint32_t f = freq[s];
        for (uint32_t j = 0; j < f; j++) tab[kum + j] = s | ((f - 1) << 8) | (j << 20); kum += f; }
#endif
}

/* ============================ Device: Kernel ============================ */
__device__ __forceinline__ float bf16_zu_f32(uint32_t b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ float warp_summe(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ uint32_t u4_byte(const uint4 &c, int j) {
    uint32_t w = (j < 4) ? c.x : (j < 8) ? c.y : (j < 12) ? c.z : c.w;
    return (w >> (8 * (j & 3))) & 0xFFu;
}
__device__ __forceinline__ float f4_el(const float4 &v, int j) {
    return (j == 0) ? v.x : (j == 1) ? v.y : (j == 2) ? v.z : v.w;
}

#if !defined(NOLDCS) && !defined(LDCG)
#define LDCS                         /* Standard: Mantisse evict-first laden (gemessen +3..5 %) */
#endif
#ifdef LDCG
#define LD_CHUNK(p) __ldcg(p)        /* Mantisse: nur L2, L1 komplett umgehen */
#elif defined(LDCS)
#define LD_CHUNK(p) __ldcs(p)        /* Mantisse: einmal gelesen, L1 nicht damit fuellen */
#else
#define LD_CHUNK(p) __ldg(p)
#endif
#ifdef CHUNKI
#define CHUNK_BASE   ((const uint4*)(low + (uint64_t)b * 512) + lane)
#define CHUNK_STRIDE (32u * (uint32_t)(n / block))     /* uint4-Einheiten zwischen Chunk c und c+1 */
#else
#define CHUNK_BASE   ((const uint4*)(low + basis) + lane)
#define CHUNK_STRIDE 32u
#endif
#ifdef SMQ
/* SM-lokale Arbeitsschlange: SM s bearbeitet CTA-Indizes [s*L, (s+1)*L) der Reihe nach -> kompakte Adressspanne
   je SM (TLB). Zaehler werden vom Finalize-Kernel zurueckgesetzt. */
__device__ unsigned int sm_zaehler[128];
__device__ __forceinline__ uint32_t smq_cta(uint32_t n_ctas) {
    uint32_t s, nsm; asm("mov.u32 %0, %%smid;" : "=r"(s)); asm("mov.u32 %0, %%nsmid;" : "=r"(nsm));
    __shared__ uint32_t cta_s;
    if (threadIdx.x == 0) {
        uint32_t L = (n_ctas + nsm - 1) / nsm, c = 0xFFFFFFFFu;
        for (uint32_t k = 0; k < nsm; k++) {
            uint32_t q = (s + k) % nsm, t = atomicAdd(&sm_zaehler[q], 1u), idx = q * L + t;
            if (t < L && idx < n_ctas) { c = idx; break; }
        }
        cta_s = c;
    }
    __syncthreads();
    return cta_s;
}
#endif
#ifdef LDCG_R
#define LD_W(p) __ldcg(p)            /* rANS-Woerter: nur L2 */
#else
#define LD_W(p) __ldg(p)
#endif
#ifdef SHTAB
#define TAB_LD(t, i) ((t)[i])        /* Tabelle im Shared Memory (bf16-Kernel) bzw. plain global */
#else
#define TAB_LD(t, i) __ldg((t) + (i))
#endif
#ifdef SECT
/* echter Load: holt den Sektor in L1/L2. ptxas streicht Loads mit totem Ergebnis, daher wird es in Lane.pf
   akkumuliert und am Kernel-Ende unter einer unbeweisbar falschen Bedingung verwendet. */
#define PF_L2(p) (pf ^= __ldg((const uint32_t*)(p)))
#elif defined(L2PF)
__device__ __forceinline__ void pf_l2(const void *p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }
#define PF_L2(p) pf_l2(p)
#else
#define PF_L2(p) ((void)0)
#endif
#ifdef TIMING
#define TIM_MAX 48000u
#define TIM_N   20u
__device__ unsigned long long tim[TIM_MAX * TIM_N];
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t smid_() { uint32_t s; asm("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
#define TIM_STAMP(b, k) do { if (lane == 0 && (b) < TIM_MAX && (k) < TIM_N) tim[(b) * TIM_N + (k)] = ((k) == 19 ? gtimer() : (unsigned long long)clock64()); } while (0)
EXPORT int nwc_tim_lesen(unsigned long long *dst) {
    return (int)cudaMemcpyFromSymbol(dst, tim, sizeof(unsigned long long) * TIM_MAX * TIM_N);
}
#else
#define TIM_STAMP(b, k) ((void)0)
#endif
/* Startoffset der Lane im Block: exklusive Praefixsumme der Lane-Laengen (u8 = Bytes/2) ueber den Warp */
__device__ __forceinline__ uint32_t lane_rel(const uint8_t *__restrict__ sublen, uint32_t b, uint32_t lane) {
    uint32_t v = __ldg(sublen + (uint64_t)b * 32 + lane), x = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) { uint32_t y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= (uint32_t)o) x += y; }
#ifdef ROWS
    return v;                                   /* ROWS: Wortzahl, kein Offset noetig */
#else
    return (x - v) * 2;
#endif
}
/* rANS-Zustand einer Lane */
#ifdef ROWS
#define PSTEP 32                     /* Woerter einer Lane liegen 32 Woerter (256 B) auseinander */
#else
#define PSTEP 1
#endif
struct Lane {
    uint32_t xs; uint64_t wb; int nb; const uint2 *p; uint2 nxt; uint32_t pf;
#ifdef PF2
    uint2 nxt2;
#endif
    __device__ __forceinline__ void init(const uint8_t *blockstart, uint32_t lane, uint32_t rel) {
        pf = 0;
#ifdef ROWS
        uint2 h = __ldg((const uint2*)blockstart + lane);
        xs = h.x; wb = h.y; nb = 32;
        p = (const uint2*)(blockstart + 8 * LANES) + lane;
        nxt = LD_W(p); p += PSTEP;
#else
        /* Start ist nur 2-B-ausgerichtet: 8-B-Wort davor laden und schieben (skip = 0,2,4,6 Bytes) */
        const uint8_t *start = blockstart + rel;
        const uint8_t *a8 = (const uint8_t*)((uintptr_t)start & ~(uintptr_t)7);
        uint32_t skip = (uint32_t)((uintptr_t)start & 7);
        uint2 q0 = __ldg((const uint2*)a8), q1 = __ldg((const uint2*)a8 + 1);
        uint64_t w0 = (uint64_t)q0.x | ((uint64_t)q0.y << 32), w1 = (uint64_t)q1.x | ((uint64_t)q1.y << 32);
        if (skip < 6) {
            xs = (uint32_t)(w0 >> (8 * skip));
            wb = (skip < 4) ? (w0 >> (8 * skip + 32)) : 0ull; nb = 32 - 8 * (int)skip;
            p = (const uint2*)a8 + 2; nxt = q1;
        } else {
            xs = (uint32_t)((w0 >> 48) | (w1 << 16));
            wb = w1 >> 16; nb = 48;
            p = (const uint2*)a8 + 3; nxt = LD_W((const uint2*)a8 + 2);
        }
#endif
        PF_L2((const uint8_t*)p + 32 * PSTEP); PF_L2((const uint8_t*)p + 64 * PSTEP); PF_L2((const uint8_t*)p + 96 * PSTEP);
#ifdef PF2
        nxt2 = LD_W(p); p += PSTEP;
#endif
    }
    __device__ __forceinline__ uint32_t symbol(const uint32_t *__restrict__ tab) {
#ifdef NODECODE
        nb -= 3;                                            /* ~24 Symbole je 8-Byte-Wort wie im Original */
        if (nb <= 0) { xs ^= nxt.x ^ nxt.y; nb = 64; nxt = LD_W(p); p += PSTEP; }
        xs += 0x9E3779B9u; return (xs >> 24);
#endif
#ifdef NOTAB
        /* Tabellenzugriff durch Arithmetik ersetzt (Timing-Bisektion): ~2,7 Bit je Symbol wie echt */
        uint32_t h = (xs & (TOT - 1)) * 2654435761u;
        uint32_t e = (h & 0xFFu) | ((511u + (h >> 24)) << 8) | (((h >> 8) & 0x1FFu) << 20);
        xs = (((e >> 8) & 0xFFFu) + 1u) * (xs >> SCALE_BITS) + (e >> 20);
#elif defined(TAB2)
        uint32_t e  = __ldg((const uint8_t*)tab + (xs & (TOT - 1)));       /* Symbol, 4-KB-Tabelle */
        uint32_t fk = __ldg(tab + TOT / 4 + e);                             /* kum<<16 | freq, 1 KB */
        xs = (fk & 0xFFFFu) * (xs >> SCALE_BITS) + (xs & (TOT - 1)) - (fk >> 16);
#else
        uint32_t e = TAB_LD(tab, xs & (TOT - 1));
        xs = (((e >> 8) & 0xFFFu) + 1u) * (xs >> SCALE_BITS) + (e >> 20);
#endif
        if (xs < L16) {
#ifdef PF2
            if (nb == 0) { wb = (uint64_t)nxt.x | ((uint64_t)nxt.y << 32); nb = 64; nxt = nxt2; nxt2 = LD_W(p); p += PSTEP; PF_L2((const uint8_t*)p + 32 * PSTEP); }
#else
            if (nb == 0) { wb = (uint64_t)nxt.x | ((uint64_t)nxt.y << 32); nb = 64; nxt = LD_W(p); p += PSTEP; PF_L2((const uint8_t*)p + 32 * PSTEP); }
#endif
            xs = (xs << 16) | (uint32_t)(wb & 0xFFFFu); wb >>= 16; nb -= 16;
        }
        return e & 0xFFu;
    }
};

__global__ void __launch_bounds__(128, 12)
kern_matvec(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
            const uint8_t *__restrict__ sublen, const uint32_t *__restrict__ tab,
            const uint8_t *__restrict__ low, const float *__restrict__ x, float *__restrict__ y,
            uint64_t n, uint32_t block, uint32_t K)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    if (basis >= n) return;
    Lane L; L.init(daten + basen[b], lane, lane_rel(sublen, b, lane));

    uint32_t m  = (uint32_t)(basis / K);
    uint32_t kk = (uint32_t)(basis % K) + lane * 16;
    const uint4 *lo4 = CHUNK_BASE; const uint32_t cstr = CHUNK_STRIDE;
    uint4 chunk_nxt = LD_CHUNK(lo4);
    float acc = 0.f;
    uint32_t n_chunks = block / 512;
    for (uint32_t c = 0; c < n_chunks; c++) {
        uint4 chunk = chunk_nxt;
        if (c + 1 < n_chunks) chunk_nxt = LD_CHUNK(lo4 + (uint64_t)cstr * (c + 1));
        const float4 *xp = (const float4*)(x + kk);
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            float4 xq;
            if ((j & 3) == 0) xq = __ldg(xp + (j >> 2));
            float w = bf16_zu_f32((L.symbol(tab) << 8) | u4_byte(chunk, j));
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

__global__ void __launch_bounds__(128)
kern_dequant(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
             const uint8_t *__restrict__ sublen, const uint32_t *__restrict__ tab,
             const uint8_t *__restrict__ low, uint16_t *__restrict__ out, uint64_t n, uint32_t block)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    if (basis >= n) return;
    Lane L; L.init(daten + basen[b], lane, lane_rel(sublen, b, lane));
    const uint4 *lo4 = CHUNK_BASE; const uint32_t cstr = CHUNK_STRIDE;
    uint4 *o4 = (uint4*)(out + basis) + 2 * lane;
    uint4 chunk_nxt = LD_CHUNK(lo4);
    uint32_t n_chunks = block / 512;
    for (uint32_t c = 0; c < n_chunks; c++) {
        uint4 chunk = chunk_nxt;
        if (c + 1 < n_chunks) chunk_nxt = LD_CHUNK(lo4 + (uint64_t)cstr * (c + 1));
        uint32_t w16[8];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint32_t v = (L.symbol(tab) << 8) | u4_byte(chunk, j);
            if (j & 1) w16[j >> 1] |= v << 16; else w16[j >> 1] = v;
        }
        o4[64 * c]     = make_uint4(w16[0], w16[1], w16[2], w16[3]);
        o4[64 * c + 1] = make_uint4(w16[4], w16[5], w16[6], w16[7]);
    }
}

/* Token-Pfad: x BF16 -> Partialsummen (fp32) -> y BF16 + Bias. Zeilen duerfen im Chunk wechseln. */
__global__ void __launch_bounds__(TB, MINB * 128 / TB)
kern_matvec_bf16(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
                 const uint8_t *__restrict__ sublen, const uint32_t *__restrict__ tab,
                 const uint8_t *__restrict__ low, const uint16_t *__restrict__ xb, float *__restrict__ teil,
                 uint64_t n, uint32_t block, uint32_t K, uint32_t P)
{
#ifdef SHTAB
    __shared__ uint32_t stab[TOT];
    for (uint32_t i = threadIdx.x; i < TOT; i += blockDim.x) stab[i] = tab[i];
    __syncthreads();
    const uint32_t *tabp = stab;
#else
    const uint32_t *tabp = tab;
#endif
#ifdef SMQ
    uint32_t cta = smq_cta(gridDim.x); if (cta == 0xFFFFFFFFu) return;
    uint32_t tid = cta * blockDim.x + threadIdx.x;
#else
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    if (basis >= n) return;
#ifdef TIMING
    if (lane == 0 && b < TIM_MAX) { tim[b * TIM_N] = (gtimer() & 0x00FFFFFFFFFFFFFFull) | ((unsigned long long)smid_() << 56);
                                    tim[b * TIM_N + 18] = (unsigned long long)clock64(); }
#endif
    Lane L; L.init(daten + basen[b], lane, lane_rel(sublen, b, lane));
    TIM_STAMP(b, 1);

    uint32_t m_cur = (uint32_t)(basis / K);
    uint64_t pl = basis + lane * 16;
    uint32_t m_l = (uint32_t)(pl / K), kk = (uint32_t)(pl - (uint64_t)m_l * K);
    const uint4 *lo4 = CHUNK_BASE; const uint32_t cstr = CHUNK_STRIDE;
    uint4 chunk_nxt = LD_CHUNK(lo4);
    float acc = 0.f;
    uint32_t n_chunks = block / 512;
#ifdef PF2C
    uint4 chunk_nxt2 = (n_chunks > 1) ? LD_CHUNK(lo4 + cstr) : chunk_nxt;
#endif

    for (uint32_t c = 0; c < n_chunks; c++) {
        TIM_STAMP(b, 2 + c);
        uint4 chunk = chunk_nxt;
#ifdef PF2C
        chunk_nxt = chunk_nxt2;
        if (c + 2 < n_chunks) chunk_nxt2 = LD_CHUNK(lo4 + (uint64_t)cstr * (c + 2));
#else
        if (c + 1 < n_chunks) chunk_nxt = LD_CHUNK(lo4 + (uint64_t)cstr * (c + 1));
#ifndef SECT
        if (c + 2 < n_chunks) PF_L2(lo4 + (uint64_t)cstr * (c + 2));
#endif
#endif
        const uint4 *xp = (const uint4*)(xb + kk);
        uint4 xa = __ldg(xp), xc = __ldg(xp + 1);
        uint32_t xw[8] = { xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w };
        float s_l = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            float w  = bf16_zu_f32((L.symbol(tabp) << 8) | u4_byte(chunk, j));
            float xv = (j & 1) ? __uint_as_float(xw[j >> 1] & 0xFFFF0000u) : __uint_as_float(xw[j >> 1] << 16);
            s_l = fmaf(w, xv, s_l);
        }
        bool gleich = (m_l == m_cur);
        if (__all_sync(0xffffffffu, gleich)) {
            acc += s_l;
        } else {
            float teil_cur = warp_summe(gleich ? acc + s_l : acc);
            if (lane == 0) teil[(uint64_t)m_cur * P + (b - (uint32_t)(((uint64_t)m_cur * K) / block))] = teil_cur;
            acc = gleich ? 0.f : s_l;
            m_cur++;
        }
        kk += 512;
        if (kk >= K) { kk -= K; m_l++; }
    }
    float s = warp_summe(acc);
    if (lane == 0) teil[(uint64_t)m_cur * P + (b - (uint32_t)(((uint64_t)m_cur * K) / block))] = s;
#ifdef SECT
    if (L.pf == 0x9E3779B9u && P == 0xFFFFFFFFu) teil[0] = 0.f;
#endif
    TIM_STAMP(b, 19);
}

/* Zeilen-Gather (Embedding-Lookup aus komprimierter [Zeilen x K]-Matrix): ein Warp je gesuchter Zeile dekodiert
   die Bloecke, die die Zeile schneiden, und schreibt nur die Elemente der Zeile. */
__global__ void __launch_bounds__(128)
kern_gather(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
            const uint8_t *__restrict__ sublen, const uint32_t *__restrict__ tab,
            const uint8_t *__restrict__ low, const int64_t *__restrict__ ids, uint32_t n_ids,
            uint16_t *__restrict__ out, uint64_t n, uint32_t block, uint32_t K)
{
    uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= n_ids) return;
    int64_t r = ids[warp];
    uint64_t g0 = (uint64_t)r * K, g1 = g0 + K;
    if (g1 > n) return;
    uint32_t b0 = (uint32_t)(g0 / block), b1 = (uint32_t)((g1 - 1) / block);
    uint16_t *o = out + (uint64_t)warp * K;
    uint32_t n_chunks = block / 512;
    for (uint32_t b = b0; b <= b1; b++) {
        uint64_t basis = (uint64_t)b * block;
        Lane L; L.init(daten + basen[b], lane, lane_rel(sublen, b, lane));
        const uint4 *lo4 = CHUNK_BASE; const uint32_t cstr = CHUNK_STRIDE;
        for (uint32_t c = 0; c < n_chunks; c++) {
            uint4 chunk = LD_CHUNK(lo4 + (uint64_t)cstr * c);
            uint64_t g = basis + (uint64_t)c * 512 + lane * 16;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                uint32_t v = (L.symbol(tab) << 8) | u4_byte(chunk, j);
                if (g + j >= g0 && g + j < g1) o[g + j - g0] = (uint16_t)v;
            }
        }
    }
}

__global__ void kern_f32_zu_bf16_bias(const float *__restrict__ teil, const uint16_t *__restrict__ bias,
                                      uint16_t *__restrict__ yb, uint32_t M, uint32_t K, uint32_t P, uint32_t block)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef SMQ
    if (i < 128) sm_zaehler[i] = 0;
#endif
    if (i >= M) return;
    uint64_t a = (uint64_t)i * K;
    uint32_t b0 = (uint32_t)(a / block), b1 = (uint32_t)((a + K - 1) / block);
    float v = 0.f;
    for (uint32_t s = 0; s <= b1 - b0; s++) v += teil[(uint64_t)i * P + s];
    if (bias) v += __uint_as_float((uint32_t)bias[i] << 16);
    uint32_t u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    yb[i] = (uint16_t)(u >> 16);
}

EXPORT int nwc_matvec(void *stream, const uint8_t *daten, const uint32_t *basen, const uint8_t *sublen,
                      const uint32_t *tab, const uint8_t *low, const float *x, float *y,
                      uint64_t n, uint32_t block, uint32_t K)
{
    if (K % 512 || block % 512 || n % block || n % K) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    uint32_t M = (uint32_t)(n / K);
    cudaMemsetAsync(y, 0, 4ull * M, st);
    int bl = (int)((n / block * 32 + 127) / 128);
    kern_matvec<<<bl, 128, 0, st>>>(daten, basen, sublen, tab, low, x, y, n, block, K);
    return (int)cudaGetLastError();
}

EXPORT int nwc_dequant(void *stream, const uint8_t *daten, const uint32_t *basen, const uint8_t *sublen,
                       const uint32_t *tab, const uint8_t *low, uint16_t *out, uint64_t n, uint32_t block)
{
    if (block % 512 || n % block) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    int bl = (int)((n / block * 32 + 127) / 128);
    kern_dequant<<<bl, 128, 0, st>>>(daten, basen, sublen, tab, low, out, n, block);
    return (int)cudaGetLastError();
}

EXPORT int nwc_linear_bf16(void *stream, const uint8_t *daten, const uint32_t *basen, const uint8_t *sublen,
                           const uint32_t *tab, const uint8_t *low, const uint16_t *x, const uint16_t *bias,
                           float *scratch, uint16_t *y, uint64_t n, uint32_t block, uint32_t K)
{
    if (K % 16 || K < 512 || block % 512 || n % block || n % K) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    uint32_t M = (uint32_t)(n / K), P = K / block + 2;
    int bl = (int)((n / block * 32 + TB - 1) / TB);
    kern_matvec_bf16<<<bl, TB, 0, st>>>(daten, basen, sublen, tab, low, x, scratch, n, block, K, P);
    kern_f32_zu_bf16_bias<<<(M + 255) / 256, 256, 0, st>>>(scratch, bias, y, M, K, P, block);
    return (int)cudaGetLastError();
}

EXPORT int nwc_gather(void *stream, const uint8_t *daten, const uint32_t *basen, const uint8_t *sublen,
                      const uint32_t *tab, const uint8_t *low, const int64_t *ids, uint32_t n_ids,
                      uint16_t *out, uint64_t n, uint32_t block, uint32_t K)
{
    if (K % 16 || block % 512 || n % block || n % K) return -1;
    if (n_ids == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    kern_gather<<<(n_ids * 32 + 127) / 128, 128, 0, st>>>(daten, basen, sublen, tab, low, ids, n_ids, out, n, block, K);
    return (int)cudaGetLastError();
}

EXPORT int nwc_setup(void) {
    cudaFuncSetAttribute(kern_matvec_bf16, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(kern_matvec, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    return 0;
}
