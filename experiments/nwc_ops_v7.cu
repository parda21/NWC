/* nwc_ops -- NWC fuer PyTorch per ctypes (Windows: nwc_ops.dll, Linux: nwc_ops.so).
   Format NWC2 Version 7:
     - BF16 in Byte-Ebenen: Mantisse roh (low[]), Exponentbyte rANS-kodiert (12 Skalenbits, 16-Bit-Renorm, LE)
     - PAAR-SYMBOLE: ein rANS-Schritt liefert zwei Exponentbytes (= zwei Gewichte). Tabelle 4096 Eintraege
       [Paarindex 10 | freq-1 11 | bias 11], Paartabelle 1024 x u16 (Index -> zwei Bytes); Index 0 = Escape,
       dann folgen die zwei Bytes roh als 16-Bit-Wort im selben Strom (seltene Paare, ~0,5 % der Masse).
     - Block = `block` Symbole (Vielfaches von 512) = 64 Stroeme = 32 Lanes x 2 Zustaende (A: Gewichte 0..7,
       B: 8..15 der 16 Gewichte einer Lane je 512er-Chunk). Zwei unabhaengige Ketten je Thread -> ILP.
     - Lane-Stroeme ohne Padding hintereinander (2-Byte-Granularitaet), je Strom 1 Byte Laenge/2,
       Startoffset per Warp-Praefixsumme; Bloecke 8-Byte-ausgerichtet (basen[]).
   Exporte:
     nwc_encode      Host. BF16 (n, n % block == 0) -> freq[1024], psym[1024], basen, sublen, daten, low
     nwc_build_tab   Host. freq/psym -> tab[4096 + 512] (Haupttabelle u32 + Paartabelle u16)
     nwc_matvec      Device, fp32-Pfad (Tests)
     nwc_linear_bf16 Device. y = W x + b, BF16 rein/raus, dekodiert in Registern, K % 16 == 0, K >= 512
     nwc_dequant     Device. komprimiert -> BF16[n]
     nwc_gather      Device. Zeilen ids[] einer [M x K]-Matrix -> BF16[n_ids x K] (Embedding-Lookup)
   Die v6-Quelle mit allen Diagnose-Schaltern liegt in experimente/nwc_ops_v6_diag.cu.                   */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define SCALE_BITS 12
#define TOT        (1u << SCALE_BITS)
#define L16        (1u << 16)
#define LANES      32u
#define STREAMS    64u
#define ALIGN      8u
#define NPAIR      1024u                 /* max. kodierte Paare (10-Bit-Index), 0 = Escape */
#define FMAX       2048u                 /* freq-1 in 11 Bit */
#define TAB_WORDS  (TOT + NPAIR / 2)     /* u32-Woerter: Haupttabelle + Paartabelle (u16) */
#ifndef MINB
#define MINB 9
#endif
#ifndef TB
#define TB 128
#endif
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

/* ============================ Host: Encoder ============================ */
static void normiere(const uint64_t *z, uint32_t n_sym, uint16_t *freq) {
    uint64_t gesamt = 0; for (uint32_t i = 0; i < n_sym; i++) gesamt += z[i];
    uint32_t *f = (uint32_t*)calloc(n_sym, 4); int *idx = (int*)malloc(n_sym * sizeof(int)); int n_idx = 0;
    for (uint32_t i = 0; i < n_sym; i++)
        if (z[i]) { uint64_t v = z[i] * TOT / gesamt; f[i] = v < 1 ? 1 : (v > FMAX ? FMAX : (uint32_t)v); idx[n_idx++] = i; }
    for (int a = 0; a < n_idx; a++)
        for (int b = a + 1; b < n_idx; b++)
            if (f[idx[b]] > f[idx[a]]) { int t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
    long diff = (long)TOT; for (uint32_t i = 0; i < n_sym; i++) diff -= f[i];
    for (int k = 0; diff != 0; k++) {
        int i = idx[k % n_idx];
        if (diff > 0 && f[i] < FMAX) { f[i]++; diff--; } else if (diff < 0 && f[i] > 1) { f[i]--; diff++; }
    }
    for (uint32_t i = 0; i < n_sym; i++) freq[i] = (uint16_t)f[i];
    free(f); free(idx);
}
/* Position des i-ten Gewichts von Strom s (= 2*lane + h) im Block */
static inline uint32_t pos_im_block(uint32_t s, uint32_t i) {
    return (i >> 3) * 512 + (s >> 1) * 16 + (s & 1) * 8 + (i & 7);
}

/* rANS rueckwaerts; Paare pr[] (16 Bit), map16[paar] = Index (0 = Escape: 16 Bit roh nach dem Symbol) */
static size_t kodiere_strom(const uint16_t *pr, uint32_t n, const uint16_t *map16, const uint16_t *freq,
                            const uint32_t *kum, uint8_t *ende) {
    uint8_t *q = ende; uint32_t x = L16;
    for (int64_t i = (int64_t)n - 1; i >= 0; i--) {
        uint32_t s = map16[pr[i]];
        if (s == 0) { q -= 2; q[0] = (uint8_t)pr[i]; q[1] = (uint8_t)(pr[i] >> 8); }   /* Dekoder liest sie nach dem Symbol */
        uint32_t f = freq[s], x_max = f << (32 - SCALE_BITS);
        while (x >= x_max) { q -= 2; q[0] = (uint8_t)x; q[1] = (uint8_t)(x >> 8); x >>= 16; }
        x = ((x / f) << SCALE_BITS) + (x % f) + kum[s];
    }
    q -= 4; q[0] = (uint8_t)x; q[1] = (uint8_t)(x >> 8); q[2] = (uint8_t)(x >> 16); q[3] = (uint8_t)(x >> 24);
    return (size_t)(ende - q);
}

EXPORT int64_t nwc_encode(const uint16_t *w, uint64_t n, uint32_t block, uint16_t *freq, uint16_t *psym,
                          uint32_t *basen, uint8_t *sublen, uint8_t *daten, uint64_t daten_cap, uint8_t *low)
{
    if (block % 512 || n % block) return -1;
    uint8_t *high = (uint8_t*)malloc(n);
    for (uint64_t i = 0; i < n; i++) { low[i] = (uint8_t)w[i]; high[i] = (uint8_t)(w[i] >> 8); }

    uint32_t n_bloecke = (uint32_t)(n / block), n_strom = block / STREAMS, n_paar = n_strom / 2;
    /* 1. Paar-Histogramm in Stromreihenfolge (Paare = benachbarte Gewichte 2k, 2k+1 eines Stroms) */
    uint64_t *z2 = (uint64_t*)calloc(65536, 8);
    uint16_t *pr = (uint16_t*)malloc(2ull * n_paar);
    for (uint32_t b = 0; b < n_bloecke; b++) {
        uint64_t basis = (uint64_t)b * block;
        for (uint32_t s = 0; s < STREAMS; s++)
            for (uint32_t k = 0; k < n_paar; k++)
                z2[high[basis + pos_im_block(s, 2 * k)] | (high[basis + pos_im_block(s, 2 * k + 1)] << 8)]++;
    }
    /* 2. Kodierte Paare: p >= 1/TOT, nach Haeufigkeit, max NPAIR-1; Rest -> Escape (Index 0) */
    uint64_t n_ges = (uint64_t)n_bloecke * STREAMS * n_paar;
    uint16_t *map16 = (uint16_t*)calloc(65536, 2);
    uint64_t *z = (uint64_t*)calloc(NPAIR, 8);
    uint32_t n_kod = 0;
    for (;;) {
        uint32_t best = 0; uint64_t bz = 0;
        for (uint32_t p = 0; p < 65536; p++) if (!map16[p] && z2[p] > bz) { bz = z2[p]; best = p; }
        if (bz == 0 || bz * TOT < n_ges || n_kod + 1 >= NPAIR) break;
        n_kod++; map16[best] = (uint16_t)n_kod; psym[n_kod] = (uint16_t)best; z[n_kod] = bz;
    }
    for (uint32_t p = 0; p < 65536; p++) if (!map16[p]) z[0] += z2[p];
    if (z[0] == 0) z[0] = 1;                                   /* Escape immer kodierbar */
    for (uint32_t i = n_kod + 1; i < NPAIR; i++) psym[i] = 0;
    psym[0] = 0;
    normiere(z, NPAIR, freq);
    uint32_t kum[NPAIR + 1]; kum[0] = 0; for (uint32_t i = 0; i < NPAIR; i++) kum[i + 1] = kum[i] + freq[i];

    /* 3. Stroeme kodieren, ohne Padding hintereinander */
    uint8_t *tmp = (uint8_t*)malloc(4ull * n_strom + 64);
    uint64_t pos = 0; int64_t rc = -1;
    for (uint32_t b = 0; b < n_bloecke; b++) {
        uint64_t basis = (uint64_t)b * block;
        pos = (pos + ALIGN - 1) / ALIGN * ALIGN;
        basen[b] = (uint32_t)pos;
        for (uint32_t s = 0; s < STREAMS; s++) {
            for (uint32_t k = 0; k < n_paar; k++)
                pr[k] = (uint16_t)(high[basis + pos_im_block(s, 2 * k)] | (high[basis + pos_im_block(s, 2 * k + 1)] << 8));
            size_t len = kodiere_strom(pr, n_paar, map16, freq, kum, tmp + 4ull * n_strom + 64);
            if (len / 2 > 255 || pos + len + 64 > daten_cap) goto ende;
            sublen[(uint64_t)b * STREAMS + s] = (uint8_t)(len / 2);
            memcpy(daten + pos, tmp + 4ull * n_strom + 64 - len, len);
            pos += len;
        }
    }
    rc = (int64_t)((pos + ALIGN - 1) / ALIGN * ALIGN + 32);
ende:
    free(tmp); free(z); free(map16); free(pr); free(z2); free(high);
    return rc;
}

EXPORT void nwc_build_tab(const uint16_t *freq, const uint16_t *psym, uint32_t *tab) {
    uint32_t kum = 0;
    for (uint32_t s = 0; s < NPAIR; s++) { uint32_t f = freq[s];
        for (uint32_t j = 0; j < f; j++) tab[kum + j] = (s << 22) | ((f - 1) << 11) | j; kum += f; }
    uint16_t *pt = (uint16_t*)(tab + TOT);
    for (uint32_t s = 0; s < NPAIR; s++) pt[s] = psym[s];
}

/* ============================ Device: Dekoder ============================ */
__device__ __forceinline__ float warp_summe(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ uint32_t u4_wort(const uint4 &c, int jj) {      /* Wort, das Byte jj enthaelt */
    return (jj < 4) ? c.x : (jj < 8) ? c.y : (jj < 12) ? c.z : c.w;
}
/* BF16-Gewicht als f32-Bitmuster: [Exponentbyte][Mantissenbyte][0][0]. s = Paar (u16, obere Bytes 0),
   Byte hb (0/1) von s; Mantisse = Byte (jj & 3) des Chunk-Worts. Ein PRMT. */
__device__ __forceinline__ float gewicht(uint32_t chunk_wort, uint32_t s, int jj, int hb) {
    return __uint_as_float(__byte_perm(chunk_wort, s, ((4 + hb) << 12) | ((jj & 3) << 8) | 0x76));
}
/* zwei BF16 als u32: [m0][s.b0][m1][s.b1] (fuer dequant/gather) */
__device__ __forceinline__ uint32_t zwei_bf16(uint32_t chunk_wort, uint32_t s, int jj) {
    return __byte_perm(chunk_wort, s, (5 << 12) | (((jj & 3) + 1) << 8) | (4 << 4) | (jj & 3));
}

/* Stromoffsets der beiden Zustaende einer Lane: exklusive Praefixsumme der 64 Laengen (u8 = Bytes/2) */
__device__ __forceinline__ void strom_rel(const uint8_t *__restrict__ sublen, uint32_t b, uint32_t lane,
                                          uint32_t &relA, uint32_t &relB) {
    uint32_t la = __ldg(sublen + (uint64_t)b * STREAMS + 2 * lane), lb = __ldg(sublen + (uint64_t)b * STREAMS + 2 * lane + 1);
    uint32_t v = la + lb, x = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) { uint32_t y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= (uint32_t)o) x += y; }
    relA = (x - v) * 2; relB = (x - v + la) * 2;
}

struct Z {
    uint32_t xs; uint64_t wb; int nb; const uint2 *p; uint2 nxt;
    __device__ __forceinline__ void init(const uint8_t *start) {          /* Start 2-B-ausgerichtet */
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
            p = (const uint2*)a8 + 3; nxt = __ldg((const uint2*)a8 + 2);
        }
    }
    __device__ __forceinline__ uint32_t bits16() {
        if (nb == 0) { wb = (uint64_t)nxt.x | ((uint64_t)nxt.y << 32); nb = 64; nxt = __ldg(p); p++; }
        uint32_t r = (uint32_t)(wb & 0xFFFFu); wb >>= 16; nb -= 16; return r;
    }
    /* ein Schritt = ein Paar (u16: Byte 0 = erstes, Byte 1 = zweites Exponentbyte) */
    __device__ __forceinline__ uint32_t schritt(const uint32_t *__restrict__ tab, const uint16_t *__restrict__ ptab) {
        uint32_t e = __ldg(tab + (xs & (TOT - 1)));
        xs = (((e >> 11) & 0x7FFu) + 1u) * (xs >> SCALE_BITS) + (e & 0x7FFu);
        if (xs < L16) xs = (xs << 16) | bits16();
        uint32_t idx = e >> 22;
        uint32_t s = __ldg(ptab + idx);
        if (idx == 0) s = bits16();
        return s;
    }
};

/* 16 Gewichte einer Lane in einem Chunk dekodieren: A liefert Paare fuer Bytes 0..7, B fuer 8..15.
   AKK(jj, w) wird je Gewicht aufgerufen (jj = Byteindex 0..15, w = f32-Gewicht bzw. Rohwort). */
#define DEKODIERE_CHUNK(A, B, chunk, tab, ptab, ...)                                              \
    _Pragma("unroll")                                                                            \
    for (int q_ = 0; q_ < 4; q_++) {                                                             \
        uint32_t pa_ = (A).schritt(tab, ptab), pb_ = (B).schritt(tab, ptab);                     \
        { const int jj = 2 * q_;     const uint32_t s_ = pa_; const uint32_t cw_ = u4_wort(chunk, jj); __VA_ARGS__ } \
        { const int jj = 2 * q_ + 8; const uint32_t s_ = pb_; const uint32_t cw_ = u4_wort(chunk, jj); __VA_ARGS__ } \
    }

/* ---- Token-Pfad: x BF16 -> Partialsummen (fp32) -> y BF16 + Bias. Zeilen duerfen im Chunk wechseln. ---- */
__global__ void __launch_bounds__(TB, MINB * 128 / TB)
kern_matvec_bf16(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
                 const uint8_t *__restrict__ sublen, const uint32_t *__restrict__ tab,
                 const uint8_t *__restrict__ low, const uint16_t *__restrict__ xb, float *__restrict__ teil,
                 uint64_t n, uint32_t block, uint32_t K, uint32_t P)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    if (basis >= n) return;
    const uint16_t *ptab = (const uint16_t*)(tab + TOT);
    uint32_t relA, relB; strom_rel(sublen, b, lane, relA, relB);
    Z A, B; A.init(daten + basen[b] + relA); B.init(daten + basen[b] + relB);

    uint32_t m_cur = (uint32_t)(basis / K);
    uint64_t pl = basis + lane * 16;
    uint32_t m_l = (uint32_t)(pl / K), kk = (uint32_t)(pl - (uint64_t)m_l * K);
    const uint4 *lo4 = (const uint4*)(low + basis) + lane;
    uint4 chunk_nxt = __ldcs(lo4);
    float acc = 0.f;
    uint32_t n_chunks = block / 512;

    for (uint32_t c = 0; c < n_chunks; c++) {
        uint4 chunk = chunk_nxt;
        if (c + 1 < n_chunks) chunk_nxt = __ldcs(lo4 + 32 * (c + 1));
        const uint4 *xp = (const uint4*)(xb + kk);
        uint4 xa = __ldg(xp), xc = __ldg(xp + 1);
        uint32_t xw[8] = { xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w };
        float sA = 0.f, sB = 0.f;
        DEKODIERE_CHUNK(A, B, chunk, tab, ptab, {
            float x0 = __uint_as_float(xw[jj >> 1] << 16), x1 = __uint_as_float(xw[jj >> 1] & 0xFFFF0000u);
            float &s = (jj < 8) ? sA : sB;
            s = fmaf(gewicht(cw_, s_, jj, 0), x0, s);
            s = fmaf(gewicht(cw_, s_, jj + 1, 1), x1, s);
        })
        float s_l = sA + sB;
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
}

__global__ void kern_f32_zu_bf16_bias(const float *__restrict__ teil, const uint16_t *__restrict__ bias,
                                      uint16_t *__restrict__ yb, uint32_t M, uint32_t K, uint32_t P, uint32_t block)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
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

/* ---- fp32-Pfad (Tests): y += W x per Atomics, K % 512 == 0 ---- */
__global__ void __launch_bounds__(128, MINB)
kern_matvec(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
            const uint8_t *__restrict__ sublen, const uint32_t *__restrict__ tab,
            const uint8_t *__restrict__ low, const float *__restrict__ x, float *__restrict__ y,
            uint64_t n, uint32_t block, uint32_t K)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    if (basis >= n) return;
    const uint16_t *ptab = (const uint16_t*)(tab + TOT);
    uint32_t relA, relB; strom_rel(sublen, b, lane, relA, relB);
    Z A, B; A.init(daten + basen[b] + relA); B.init(daten + basen[b] + relB);
    uint32_t m = (uint32_t)(basis / K), kk = (uint32_t)(basis % K) + lane * 16;
    const uint4 *lo4 = (const uint4*)(low + basis) + lane;
    float acc = 0.f;
    uint32_t n_chunks = block / 512;
    for (uint32_t c = 0; c < n_chunks; c++) {
        uint4 chunk = __ldcs(lo4 + 32 * c);
        const float *xp = x + kk;
        DEKODIERE_CHUNK(A, B, chunk, tab, ptab, {
            acc = fmaf(gewicht(cw_, s_, jj, 0), __ldg(xp + jj), acc);
            acc = fmaf(gewicht(cw_, s_, jj + 1, 1), __ldg(xp + jj + 1), acc);
        })
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

/* ---- Dequantisierung: komprimiert -> BF16[n] ---- */
__global__ void __launch_bounds__(128)
kern_dequant(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
             const uint8_t *__restrict__ sublen, const uint32_t *__restrict__ tab,
             const uint8_t *__restrict__ low, uint16_t *__restrict__ out, uint64_t n, uint32_t block)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    if (basis >= n) return;
    const uint16_t *ptab = (const uint16_t*)(tab + TOT);
    uint32_t relA, relB; strom_rel(sublen, b, lane, relA, relB);
    Z A, B; A.init(daten + basen[b] + relA); B.init(daten + basen[b] + relB);
    const uint4 *lo4 = (const uint4*)(low + basis) + lane;
    uint4 *o4 = (uint4*)(out + basis) + 2 * lane;
    uint32_t n_chunks = block / 512;
    for (uint32_t c = 0; c < n_chunks; c++) {
        uint4 chunk = __ldcs(lo4 + 32 * c);
        uint32_t w16[8];
        DEKODIERE_CHUNK(A, B, chunk, tab, ptab, { w16[jj >> 1] = zwei_bf16(cw_, s_, jj); })
        o4[64 * c]     = make_uint4(w16[0], w16[1], w16[2], w16[3]);
        o4[64 * c + 1] = make_uint4(w16[4], w16[5], w16[6], w16[7]);
    }
}

/* ---- Zeilen-Gather (Embedding-Lookup): ein Warp je gesuchter Zeile ---- */
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
    const uint16_t *ptab = (const uint16_t*)(tab + TOT);
    uint32_t b0 = (uint32_t)(g0 / block), b1 = (uint32_t)((g1 - 1) / block);
    uint16_t *o = out + (uint64_t)warp * K;
    uint32_t n_chunks = block / 512;
    for (uint32_t b = b0; b <= b1; b++) {
        uint64_t basis = (uint64_t)b * block;
        uint32_t relA, relB; strom_rel(sublen, b, lane, relA, relB);
        Z A, B; A.init(daten + basen[b] + relA); B.init(daten + basen[b] + relB);
        const uint4 *lo4 = (const uint4*)(low + basis) + lane;
        for (uint32_t c = 0; c < n_chunks; c++) {
            uint4 chunk = __ldg(lo4 + 32 * c);
            uint64_t g = basis + (uint64_t)c * 512 + lane * 16;
            DEKODIERE_CHUNK(A, B, chunk, tab, ptab, {
                uint32_t v2 = zwei_bf16(cw_, s_, jj);
                if (g + jj     >= g0 && g + jj     < g1) o[g + jj     - g0] = (uint16_t)v2;
                if (g + jj + 1 >= g0 && g + jj + 1 < g1) o[g + jj + 1 - g0] = (uint16_t)(v2 >> 16);
            })
        }
    }
}

/* ============================ Host: Starter ============================ */
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
