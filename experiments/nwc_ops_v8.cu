/* nwc_ops -- NWC fuer PyTorch per ctypes (Windows: nwc_ops.dll, Linux: nwc_ops.so).
   Format NWC2 Version 8:
     - BF16 in Byte-Ebenen: Mantisse roh (low[]), Exponentbyte rANS-kodiert (12 Skalenbits)
     - PAAR-SYMBOLE: ein rANS-Schritt liefert zwei Exponentbytes (= zwei Gewichte).
     - EINE 32-Bit-Tabelle mit 4096 Eintraegen: [Paar 16 | freq 8 | bias 8] (freq <= 255 kostet keine Rate,
       das haeufigste Paar hat ~1/16). Paar 0xFFFF = Escape-Sentinel (wird nie kodiert): die zwei Bytes stehen
       dann roh im ESCAPE-SEITENSTROM der Lane (eigene Warteschlange, damit Renorm und Escape nie im selben
       Schritt aus derselben Warteschlange lesen).
     - EIN 64-Bit-rANS-Zustand je Lane, Renormierung in 32-Bit-Worten (x in [2^32, 2^64)): halb so viele
       Stromladungen wie zwei 32-Bit-Zustaende, gleicher Flush (8 B je Lane), Stroeme 4-Byte-ausgerichtet.
     - Block = `block` Symbole (Vielfaches von 512) = 32 Lanes; Lane L traegt je 512er-Chunk die 16
       aufeinanderfolgenden Gewichte 16L..16L+15, als 8 Paare in Folge.
     - Blocklayout: [32 Lane-Stroeme (Vielfache von 4 B)][Escape-Worte aller Lanes (2 B, Lane-Reihenfolge)];
       Header je Lane 2 Bytes: Stromlaenge/4, Anzahl Escape-Worte; Offsets per Warp-Praefixsummen;
       Bloecke 8-Byte-ausgerichtet (basen[]).
     - Token-Pfad: persistente Bloecke (1024 Threads, einer je SM), Tabelle + x im Shared Memory, Dekodier-
       schritt mit wenigen INT-Pipe-Instruktionen (GA10x: IADD3/LOP3/SHF/ISETP laufen mit halber Rate).
   Exporte:
     nwc_info        Host. 0: Formatversion, 1: Zustaende je Lane (1), 2: Tabellenwoerter (u32), 3: Headerbytes je Block
     nwc_encode      Host. BF16 (n, n % block == 0) -> freq[1024], psym[1024], basen, hdr, daten, low
     nwc_build_tab   Host. freq/psym -> tab[4096 u32]
     nwc_matvec      Device, fp32-Pfad (Tests)
     nwc_linear_bf16 Device. y = W x + b, x fp32, y BF16, dekodiert in Registern, K % 16 == 0, K >= 512
     nwc_dequant     Device. komprimiert -> BF16[n]
     nwc_gather      Device. Zeilen ids[] einer [M x K]-Matrix -> BF16[n_ids x K] (Embedding-Lookup)
     nwc_setup       Host. SM-Zahl, Shared-Memory-Limit (wird beim Laden aufgerufen)
   v7 (zwei Tabellen, 64-Bit-Fenster) liegt in experimente/nwc_ops_v7.cu, v6 mit Diagnoseschaltern in
   experimente/nwc_ops_v6_diag.cu, die v8-Zwischenstufe mit allen Diagnoseschaltern (NOGATHER, NOLOAD, ...)
   in experimente/nwc_ops_v8_diag.cu.                                                                    */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define VERSION    8
#define SCALE_BITS 12
#define TOT        (1u << SCALE_BITS)
#define LANES      32u
#define HDR        2u                    /* Headerbytes je Lane: Laenge/4, Escape-Worte */
#define HDR_BLOCK  (LANES * HDR)
#define ALIGN      8u
#define NPAIR      1024u                 /* max. kodierte Paare, 0 = Escape */
#define FMAX       255u                  /* freq in 8 Bit */
#define TAB_WORDS  TOT
#define ESC_PAAR   0xFFFFu               /* Sentinel im Paarfeld */
#define TBP        1024u                 /* Threads je persistentem Block */
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

EXPORT int nwc_info(int was) {
    switch (was) { case 0: return VERSION; case 1: return 1; case 2: return (int)TAB_WORDS; case 3: return (int)HDR_BLOCK; }
    return -1;
}

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
/* Position des i-ten Gewichts von Lane L im Block */
static inline uint32_t pos_im_block(uint32_t L, uint32_t i) { return (i >> 4) * 512 + L * 16 + (i & 15); }

/* rANS rueckwaerts, 64-Bit-Zustand, 32-Bit-Renorm; Paare pr[] (16 Bit), map16[paar] = Index (0 = Escape,
   Rohbytes gehen in den Seitenstrom). Liefert Laenge (Vielfaches von 4). */
static size_t kodiere_strom(const uint16_t *pr, uint32_t n, const uint16_t *map16, const uint16_t *freq,
                            const uint32_t *kum, uint8_t *ende) {
    uint8_t *q = ende; uint64_t x = 1ull << 32;
    for (int64_t i = (int64_t)n - 1; i >= 0; i--) {
        uint32_t s = map16[pr[i]];
        uint64_t f = freq[s], x_max = f << (64 - SCALE_BITS);      /* x in [2^32, 2^64): emittieren ab f * 2^52 */
        while (x >= x_max) { q -= 4; uint32_t w = (uint32_t)x; memcpy(q, &w, 4); x >>= 32; }
        x = ((x / f) << SCALE_BITS) + (x % f) + kum[s];
    }
    q -= 8; memcpy(q, &x, 8);
    return (size_t)(ende - q);
}

EXPORT int64_t nwc_encode(const uint16_t *w, uint64_t n, uint32_t block, uint16_t *freq, uint16_t *psym,
                          uint32_t *basen, uint8_t *hdr, uint8_t *daten, uint64_t daten_cap, uint8_t *low)
{
    if (block % 512 || n % block) return -1;
    uint8_t *high = (uint8_t*)malloc(n);
    for (uint64_t i = 0; i < n; i++) { low[i] = (uint8_t)w[i]; high[i] = (uint8_t)(w[i] >> 8); }

    uint32_t n_bloecke = (uint32_t)(n / block), n_lane = block / LANES, n_paar = n_lane / 2;
    if (n_paar > 255) { free(high); return -1; }                    /* Escape-Zaehler ist u8 */
    /* 1. Paar-Histogramm in Stromreihenfolge (Paare = benachbarte Gewichte 2k, 2k+1 einer Lane) */
    uint64_t *z2 = (uint64_t*)calloc(65536, 8);
    uint16_t *pr = (uint16_t*)malloc(2ull * n_paar);
    for (uint32_t b = 0; b < n_bloecke; b++) {
        uint64_t basis = (uint64_t)b * block;
        for (uint32_t L = 0; L < LANES; L++)
            for (uint32_t k = 0; k < n_paar; k++)
                z2[high[basis + pos_im_block(L, 2 * k)] | (high[basis + pos_im_block(L, 2 * k + 1)] << 8)]++;
    }
    /* 2. Kodierte Paare: p >= 1/TOT, nach Haeufigkeit, max NPAIR-1, nie den Sentinel; Rest -> Escape (Index 0) */
    uint64_t n_ges = (uint64_t)n_bloecke * LANES * n_paar;
    uint16_t *map16 = (uint16_t*)calloc(65536, 2);
    uint64_t *z = (uint64_t*)calloc(NPAIR, 8);
    uint32_t n_kod = 0;
    for (;;) {
        uint32_t best = 0; uint64_t bz = 0;
        for (uint32_t p = 0; p < 65536; p++) if (p != ESC_PAAR && !map16[p] && z2[p] > bz) { bz = z2[p]; best = p; }
        if (bz == 0 || bz * TOT < n_ges || n_kod + 1 >= NPAIR) break;
        n_kod++; map16[best] = (uint16_t)n_kod; psym[n_kod] = (uint16_t)best; z[n_kod] = bz;
    }
    for (uint32_t p = 0; p < 65536; p++) if (!map16[p]) z[0] += z2[p];
    if (z[0] == 0) z[0] = 1;                                   /* Escape immer kodierbar */
    for (uint32_t i = n_kod + 1; i < NPAIR; i++) psym[i] = 0;
    psym[0] = (uint16_t)ESC_PAAR;
    normiere(z, NPAIR, freq);
    uint32_t kum[NPAIR + 1]; kum[0] = 0; for (uint32_t i = 0; i < NPAIR; i++) kum[i + 1] = kum[i] + freq[i];

    /* 3. Bloecke: [32 Lane-Stroeme][Escape-Worte aller Lanes] */
    uint8_t *tmp = (uint8_t*)malloc(8ull * n_paar + 64);
    uint16_t *esc = (uint16_t*)malloc(2ull * n_paar * LANES);
    uint64_t pos = 0; int64_t rc = -1;
    for (uint32_t b = 0; b < n_bloecke; b++) {
        uint64_t basis = (uint64_t)b * block;
        pos = (pos + ALIGN - 1) / ALIGN * ALIGN;
        basen[b] = (uint32_t)pos;
        uint32_t n_esc_ges = 0;
        for (uint32_t L = 0; L < LANES; L++) {
            for (uint32_t k = 0; k < n_paar; k++)
                pr[k] = (uint16_t)(high[basis + pos_im_block(L, 2 * k)] | (high[basis + pos_im_block(L, 2 * k + 1)] << 8));
            size_t len = kodiere_strom(pr, n_paar, map16, freq, kum, tmp + 8ull * n_paar + 64);
            if (len / 4 > 255 || pos + len + 64 > daten_cap) goto ende;
            memcpy(daten + pos, tmp + 8ull * n_paar + 64 - len, len);
            pos += len;
            uint32_t n_esc = 0;
            for (uint32_t k = 0; k < n_paar; k++) if (map16[pr[k]] == 0) esc[n_esc_ges + n_esc++] = pr[k];
            hdr[(uint64_t)b * HDR_BLOCK + L * HDR] = (uint8_t)(len / 4);
            hdr[(uint64_t)b * HDR_BLOCK + L * HDR + 1] = (uint8_t)n_esc;
            n_esc_ges += n_esc;
        }
        if (pos + 2ull * n_esc_ges + 64 > daten_cap) goto ende;
        memcpy(daten + pos, esc, 2ull * n_esc_ges); pos += 2ull * n_esc_ges;
    }
    rc = (int64_t)((pos + ALIGN - 1) / ALIGN * ALIGN + 32);
ende:
    free(esc); free(tmp); free(z); free(map16); free(pr); free(z2); free(high);
    return rc;
}

/* Tabelle: Slot j des Symbols s -> [Paar 16 | freq 8 | j 8]; Escape: Paar = 0xFFFF */
EXPORT void nwc_build_tab(const uint16_t *freq, const uint16_t *psym, uint32_t *tab) {
    uint32_t kum = 0;
    for (uint32_t s = 0; s < NPAIR; s++) { uint32_t f = freq[s], pp = s ? psym[s] : ESC_PAAR;
        for (uint32_t j = 0; j < f; j++) tab[kum + j] = (pp << 16) | (f << 8) | j;
        kum += f; }
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
/* BF16-Gewicht als f32-Bitmuster: [Exponentbyte][Mantissenbyte][0][0]. s = Paar in Bytes 0/1 (Bytes 2/3 = 0),
   Byte hb (0/1) von s; Mantisse = Byte (jj & 3) des Chunk-Worts. Ein PRMT. */
__device__ __forceinline__ float gewicht(uint32_t chunk_wort, uint32_t s, int jj, int hb) {
    return __uint_as_float(__byte_perm(chunk_wort, s, ((4 + hb) << 12) | ((jj & 3) << 8) | 0x76));
}
/* zwei BF16 als u32: [m0][s.b0][m1][s.b1] (fuer dequant/gather) */
__device__ __forceinline__ uint32_t zwei_bf16(uint32_t chunk_wort, uint32_t s, int jj) {
    return __byte_perm(chunk_wort, s, (5 << 12) | (((jj & 3) + 1) << 8) | (4 << 4) | (jj & 3));
}
__device__ __forceinline__ uint32_t warp_scan_excl(uint32_t v, uint32_t lane, uint32_t &summe) {
    uint32_t x = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) { uint32_t y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= (uint32_t)o) x += y; }
    summe = __shfl_sync(0xffffffffu, x, 31);
    return x - v;
}

/* Escape-Seitenstrom einer Lane: ein vorgeladenes Wort */
struct Esc {
    uint32_t ea; const uint8_t *pe;
    __device__ __forceinline__ void init(const uint8_t *start) { ea = __ldg((const uint16_t*)start); pe = start + 2; }
};

/* rANS-Zustand einer Lane: x = hi:lo in [2^32, 2^64), a = naechstes 32-Bit-Wort, p = Leseposition */
struct Z {
    uint32_t lo, hi, a; const uint8_t *p;
    __device__ __forceinline__ void init(const uint8_t *start) {          /* Start 4-B-ausgerichtet */
        const uint32_t *q = (const uint32_t*)start;
        lo = __ldg(q); hi = __ldg(q + 1); a = __ldg(q + 2); p = (const uint8_t*)(q + 3);
    }
    /* Lane-Region und Zustand aus dem Header aufsetzen */
    __device__ __forceinline__ void aufsetzen(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
                                              const uint8_t *__restrict__ hdr, uint32_t b, uint32_t lane, Esc &E) {
        const uint8_t *h = hdr + (uint64_t)b * HDR_BLOCK + lane * HDR;
        uint32_t l4 = __ldg(h), ne = __ldg(h + 1), sum_l4, sum_ne;
        uint32_t off_s = warp_scan_excl(l4, lane, sum_l4) * 4;
        uint32_t off_e = sum_l4 * 4 + warp_scan_excl(ne, lane, sum_ne) * 2;
        const uint8_t *bs = daten + basen[b];
        init(bs + off_s); E.init(bs + off_e);
    }
    /* ein Schritt = ein Paar in Bytes 0/1 (Bytes 2/3 = 0). e = Tabelleneintrag (vom Aufrufer geladen) */
    __device__ __forceinline__ uint32_t schritt_mit(uint32_t e, uint32_t q_lo, uint32_t q_hi, Esc &E) {
        uint32_t f = __byte_perm(e, 0u, 0x4441), bias = __byte_perm(e, 0u, 0x4440);
        uint64_t t;
        asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(t) : "r"(f), "r"(q_lo), "l"((uint64_t)bias));   /* f*q_lo + bias */
        lo = (uint32_t)t; hi = (uint32_t)(t >> 32) + f * q_hi;
        /* Renorm, praediziert: x = (x << 32) | a; a = naechstes Wort; p += 4 */
        asm("{\n\t.reg .pred q;\n\t"
            "setp.eq.u32 q, %1, 0;\n\t"
            "@q mov.b32 %1, %0;\n\t"
            "@q mov.b32 %0, %2;\n\t"
            "@q ld.global.nc.u32 %2, [%3];\n\t"
            "@q add.s64 %3, %3, 4;\n\t}"
            : "+r"(lo), "+r"(hi), "+r"(a), "+l"(p));
        uint32_t s = __byte_perm(e, 0u, 0x4432);                               /* Paar (Bytes 2/3 -> 0/1) */
#ifdef ESC_ANY  /* warp-uniforme Verzweigung (A16: gleich schnell wie praediziert, daher aus) */
        if (__any_sync(0xffffffffu, s == ESC_PAAR)) {
            if (s == ESC_PAAR) { s = E.ea; E.ea = __ldg((const uint16_t*)E.pe); E.pe += 2; }
        }
#else
        if (s == ESC_PAAR) { s = E.ea; E.ea = __ldg((const uint16_t*)E.pe); E.pe += 2; }
#endif
        return s;
    }
    __device__ __forceinline__ void quotient(uint32_t &q_lo, uint32_t &q_hi) const {           /* q = x >> 12 */
#ifdef QHI      /* auf der FMA-Pipe: lo>>12 per mul.hi, hi<<20 per IMAD dazu (Bits ueberlappen nicht) */
        q_lo = __umulhi(lo, 1u << 20) + hi * (1u << 20); q_hi = __umulhi(hi, 1u << 20);
#else
        q_lo = __funnelshift_r(lo, hi, SCALE_BITS); q_hi = hi >> SCALE_BITS;
#endif
    }
    __device__ __forceinline__ uint32_t schritt_g(const uint32_t *__restrict__ tab, Esc &E) {   /* Tabelle global (L1) */
        uint32_t q_lo, q_hi; quotient(q_lo, q_hi);
        return schritt_mit(__ldg(tab + (lo & (TOT - 1))), q_lo, q_hi, E);
    }
    __device__ __forceinline__ uint32_t schritt_s(uint32_t sm_tab, Esc &E) {                    /* Tabelle im Shared Memory */
        uint32_t q_lo, q_hi; quotient(q_lo, q_hi);
        uint32_t adr, e;
        asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(adr) : "r"(q_lo), "r"(0xFFFFC000u), "r"(sm_tab + lo * 4));  /* (lo & 4095)*4 + base */
        asm("ld.shared.u32 %0, [%1];" : "=r"(e) : "r"(adr));
        return schritt_mit(e, q_lo, q_hi, E);
    }
};

/* 16 Gewichte einer Lane in einem Chunk = 8 Paare in Folge. Rumpf: jj = Byteindex (gerade), s_ = Paar,
   cw_ = Chunk-Wort mit den Mantissenbytes jj, jj+1 */
#define DEKODIERE_CHUNK(SCHRITT, chunk, ...)                                                      \
    _Pragma("unroll")                                                                            \
    for (int q_ = 0; q_ < 8; q_++) {                                                             \
        const int jj = 2 * q_; const uint32_t s_ = SCHRITT; const uint32_t cw_ = u4_wort(chunk, jj); __VA_ARGS__ \
    }

/* ---- Token-Pfad: persistente Bloecke, x fp32 -> Partialsummen (fp32) -> y BF16 + Bias. ---- */
template<bool XSM>
__global__ void __launch_bounds__(TBP, 1)
kern_matvec_bf16(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
                 const uint8_t *__restrict__ hdr, const uint32_t *__restrict__ tab,
                 const uint8_t *__restrict__ low, const float *__restrict__ x, float *__restrict__ teil,
                 uint64_t n, uint32_t block, uint32_t K, uint32_t P, uint32_t n_items)
{
    extern __shared__ uint32_t smem[];                          /* [0, 4096): Tabelle; danach x (fp32, K) */
    for (uint32_t i = threadIdx.x; i < TOT; i += TBP) smem[i] = __ldg(tab + i);
    if (XSM) { float4 *xs4 = (float4*)(smem + TOT); const float4 *xg = (const float4*)x;
               for (uint32_t i = threadIdx.x; i < K / 4; i += TBP) xs4[i] = __ldg(xg + i); }
    __syncthreads();
    const uint32_t sm_tab = (uint32_t)__cvta_generic_to_shared(smem);
    const float *xq = XSM ? (const float*)(smem + TOT) : x;
    uint32_t lane = threadIdx.x & 31, n_chunks = block / 512;

    for (uint32_t b = blockIdx.x * (TBP / 32) + (threadIdx.x >> 5); b < n_items; b += gridDim.x * (TBP / 32)) {
        uint64_t basis = (uint64_t)b * block;
        Z st; Esc E; st.aufsetzen(daten, basen, hdr, b, lane, E);
        uint32_t m_cur = (uint32_t)(basis / K);
        uint64_t pl = basis + lane * 16;
        uint32_t m_l = (uint32_t)(pl / K), kk = (uint32_t)(pl - (uint64_t)m_l * K);
        const uint4 *lo4 = (const uint4*)(low + basis) + lane;
        uint4 chunk_nxt = __ldcs(lo4);
        float acc = 0.f;
        for (uint32_t c = 0; c < n_chunks; c++) {
            uint4 chunk = chunk_nxt;
            if (c + 1 < n_chunks) chunk_nxt = __ldcs(lo4 + 32 * (c + 1));
            const float4 *xp = (const float4*)(xq + kk);
            float4 x4[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) x4[i] = XSM ? xp[i] : __ldg(xp + i);
            const float *xw = (const float*)x4;
            float s0 = 0.f, s1 = 0.f;
            DEKODIERE_CHUNK(st.schritt_s(sm_tab, E), chunk, {
                float &sa = (q_ & 1) ? s1 : s0;
                sa = fmaf(gewicht(cw_, s_, jj, 0), xw[jj], sa);
                sa = fmaf(gewicht(cw_, s_, jj + 1, 1), xw[jj + 1], sa);
            })
            float s_l = s0 + s1;
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

/* ---- fp32-Pfad (Tests): y += W x per Atomics, K % 512 == 0, Tabelle global ---- */
__global__ void __launch_bounds__(128, 8)
kern_matvec(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
            const uint8_t *__restrict__ hdr, const uint32_t *__restrict__ tab,
            const uint8_t *__restrict__ low, const float *__restrict__ x, float *__restrict__ y,
            uint64_t n, uint32_t block, uint32_t K)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    if (basis >= n) return;
    Z st; Esc E; st.aufsetzen(daten, basen, hdr, b, lane, E);
    uint32_t m = (uint32_t)(basis / K), kk = (uint32_t)(basis % K) + lane * 16;
    const uint4 *lo4 = (const uint4*)(low + basis) + lane;
    float acc = 0.f;
    uint32_t n_chunks = block / 512;
    for (uint32_t c = 0; c < n_chunks; c++) {
        uint4 chunk = __ldcs(lo4 + 32 * c);
        const float4 *xp = (const float4*)(x + kk);
        float4 x4[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) x4[i] = __ldg(xp + i);
        const float *xw = (const float*)x4;
        DEKODIERE_CHUNK(st.schritt_g(tab, E), chunk, {
            acc = fmaf(gewicht(cw_, s_, jj, 0), xw[jj], acc);
            acc = fmaf(gewicht(cw_, s_, jj + 1, 1), xw[jj + 1], acc);
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
             const uint8_t *__restrict__ hdr, const uint32_t *__restrict__ tab,
             const uint8_t *__restrict__ low, uint16_t *__restrict__ out, uint64_t n, uint32_t block)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    if (basis >= n) return;
    Z st; Esc E; st.aufsetzen(daten, basen, hdr, b, lane, E);
    const uint4 *lo4 = (const uint4*)(low + basis) + lane;
    uint4 *o4 = (uint4*)(out + basis) + 2 * lane;
    uint32_t n_chunks = block / 512;
    for (uint32_t c = 0; c < n_chunks; c++) {
        uint4 chunk = __ldcs(lo4 + 32 * c);
        uint32_t w16[8];
        DEKODIERE_CHUNK(st.schritt_g(tab, E), chunk, { w16[jj >> 1] = zwei_bf16(cw_, s_, jj); })
        o4[64 * c]     = make_uint4(w16[0], w16[1], w16[2], w16[3]);
        o4[64 * c + 1] = make_uint4(w16[4], w16[5], w16[6], w16[7]);
    }
}

/* ---- Zeilen-Gather (Embedding-Lookup): ein Warp je gesuchter Zeile ---- */
__global__ void __launch_bounds__(128)
kern_gather(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
            const uint8_t *__restrict__ hdr, const uint32_t *__restrict__ tab,
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
        Z st; Esc E; st.aufsetzen(daten, basen, hdr, b, lane, E);
        const uint4 *lo4 = (const uint4*)(low + basis) + lane;
        for (uint32_t c = 0; c < n_chunks; c++) {
            uint4 chunk = __ldg(lo4 + 32 * c);
            uint64_t g = basis + (uint64_t)c * 512 + lane * 16;
            DEKODIERE_CHUNK(st.schritt_g(tab, E), chunk, {
                uint32_t v2 = zwei_bf16(cw_, s_, jj);
                if (g + jj     >= g0 && g + jj     < g1) o[g + jj     - g0] = (uint16_t)v2;
                if (g + jj + 1 >= g0 && g + jj + 1 < g1) o[g + jj + 1 - g0] = (uint16_t)(v2 >> 16);
            })
        }
    }
}

/* ============================ Host: Starter ============================ */
static int g_sms = 0; static size_t g_sm_max = 0;     /* SMs und max. dynamisches Shared Memory (nwc_setup) */
EXPORT int nwc_setup(void);

EXPORT int nwc_matvec(void *stream, const uint8_t *daten, const uint32_t *basen, const uint8_t *hdr,
                      const uint32_t *tab, const uint8_t *low, const float *x, float *y,
                      uint64_t n, uint32_t block, uint32_t K)
{
    if (K % 512 || block % 512 || n % block || n % K) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    uint32_t M = (uint32_t)(n / K);
    cudaMemsetAsync(y, 0, 4ull * M, st);
    int bl = (int)((n / block * 32 + 127) / 128);
    kern_matvec<<<bl, 128, 0, st>>>(daten, basen, hdr, tab, low, x, y, n, block, K);
    return (int)cudaGetLastError();
}

EXPORT int nwc_dequant(void *stream, const uint8_t *daten, const uint32_t *basen, const uint8_t *hdr,
                       const uint32_t *tab, const uint8_t *low, uint16_t *out, uint64_t n, uint32_t block)
{
    if (block % 512 || n % block) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    int bl = (int)((n / block * 32 + 127) / 128);
    kern_dequant<<<bl, 128, 0, st>>>(daten, basen, hdr, tab, low, out, n, block);
    return (int)cudaGetLastError();
}

/* x fp32 (K), Bias BF16 (M) oder NULL, y BF16 (M) */
EXPORT int nwc_linear_bf16(void *stream, const uint8_t *daten, const uint32_t *basen, const uint8_t *hdr,
                           const uint32_t *tab, const uint8_t *low, const float *x, const uint16_t *bias,
                           float *scratch, uint16_t *y, uint64_t n, uint32_t block, uint32_t K)
{
    if (K % 16 || K < 512 || block % 512 || n % block || n % K) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    uint32_t M = (uint32_t)(n / K), P = K / block + 2;
    if (g_sms == 0) nwc_setup();
    uint32_t n_items = (uint32_t)(n / block);
    int bl = (int)((n_items + TBP / 32 - 1) / (TBP / 32)); if (bl > g_sms) bl = g_sms;
    size_t sm_x = 4096 * 4 + (size_t)K * 4;
    if (sm_x <= g_sm_max) kern_matvec_bf16<true ><<<bl, TBP, sm_x,     st>>>(daten, basen, hdr, tab, low, x, scratch, n, block, K, P, n_items);
    else                  kern_matvec_bf16<false><<<bl, TBP, 4096 * 4, st>>>(daten, basen, hdr, tab, low, x, scratch, n, block, K, P, n_items);
    kern_f32_zu_bf16_bias<<<(M + 255) / 256, 256, 0, st>>>(scratch, bias, y, M, K, P, block);
    return (int)cudaGetLastError();
}

EXPORT int nwc_gather(void *stream, const uint8_t *daten, const uint32_t *basen, const uint8_t *hdr,
                      const uint32_t *tab, const uint8_t *low, const int64_t *ids, uint32_t n_ids,
                      uint16_t *out, uint64_t n, uint32_t block, uint32_t K)
{
    if (K % 16 || block % 512 || n % block || n % K) return -1;
    if (n_ids == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    kern_gather<<<(n_ids * 32 + 127) / 128, 128, 0, st>>>(daten, basen, hdr, tab, low, ids, n_ids, out, n, block, K);
    return (int)cudaGetLastError();
}

EXPORT int nwc_setup(void) {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev);
    int optin = 0; cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    g_sm_max = (size_t)optin;
    cudaFuncSetAttribute(kern_matvec_bf16<true >, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
    cudaFuncSetAttribute(kern_matvec_bf16<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096 * 4);
    cudaFuncSetAttribute(kern_matvec, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    return g_sms;
}
