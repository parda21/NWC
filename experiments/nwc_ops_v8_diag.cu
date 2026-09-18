/* nwc_ops -- NWC fuer PyTorch per ctypes (Windows: nwc_ops.dll, Linux: nwc_ops.so).
   Format NWC2 Version 8:
     - BF16 in Byte-Ebenen: Mantisse roh (low[]), Exponentbyte rANS-kodiert (12 Skalenbits, 16-Bit-Renorm, LE)
     - PAAR-SYMBOLE: ein rANS-Schritt liefert zwei Exponentbytes (= zwei Gewichte).
     - EINE 32-Bit-Tabelle mit 4096 Eintraegen: [Paar 16 | freq 8 | bias 8] (freq <= 255: kostet keine Rate,
       das haeufigste Paar hat ~1/16). Paar 0xFFFF = Escape-Sentinel (wird nie kodiert): die zwei Bytes stehen
       dann roh im ESCAPE-SEITENSTROM der Lane (eigene Warteschlange, damit Renorm und Escape nie im selben
       Schritt aus derselben Warteschlange lesen).
     - Block = `block` Symbole (Vielfaches von 512) = 32 Lanes x NSTATE rANS-Zustaende (2 oder 4). Zustand h
       einer Lane traegt je 512er-Chunk die Gewichte h*WPS .. h*WPS+WPS-1 der 16 Gewichte der Lane.
     - Lane-Region: [Strom 0][Strom 1]..[Escape-Worte], ohne Padding (2-Byte-Granularitaet); Header je Lane
       NSTATE+1 Bytes: Laenge/2 je Strom, Anzahl Escape-Worte. Startoffset per Warp-Praefixsumme;
       Bloecke 8-Byte-ausgerichtet (basen[]).
     - Bitleser ohne Fenster: je Zustand ein vorgeladenes 16-Bit-Wort (LDG.U16 je Renorm, praedizierte
       PTX-Ladung statt Verzweigung) -> ~19 Instruktionen je rANS-Schritt statt ~45 in v7.
   Exporte:
     nwc_info        Host. 0: Formatversion, 1: NSTATE, 2: Tabellenwoerter (u32), 3: Headerbytes je Block
     nwc_encode      Host. BF16 (n, n % block == 0) -> freq[1024], psym[1024], basen, hdr, daten, low
     nwc_build_tab   Host. freq/psym -> tab[4096 u32]
     nwc_matvec      Device, fp32-Pfad (Tests)
     nwc_linear_bf16 Device. y = W x + b, x fp32, y BF16, dekodiert in Registern, K % 16 == 0, K >= 512
     nwc_dequant     Device. komprimiert -> BF16[n]
     nwc_gather      Device. Zeilen ids[] einer [M x K]-Matrix -> BF16[n_ids x K] (Embedding-Lookup)
   v7 (zwei Tabellen, 64-Bit-Fenster) liegt in experimente/nwc_ops_v7.cu, v6 mit Diagnoseschaltern in
   experimente/nwc_ops_v6_diag.cu.                                                                       */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define VERSION    8
#define SCALE_BITS 12
#define TOT        (1u << SCALE_BITS)
#define L16        (1u << 16)
#define LANES      32u
#ifndef NSTATE
#define NSTATE 2                         /* rANS-Zustaende je Lane (2 oder 4) */
#endif
#define WPS        (16 / NSTATE)         /* Gewichte je Zustand und Chunk */
#define STREAMS    (LANES * NSTATE)
#define HDR        (NSTATE + 1)          /* Headerbytes je Lane */
#define HDR_BLOCK  (LANES * HDR)         /* Headerbytes je Block */
#define ALIGN      8u
#define NPAIR      1024u                 /* max. kodierte Paare, 0 = Escape */
#define FMAX       255u                  /* freq in 8 Bit */
#define TAB_WORDS  TOT
#define ESC_PAAR   0xFFFFu               /* Sentinel im Paarfeld */
#ifndef MINB
#define MINB (NSTATE == 4 ? 8 : 10)
#endif
#ifndef TB
#ifdef SMTAB
#define TB 512                           /* grosse Bloecke: Tabellenkopie (16 KB) amortisieren */
#else
#define TB 128
#endif
#endif
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

EXPORT int nwc_info(int was) {
    switch (was) { case 0: return VERSION; case 1: return NSTATE; case 2: return (int)TAB_WORDS; case 3: return (int)HDR_BLOCK; }
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
/* Position des i-ten Gewichts von Strom s (= NSTATE*lane + h) im Block */
static inline uint32_t pos_im_block(uint32_t s, uint32_t i) {
    return (i / WPS) * 512 + (s / NSTATE) * 16 + (s % NSTATE) * WPS + (i % WPS);
}

/* rANS rueckwaerts; Paare pr[] (16 Bit), map16[paar] = Index (0 = Escape, Rohbytes gehen in den Seitenstrom) */
static size_t kodiere_strom(const uint16_t *pr, uint32_t n, const uint16_t *map16, const uint16_t *freq,
                            const uint32_t *kum, uint8_t *ende) {
    uint8_t *q = ende; uint32_t x = L16;
    for (int64_t i = (int64_t)n - 1; i >= 0; i--) {
        uint32_t s = map16[pr[i]];
        uint32_t f = freq[s], x_max = f << (32 - SCALE_BITS);
        while (x >= x_max) { q -= 2; q[0] = (uint8_t)x; q[1] = (uint8_t)(x >> 8); x >>= 16; }
        x = ((x / f) << SCALE_BITS) + (x % f) + kum[s];
    }
    q -= 4; q[0] = (uint8_t)x; q[1] = (uint8_t)(x >> 8); q[2] = (uint8_t)(x >> 16); q[3] = (uint8_t)(x >> 24);
    return (size_t)(ende - q);
}

EXPORT int64_t nwc_encode(const uint16_t *w, uint64_t n, uint32_t block, uint16_t *freq, uint16_t *psym,
                          uint32_t *basen, uint8_t *hdr, uint8_t *daten, uint64_t daten_cap, uint8_t *low)
{
    if (block % 512 || n % block) return -1;
    uint8_t *high = (uint8_t*)malloc(n);
    for (uint64_t i = 0; i < n; i++) { low[i] = (uint8_t)w[i]; high[i] = (uint8_t)(w[i] >> 8); }

    uint32_t n_bloecke = (uint32_t)(n / block), n_strom = block / STREAMS, n_paar = n_strom / 2;
    if (n_paar * NSTATE > 255) { free(high); return -1; }             /* Escape-Zaehler ist u8 */
    /* 1. Paar-Histogramm in Stromreihenfolge (Paare = benachbarte Gewichte 2k, 2k+1 eines Stroms) */
    uint64_t *z2 = (uint64_t*)calloc(65536, 8);
    uint16_t *pr = (uint16_t*)malloc(2ull * n_paar * NSTATE);
    for (uint32_t b = 0; b < n_bloecke; b++) {
        uint64_t basis = (uint64_t)b * block;
        for (uint32_t s = 0; s < STREAMS; s++)
            for (uint32_t k = 0; k < n_paar; k++)
                z2[high[basis + pos_im_block(s, 2 * k)] | (high[basis + pos_im_block(s, 2 * k + 1)] << 8)]++;
    }
    /* 2. Kodierte Paare: p >= 1/TOT, nach Haeufigkeit, max NPAIR-1, nie den Sentinel; Rest -> Escape (Index 0) */
    uint64_t n_ges = (uint64_t)n_bloecke * STREAMS * n_paar;
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

    /* 3. Lane-Regionen kodieren: [Strom 0..NSTATE-1][Escape-Worte in Dekodierreihenfolge] */
    uint8_t *tmp = (uint8_t*)malloc(4ull * n_strom + 64);
    uint64_t pos = 0; int64_t rc = -1;
    for (uint32_t b = 0; b < n_bloecke; b++) {
        uint64_t basis = (uint64_t)b * block;
        pos = (pos + ALIGN - 1) / ALIGN * ALIGN;
        basen[b] = (uint32_t)pos;
        for (uint32_t lane = 0; lane < LANES; lane++) {
            uint8_t *h = hdr + (uint64_t)b * HDR_BLOCK + lane * HDR;
            for (uint32_t hs = 0; hs < NSTATE; hs++) {
                uint32_t s = NSTATE * lane + hs; uint16_t *p = pr + hs * n_paar;
                for (uint32_t k = 0; k < n_paar; k++)
                    p[k] = (uint16_t)(high[basis + pos_im_block(s, 2 * k)] | (high[basis + pos_im_block(s, 2 * k + 1)] << 8));
                size_t len = kodiere_strom(p, n_paar, map16, freq, kum, tmp + 4ull * n_strom + 64);
                if (len / 2 > 255 || pos + len + 64 > daten_cap) goto ende;
                h[hs] = (uint8_t)(len / 2);
                memcpy(daten + pos, tmp + 4ull * n_strom + 64 - len, len);
                pos += len;
            }
            uint32_t n_esc = 0;                                 /* Dekodierreihenfolge: Paar k, darin Zustand hs */
            for (uint32_t k = 0; k < n_paar; k++)
                for (uint32_t hs = 0; hs < NSTATE; hs++) {
                    uint16_t pp = pr[hs * n_paar + k];
                    if (map16[pp] == 0) {
                        if (pos + 2 + 64 > daten_cap) goto ende;
                        daten[pos] = (uint8_t)pp; daten[pos + 1] = (uint8_t)(pp >> 8); pos += 2; n_esc++;
                    }
                }
            h[NSTATE] = (uint8_t)n_esc;
        }
    }
    rc = (int64_t)((pos + ALIGN - 1) / ALIGN * ALIGN + 32);
ende:
    free(tmp); free(z); free(map16); free(pr); free(z2); free(high);
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

/* Lane-Region finden: exklusive Praefixsumme der Lane-Laengen (in 2-Byte-Worten) ueber den Warp.
   rel[h] = Byteoffset des Stroms h, rel[NSTATE] = Byteoffset der Escape-Worte */
__device__ __forceinline__ void lane_region(const uint8_t *__restrict__ hdr, uint32_t b, uint32_t lane, uint32_t *rel) {
    const uint8_t *h = hdr + (uint64_t)b * HDR_BLOCK + lane * HDR;
    uint32_t l[HDR], v = 0;
    #pragma unroll
    for (int i = 0; i < HDR; i++) { l[i] = __ldg(h + i); v += l[i]; }
    uint32_t x = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) { uint32_t y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= (uint32_t)o) x += y; }
    uint32_t start = x - v;
    #pragma unroll
    for (int i = 0; i < HDR; i++) { rel[i] = start * 2; start += l[i]; }
    rel[HDR] = start * 2;                                       /* Ende der Lane-Region */
}

/* Tabellenzugriff: global (L1) oder Kopie im Shared Memory (-DSMTAB, immun gegen L1-Verdraengung) */
#ifdef SMTAB
__shared__ uint32_t stab_[TOT];
__device__ __forceinline__ void tab_laden(const uint32_t *__restrict__ tab) {
    for (uint32_t i = threadIdx.x; i < TOT; i += blockDim.x) stab_[i] = __ldg(tab + i);
    __syncthreads();
}
#define TAB_LESEN(tab, slot) stab_[(slot)]
#else
#define tab_laden(tab) ((void)0)
#ifdef NOGATHER   /* Diagnose: Tabelleneintrag aus Arithmetik (dekodiert Muell) */
#define TAB_LESEN(tab, slot) (((slot) * 2654435761u) | 0x00010100u)
#else
#define TAB_LESEN(tab, slot) __ldg((tab) + (slot))
#endif
#endif

/* Escape-Seitenstrom einer Lane: ein vorgeladenes Wort */
struct Esc {
    uint32_t ea; const uint8_t *pe;
#ifdef INTPLUS
    uint32_t sel;
#endif
    __device__ __forceinline__ void init(const uint8_t *start) { ea = __ldg((const uint16_t*)start); pe = start + 2; }
};

struct Z {
    uint32_t xs, a; const uint8_t *p;                                    /* Zustand, naechstes Wort, Leseposition */
    __device__ __forceinline__ void init(const uint8_t *start) {          /* Start 2-B-ausgerichtet */
        const uint16_t *q = (const uint16_t*)start;
        xs = __ldg(q) | ((uint32_t)__ldg(q + 1) << 16);
        a = __ldg(q + 2); p = (const uint8_t*)(q + 3);
    }
    /* ein Schritt = ein Paar in Bytes 0/1 (Bytes 2/3 = 0) */
    __device__ __forceinline__ uint32_t schritt(const uint32_t *__restrict__ tab, Esc &E) {
        uint32_t e = TAB_LESEN(tab, xs & (TOT - 1));
        xs = __byte_perm(e, 0u, 0x4441) * (xs >> SCALE_BITS) + (e & 0xFFu);          /* freq * q + bias */
#ifdef INTPLUS   /* Diagnose: INTPLUS zusaetzliche INT-Instruktionen (Identitaets-PRMT mit Laufzeit-Selektor) auf der Kette */
        #pragma unroll
        for (int k_ = 0; k_ < INTPLUS; k_++) xs = __byte_perm(xs, 0u, E.sel);
#endif
        /* Renorm, praediziert: xs = (xs << 16) | a; a = naechstes Wort; p += 2 */
#if defined(NORENORM)   /* Diagnose: kein Renorm-Block */
        xs |= 0x10000u;
#elif defined(NOLOAD)   /* Diagnose: Renorm-Wort aus Registern statt Speicher (dekodiert Muell) */
        asm("{\n\t.reg .pred q;\n\t"
            "setp.lt.u32 q, %0, 65536;\n\t"
            "@q prmt.b32 %0, %0, %1, 0x1054;\n\t"
            "@q mad.lo.u32 %1, %1, 3, 12345;\n\t"
            "@q add.s64 %2, %2, 2;\n\t}"
            : "+r"(xs), "+r"(a), "+l"(p));
#elif defined(PF_LOOP)   /* am Sektoranfang den naechsten Sektor in den L1 holen (CCTL.PF1) */
        asm("{\n\t.reg .pred q, r;\n\t.reg .b64 t;\n\t"
            "setp.lt.u32 q, %0, 65536;\n\t"
            "@q prmt.b32 %0, %0, %1, 0x1054;\n\t"
            "@q ld.global.nc.u16 %1, [%2];\n\t"
            "and.b64 t, %2, 30;\n\t"
            "setp.eq.and.b64 r, t, 0, q;\n\t"
            "@r prefetch.global.L1 [%2+32];\n\t"
            "@q add.s64 %2, %2, 2;\n\t}"
            : "+r"(xs), "+r"(a), "+l"(p));
#else
        asm("{\n\t.reg .pred q;\n\t"
            "setp.lt.u32 q, %0, 65536;\n\t"
            "@q prmt.b32 %0, %0, %1, 0x1054;\n\t"
            "@q ld.global.nc.u16 %1, [%2];\n\t"
            "@q add.s64 %2, %2, 2;\n\t}"
            : "+r"(xs), "+r"(a), "+l"(p));
#endif
        uint32_t s = e >> 16;
#ifndef NOESC   /* Diagnose: ohne Escape-Pfad (dekodiert Escape-Paare falsch) */
        /* Escape (Paar 0xFFFF), praediziert: s = Seitenstrom-Wort, nachladen */
        asm("{\n\t.reg .pred q;\n\t"
            "setp.ge.u32 q, %3, 0xFFFF0000;\n\t"
            "@q mov.b32 %0, %1;\n\t"
            "@q ld.global.nc.u16 %1, [%2];\n\t"
            "@q add.s64 %2, %2, 2;\n\t}"
            : "+r"(s), "+r"(E.ea), "+l"(E.pe) : "r"(e));
#endif
        return s;
    }
};

/* 16 Gewichte einer Lane in einem Chunk dekodieren: Zustand hh liefert Paare fuer die Bytes hh*WPS + 2q.
   Im Rumpf: jj = Byteindex (gerade), s_ = Paar, cw_ = Chunk-Wort mit den Mantissenbytes jj, jj+1, hh = Zustand. */
#define DEKODIERE_CHUNK(st, E, chunk, tab, ...)                                                   \
    _Pragma("unroll")                                                                            \
    for (int q_ = 0; q_ < WPS / 2; q_++) {                                                       \
        uint32_t ps_[NSTATE];                                                                    \
        _Pragma("unroll")                                                                        \
        for (int h_ = 0; h_ < NSTATE; h_++) ps_[h_] = (st)[h_].schritt(tab, E);                  \
        _Pragma("unroll")                                                                        \
        for (int h_ = 0; h_ < NSTATE; h_++) {                                                    \
            const int hh = h_, jj = h_ * WPS + 2 * q_; const uint32_t s_ = ps_[h_];               \
            const uint32_t cw_ = u4_wort(chunk, jj); (void)hh; __VA_ARGS__                       \
        }                                                                                        \
    }

#ifndef PF_INIT
#define PF_INIT 0      /* >0: so viele weitere 32-B-Sektoren der Lane-Region beim Start in den L1 holen (CCTL.PF1) */
#endif
#ifdef INTPLUS         /* Selektor 0x3210 (Identitaet), fuer den Compiler aber ein Laufzeitwert (n < 2^40) */
#define INTPLUS_INIT(E) (E).sel = 0x3210u | (uint32_t)(n >> 40);
#else
#define INTPLUS_INIT(E)
#endif
#define ZUSTAENDE_INIT(st, E, daten, basen, hdr, b, lane)                                         \
    uint32_t rel_[HDR + 1]; lane_region(hdr, b, lane, rel_);                                     \
    _Pragma("unroll")                                                                            \
    for (int h_ = 0; h_ < NSTATE; h_++) (st)[h_].init(daten + basen[b] + rel_[h_]);              \
    E.init(daten + basen[b] + rel_[NSTATE]);                                                     \
    INTPLUS_INIT(E)                                                                              \
    _Pragma("unroll")                                                                            \
    for (int k_ = 1; k_ <= PF_INIT; k_++)                                                        \
        if (rel_[0] + 32 * k_ < rel_[HDR])                                                       \
            asm volatile("prefetch.global.L1 [%0];" :: "l"(daten + basen[b] + rel_[0] + 32 * k_));

/* ======== Token-Pfad v8: persistente Bloecke (1024 Threads, einer je SM), Tabelle + x im Shared Memory,
   Dekodierschritt mit moeglichst wenigen INT-Pipe-Instruktionen (GA10x: IADD3/LOP3/SHF/ISETP/SEL laufen mit
   halber Rate, IMAD/PRMT/FFMA mit voller):
     slot = xs - (q << 12) per IMAD statt LOP3; freq/bias/Paar per PRMT; Escape als Verzweigung (selten). ======== */
#define TBP 1024u
struct Zp {
    uint32_t xs, a; const uint8_t *p;
    __device__ __forceinline__ void init(const uint8_t *start) {
        const uint16_t *q = (const uint16_t*)start;
        xs = __ldg(q) | ((uint32_t)__ldg(q + 1) << 16);
        a = __ldg(q + 2); p = (const uint8_t*)(q + 3);
    }
    __device__ __forceinline__ uint32_t schritt(uint32_t sm_tab, Esc &E) {
#ifdef QHI
        uint32_t q = __umulhi(xs, 1u << 20);                                  /* xs >> 12 auf der FMA-Pipe? */
#else
        uint32_t q = xs >> SCALE_BITS;
#endif
        uint32_t adr, e;
        asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(adr) : "r"(q), "r"(0xFFFFC000u), "r"(sm_tab + xs * 4));   /* (xs - (q<<12))*4 + base = xs*4 - q*16384 + base */
        asm("ld.shared.u32 %0, [%1];" : "=r"(e) : "r"(adr));
        xs = __byte_perm(e, 0u, 0x4441) * q + __byte_perm(e, 0u, 0x4440);   /* freq * q + bias */
        asm("{\n\t.reg .pred q;\n\t"
            "setp.lt.u32 q, %0, 65536;\n\t"
            "@q prmt.b32 %0, %0, %1, 0x1054;\n\t"
            "@q ld.global.nc.u16 %1, [%2];\n\t"
            "@q add.s64 %2, %2, 2;\n\t}"
            : "+r"(xs), "+r"(a), "+l"(p));
        uint32_t s = __byte_perm(e, 0u, 0x4432);                               /* Paar (Bytes 2/3 -> 0/1) */
        if (s == ESC_PAAR) { s = E.ea; E.ea = __ldg((const uint16_t*)E.pe); E.pe += 2; }
        return s;
    }
};

template<bool XSM>
__global__ void __launch_bounds__(TBP, 1)
kern_matvec_bf16_p(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
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
        uint32_t rel_[HDR + 1]; lane_region(hdr, b, lane, rel_);
        Zp st[NSTATE]; Esc E;
        #pragma unroll
        for (int h = 0; h < NSTATE; h++) st[h].init(daten + basen[b] + rel_[h]);
        E.init(daten + basen[b] + rel_[NSTATE]);

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
            float sacc[NSTATE];
            #pragma unroll
            for (int h = 0; h < NSTATE; h++) sacc[h] = 0.f;
            #pragma unroll
            for (int q_ = 0; q_ < WPS / 2; q_++) {
                uint32_t ps_[NSTATE];
                #pragma unroll
                for (int h = 0; h < NSTATE; h++) ps_[h] = st[h].schritt(sm_tab, E);
                #pragma unroll
                for (int h = 0; h < NSTATE; h++) {
                    const int jj = h * WPS + 2 * q_; const uint32_t cw_ = u4_wort(chunk, jj);
                    sacc[h] = fmaf(gewicht(cw_, ps_[h], jj, 0), xw[jj], sacc[h]);
                    sacc[h] = fmaf(gewicht(cw_, ps_[h], jj + 1, 1), xw[jj + 1], sacc[h]);
                }
            }
            float s_l = 0.f;
            #pragma unroll
            for (int h = 0; h < NSTATE; h++) s_l += sacc[h];
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

/* ---- Token-Pfad (alt, Diagnoseschalter; wird von nwc_linear_bf16 nur mit -DALTPFAD benutzt) ---- */
__global__ void __launch_bounds__(TB, MINB * 128 / TB)
kern_matvec_bf16(const uint8_t *__restrict__ daten, const uint32_t *__restrict__ basen,
                 const uint8_t *__restrict__ hdr, const uint32_t *__restrict__ tab,
                 const uint8_t *__restrict__ low, const float *__restrict__ x, float *__restrict__ teil,
                 uint64_t n, uint32_t block, uint32_t K, uint32_t P)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    tab_laden(tab);
    if (basis >= n) return;
    Z st[NSTATE]; Esc E; ZUSTAENDE_INIT(st, E, daten, basen, hdr, b, lane)

    uint32_t m_cur = (uint32_t)(basis / K);
    uint64_t pl = basis + lane * 16;
    uint32_t m_l = (uint32_t)(pl / K), kk = (uint32_t)(pl - (uint64_t)m_l * K);
    const uint4 *lo4 = (const uint4*)(low + basis) + lane;
    uint4 chunk_nxt = __ldcs(lo4);
    float acc = 0.f;
    uint32_t n_chunks = block / 512;

    for (uint32_t c = 0; c < n_chunks; c++) {
#ifdef NOCHUNK    /* Diagnose: keine Mantissen-Ladung */
        uint4 chunk = make_uint4(c, c * 3u, c * 5u, c * 7u);
#else
        uint4 chunk = chunk_nxt;
        if (c + 1 < n_chunks) chunk_nxt = __ldcs(lo4 + 32 * (c + 1));
#endif
        float4 x4[4];
#ifdef NOX        /* Diagnose: keine x-Ladung */
        #pragma unroll
        for (int i = 0; i < 4; i++) x4[i] = make_float4(1.f, 2.f, 3.f, (float)c);
#else
        const float4 *xp = (const float4*)(x + kk);
        #pragma unroll
        for (int i = 0; i < 4; i++) x4[i] = __ldg(xp + i);
#endif
        const float *xw = (const float*)x4;
        float sacc[NSTATE];
        #pragma unroll
        for (int h = 0; h < NSTATE; h++) sacc[h] = 0.f;
#ifdef NOWEIGHT   /* Diagnose: keine Gewichtsbildung/FFMA */
        DEKODIERE_CHUNK(st, E, chunk, tab, { sacc[hh] += __uint_as_float(s_ ^ cw_); })
#else
        DEKODIERE_CHUNK(st, E, chunk, tab, {
            sacc[hh] = fmaf(gewicht(cw_, s_, jj, 0), xw[jj], sacc[hh]);
            sacc[hh] = fmaf(gewicht(cw_, s_, jj + 1, 1), xw[jj + 1], sacc[hh]);
        })
#endif
        float s_l = 0.f;
        #pragma unroll
        for (int h = 0; h < NSTATE; h++) s_l += sacc[h];
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
            const uint8_t *__restrict__ hdr, const uint32_t *__restrict__ tab,
            const uint8_t *__restrict__ low, const float *__restrict__ x, float *__restrict__ y,
            uint64_t n, uint32_t block, uint32_t K)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = tid >> 5, lane = tid & 31;
    uint64_t basis = (uint64_t)b * block;
    tab_laden(tab);
    if (basis >= n) return;
    Z st[NSTATE]; Esc E; ZUSTAENDE_INIT(st, E, daten, basen, hdr, b, lane)
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
        DEKODIERE_CHUNK(st, E, chunk, tab, {
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
    tab_laden(tab);
    if (basis >= n) return;
    Z st[NSTATE]; Esc E; ZUSTAENDE_INIT(st, E, daten, basen, hdr, b, lane)
    const uint4 *lo4 = (const uint4*)(low + basis) + lane;
    uint4 *o4 = (uint4*)(out + basis) + 2 * lane;
    uint32_t n_chunks = block / 512;
    for (uint32_t c = 0; c < n_chunks; c++) {
        uint4 chunk = __ldcs(lo4 + 32 * c);
        uint32_t w16[8];
        DEKODIERE_CHUNK(st, E, chunk, tab, { w16[jj >> 1] = zwei_bf16(cw_, s_, jj); })
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
    tab_laden(tab);
    if (warp >= n_ids) return;
    int64_t r = ids[warp];
    uint64_t g0 = (uint64_t)r * K, g1 = g0 + K;
    if (g1 > n) return;
    uint32_t b0 = (uint32_t)(g0 / block), b1 = (uint32_t)((g1 - 1) / block);
    uint16_t *o = out + (uint64_t)warp * K;
    uint32_t n_chunks = block / 512;
    for (uint32_t b = b0; b <= b1; b++) {
        uint64_t basis = (uint64_t)b * block;
        Z st[NSTATE]; Esc E; ZUSTAENDE_INIT(st, E, daten, basen, hdr, b, lane)
        const uint4 *lo4 = (const uint4*)(low + basis) + lane;
        for (uint32_t c = 0; c < n_chunks; c++) {
            uint4 chunk = __ldg(lo4 + 32 * c);
            uint64_t g = basis + (uint64_t)c * 512 + lane * 16;
            DEKODIERE_CHUNK(st, E, chunk, tab, {
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
#ifdef ALTPFAD
    int bl = (int)((n / block * 32 + TB - 1) / TB);
    kern_matvec_bf16<<<bl, TB, 0, st>>>(daten, basen, hdr, tab, low, x, scratch, n, block, K, P);
#else
    if (g_sms == 0) nwc_setup();
    uint32_t n_items = (uint32_t)(n / block);
    int bl = (int)((n_items + TBP / 32 - 1) / (TBP / 32)); if (bl > g_sms) bl = g_sms;
    size_t sm_x = 4096 * 4 + (size_t)K * 4;
    if (sm_x <= g_sm_max) kern_matvec_bf16_p<true ><<<bl, TBP, sm_x,     st>>>(daten, basen, hdr, tab, low, x, scratch, n, block, K, P, n_items);
    else                  kern_matvec_bf16_p<false><<<bl, TBP, 4096 * 4, st>>>(daten, basen, hdr, tab, low, x, scratch, n, block, K, P, n_items);
#endif
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
    cudaFuncSetAttribute(kern_matvec_bf16_p<true >, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
    cudaFuncSetAttribute(kern_matvec_bf16_p<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096 * 4);
    cudaFuncSetAttribute(kern_matvec_bf16, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(kern_matvec, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    return g_sms;
}
