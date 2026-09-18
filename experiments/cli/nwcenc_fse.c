/* NWC2-Encoder, Formatversion 5: tANS (FSE) statt rANS.
   - 12-Bit-Tabelle (4096 Zustaende), Symbole nach FSE-Schema gespreizt
   - pro Symbol: Zustand -> (Symbol, nbBits, Basiszustand); Dekoder liest nbBits ohne Verzweigung
   - Bitstrom je Teilstrom: [u32 Startzustand][u32 frei][u32-Woerter LSB-first, vorwaerts lesbar]
   - Symbolzuordnung wie Version 4 (Chunk-Layout, 16 aufeinanderfolgende Gewichte je Lane)
   - Selbsttest dekodiert bitexakt

   nutzung: nwcenc_fse <in.raw> <out.nwc> [block=8192]                                   */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>

#define TLOG   12
#define L      (1u << TLOG)
#define LANES  32
#define ALIGN  8

static double jetzt(void) {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
static int highbit(uint32_t v) { int r = 0; while (v >>= 1) r++; return r; }

static void normiere(const uint64_t *z, uint16_t *freq) {
    uint64_t gesamt = 0; for (int i = 0; i < 256; i++) gesamt += z[i];
    uint32_t f[256]; int idx[256], n_idx = 0;
    for (int i = 0; i < 256; i++) {
        f[i] = 0;
        if (z[i]) { uint64_t v = z[i] * L / gesamt; f[i] = v < 1 ? 1 : (uint32_t)v; idx[n_idx++] = i; }
    }
    for (int a = 0; a < n_idx; a++)
        for (int b = a + 1; b < n_idx; b++)
            if (f[idx[b]] > f[idx[a]]) { int t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
    long diff = (long)L; for (int i = 0; i < 256; i++) diff -= f[i];
    for (int k = 0; diff != 0; k++) {
        int i = idx[k % n_idx];
        if (diff > 0) { f[i]++; diff--; }
        else if (f[i] > 1) { f[i]--; diff++; }
    }
    for (int i = 0; i < 256; i++) freq[i] = (uint16_t)f[i];
}

/* ---- FSE-Tabellen ---- */
typedef struct { uint32_t deltaNbBits; int32_t deltaFindState; } SymTT;
static uint8_t  tab_sym[L];         /* Dekoder: Zustand -> Symbol           */
static uint8_t  tab_nb[L];          /* Dekoder: Zustand -> nbBits           */
static uint16_t tab_neu[L];         /* Dekoder: Zustand -> Basis-Neuzustand */
static uint16_t stateTable[L];      /* Encoder                              */
static SymTT    symTT[256];

static void baue_tabellen(const uint16_t *freq) {
    /* Spreizung */
    uint32_t step = (L >> 1) + (L >> 3) + 3, pos = 0;
    for (int s = 0; s < 256; s++)
        for (uint32_t i = 0; i < freq[s]; i++) { tab_sym[pos] = (uint8_t)s; pos = (pos + step) & (L - 1); }
    /* Dekodertabelle */
    uint32_t next[256]; for (int s = 0; s < 256; s++) next[s] = freq[s];
    for (uint32_t u = 0; u < L; u++) {
        int s = tab_sym[u]; uint32_t x = next[s]++;
        int nb = TLOG - highbit(x);
        tab_nb[u] = (uint8_t)nb; tab_neu[u] = (uint16_t)((x << nb) - L);
    }
    /* Encodertabellen */
    uint32_t cumul[257]; cumul[0] = 0; for (int s = 0; s < 256; s++) cumul[s + 1] = cumul[s] + freq[s];
    uint32_t c2[256]; memcpy(c2, cumul, 256 * 4);
    for (uint32_t u = 0; u < L; u++) stateTable[c2[tab_sym[u]]++] = (uint16_t)(L + u);
    uint32_t total = 0;
    for (int s = 0; s < 256; s++) {
        uint32_t f = freq[s];
        if (f == 0) { symTT[s].deltaNbBits = 0; symTT[s].deltaFindState = 0; continue; }
        if (f == 1) { symTT[s].deltaNbBits = (TLOG << 16) - L; symTT[s].deltaFindState = (int32_t)total - 1; }
        else {
            uint32_t maxBitsOut = TLOG - highbit(f - 1);
            uint32_t minStatePlus = f << maxBitsOut;
            symTT[s].deltaNbBits = (maxBitsOut << 16) - minStatePlus;
            symTT[s].deltaFindState = (int32_t)total - (int32_t)f;
        }
        total += f;
    }
}

/* kodiert n Symbole rueckwaerts; Stream waechst nach vorn ab `ende`; liefert Laenge inkl. 8-Byte-Kopf */
static size_t kodiere_lane(const uint8_t *sym, uint32_t n, uint8_t *ende) {
    uint8_t *q = ende;
    uint64_t acc = 0; int nbits = 0;
    uint32_t x = L;                                       /* Startzustand, in [L, 2L) */
    for (int64_t i = (int64_t)n - 1; i >= 0; i--) {
        int s = sym[i];
        uint32_t nbOut = (x + symTT[s].deltaNbBits) >> 16;
        acc = (acc << nbOut) | (x & ((1u << nbOut) - 1)); nbits += nbOut;   /* neueste Bits unten */
        if (nbits >= 32) { q -= 4; uint32_t w = (uint32_t)(acc >> (nbits - 32)); memcpy(q, &w, 4);
                           nbits -= 32; acc &= (nbits ? ((1ull << nbits) - 1) : 0); }
        x = stateTable[(x >> nbOut) + symTT[s].deltaFindState];
    }
    uint32_t pos0 = 0;
    if (nbits > 0) {                                      /* Fuellbits nach UNTEN, Start bei pos0 */
        pos0 = 32 - (uint32_t)nbits;
        q -= 4; uint32_t w = (uint32_t)(acc << pos0); memcpy(q, &w, 4);
    }
    uint32_t start = x - L;
    q -= 4; memcpy(q, &pos0, 4);
    q -= 4; memcpy(q, &start, 4);
    return (size_t)(ende - q);
}

/* Referenzdekoder: liest vorwaerts, LSB-first */
static void dekodiere_lane(const uint8_t *p, uint32_t n, uint8_t *out) {
    uint32_t u, pos0; memcpy(&u, p, 4); memcpy(&pos0, p + 4, 4); p += 8;
    uint32_t w0, w1; memcpy(&w0, p, 4); memcpy(&w1, p + 4, 4); p += 8;
    uint64_t buf = (uint64_t)w0 | ((uint64_t)w1 << 32); int pos = (int)pos0;
    for (uint32_t i = 0; i < n; i++) {
        out[i] = tab_sym[u];
        int nb = tab_nb[u];
        u = tab_neu[u] + (uint32_t)((buf >> pos) & ((1u << nb) - 1)); pos += nb;
        if (pos >= 32) { uint32_t w; memcpy(&w, p, 4); p += 4; buf = (buf >> 32) | ((uint64_t)w << 32); pos -= 32; }
    }
}

static inline uint32_t pos_im_block(uint32_t lane, uint32_t i) { return (i >> 4) * 512 + lane * 16 + (i & 15); }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "nutzung: nwcenc_fse <in.raw> <out.nwc> [block]\n"); return 2; }
    uint32_t block = argc > 3 ? (uint32_t)atoi(argv[3]) : 8192;
    if (block % 512) { fprintf(stderr, "block Vielfaches von 512\n"); return 1; }

    FILE *fp = fopen(argv[1], "rb"); if (!fp) { perror("in"); return 1; }
    fseek(fp, 0, SEEK_END); long fg = ftell(fp); fseek(fp, 0, SEEK_SET);
    if (fg & 1) { fprintf(stderr, "ungerade Byteanzahl\n"); return 1; }
    uint8_t *in = (uint8_t*)malloc(fg);
    if (fread(in, 1, fg, fp) != (size_t)fg) { fprintf(stderr, "read\n"); return 1; }
    fclose(fp);

    double t0 = jetzt();
    uint64_t n_high = (uint64_t)fg / 2;
    if (n_high % block) { fprintf(stderr, "n muss Vielfaches von block sein (Testharness)\n"); return 1; }
    uint8_t *high = (uint8_t*)malloc(n_high), *low = (uint8_t*)malloc(n_high);
    for (uint64_t i = 0; i < n_high; i++) { low[i] = in[2*i]; high[i] = in[2*i+1]; }

    uint64_t z[256] = {0}; for (uint64_t i = 0; i < n_high; i++) z[high[i]]++;
    uint16_t freq[256]; normiere(z, freq);
    baue_tabellen(freq);

    uint32_t n_bloecke = (uint32_t)(n_high / block);
    uint32_t *basen = (uint32_t*)malloc(4ull * n_bloecke);
    uint16_t *subrel = (uint16_t*)malloc(2ull * n_bloecke * LANES);
    size_t kap = n_high + n_bloecke * (size_t)LANES * 32 + 1024;
    uint8_t *daten = (uint8_t*)calloc(kap, 1);
    size_t pos = 0;
    uint8_t *tmp = (uint8_t*)malloc(2ull * block + 64), *symtmp = (uint8_t*)malloc(block);
    uint32_t n_lane = block / LANES;

    for (uint32_t b = 0; b < n_bloecke; b++) {
        uint64_t basis = (uint64_t)b * block;
        pos = (pos + ALIGN - 1) / ALIGN * ALIGN;
        basen[b] = (uint32_t)pos;
        for (uint32_t lane = 0; lane < LANES; lane++) {
            for (uint32_t i = 0; i < n_lane; i++) symtmp[i] = high[basis + pos_im_block(lane, i)];
            size_t len = kodiere_lane(symtmp, n_lane, tmp + 2ull * block + 64);
            pos = (pos + ALIGN - 1) / ALIGN * ALIGN;
            size_t rel = pos - basen[b];
            if (rel > 65535) { fprintf(stderr, "Block %u: Offset %zu > 65535\n", b, rel); return 1; }
            subrel[(uint64_t)b * LANES + lane] = (uint16_t)rel;
            memcpy(daten + pos, tmp + 2ull * block + 64 - len, len);
            pos += len;
        }
    }
    size_t daten_len = (pos + ALIGN - 1) / ALIGN * ALIGN + 16;
    double t1 = jetzt();

    uint8_t *pruef = (uint8_t*)malloc(n_high);
    for (uint32_t b = 0; b < n_bloecke; b++) {
        uint64_t basis = (uint64_t)b * block;
        for (uint32_t lane = 0; lane < LANES; lane++) {
            dekodiere_lane(daten + basen[b] + subrel[(uint64_t)b * LANES + lane], n_lane, symtmp);
            for (uint32_t i = 0; i < n_lane; i++) pruef[basis + pos_im_block(lane, i)] = symtmp[i];
        }
    }
    int ok = !memcmp(pruef, high, n_high);
    double t2 = jetzt();

    FILE *out = fopen(argv[2], "wb"); if (!out) { perror("out"); return 1; }
    uint8_t kopf[28] = { 'N','W','C','2', 5, LANES, TLOG, 0 };
    memcpy(kopf + 8, &n_high, 8);
    uint32_t stride = 0; memcpy(kopf + 16, &stride, 4);
    memcpy(kopf + 20, &block, 4); memcpy(kopf + 24, &n_bloecke, 4);
    fwrite(kopf, 1, 28, out);
    fwrite(freq, 2, 256, out);
    fwrite(basen, 4, n_bloecke, out);
    fwrite(subrel, 2, (size_t)n_bloecke * LANES, out);
    fwrite(daten, 1, daten_len, out);
    fwrite(low, 1, n_high, out);
    long ausgroesse = ftell(out);
    fclose(out);

    printf("%s -> %s  (Formatversion 5, tANS/FSE)\n", argv[1], argv[2]);
    printf("  %.1f MB -> %.1f MB   Rate %.4f   (block=%u, %u Bloecke)\n",
           fg / 1e6, ausgroesse / 1e6, (double)ausgroesse / fg, block, n_bloecke);
    printf("  kodieren %.2f s (%.1f MB/s), Selbsttest %.2f s: %s\n",
           t1 - t0, fg / 1e6 / (t1 - t0), t2 - t1, ok ? "bitexakt OK" : "FEHLER");
    return ok ? 0 : 1;
}
