/* NWC2-Encoder in C.
   - Byte-Ebenen trennen, Exponentenebene mit rANS (16-Bit-Renormalisierung)
   - je Block LANES unabhaengige Teilstroeme, jeder auf ALIGN Bytes ausgerichtet
   - Zustand (u32) und Datenwoerter (u16) little-endian -> aligned loads im Dekoder
   - eingebauter Selbsttest: dekodiert und vergleicht bitexakt

   Symbolzuordnung (Formatversion):
     3: Lane L bekommt Symbole L, L+LANES, L+2*LANES, ...            (verschraenkt)
     4: Lane L bekommt je 512er-Chunk die 16 Symbole 16L .. 16L+15   (Chunk-Layout,
        Mantisse und Aktivierung je Lane als 16-Byte-Block ladbar; LANES muss 32 sein)

   nutzung: nwcenc <in.raw> <out.nwc> [block=8192] [lanes=32] [align=8] [version=4]     */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>

#define SCALE_BITS 12
#define TOT        (1u << SCALE_BITS)
#define L16        (1u << 16)

static double jetzt(void) {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

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
        if (diff > 0) { f[i]++; diff--; }
        else if (f[i] > 1) { f[i]--; diff++; }
    }
    for (int i = 0; i < 256; i++) freq[i] = (uint16_t)f[i];
}

/* Position des i-ten Symbols von Lane `lane` innerhalb des Blocks */
static inline uint32_t pos_im_block(uint32_t version, uint32_t lanes, uint32_t lane, uint32_t i) {
    if (version == 4) return (i >> 4) * 512 + lane * 16 + (i & 15);
    return lane + i * lanes;
}

static size_t kodiere_lane(const uint8_t *sym, uint32_t n, const uint16_t *freq, const uint32_t *kum, uint8_t *ende) {
    uint8_t *q = ende;
    uint32_t x = L16;
    for (int64_t i = (int64_t)n - 1; i >= 0; i--) {
        uint32_t s = sym[i], f = freq[s], x_max = f << 20;
        while (x >= x_max) { q -= 2; q[0] = (uint8_t)x; q[1] = (uint8_t)(x >> 8); x >>= 16; }
        x = ((x / f) << SCALE_BITS) + (x % f) + kum[s];
    }
    q -= 4; q[0] = (uint8_t)x; q[1] = (uint8_t)(x >> 8); q[2] = (uint8_t)(x >> 16); q[3] = (uint8_t)(x >> 24);
    return (size_t)(ende - q);
}

static void dekodiere_lane(const uint8_t *p, uint32_t n, const uint16_t *freq, const uint32_t *kum,
                           const uint8_t *slot2sym, uint8_t *out) {
    uint32_t x = p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); p += 4;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t slot = x & (TOT - 1); uint8_t s = slot2sym[slot];
        out[i] = s;
        x = freq[s] * (x >> SCALE_BITS) + slot - kum[s];
        if (x < L16) { x = (x << 16) | p[0] | (p[1] << 8); p += 2; }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "nutzung: nwcenc <in.raw> <out.nwc> [block] [lanes] [align] [version]\n"); return 2; }
    uint32_t block   = argc > 3 ? (uint32_t)atoi(argv[3]) : 8192;
    uint32_t lanes   = argc > 4 ? (uint32_t)atoi(argv[4]) : 32;
    uint32_t align   = argc > 5 ? (uint32_t)atoi(argv[5]) : 8;
    uint32_t version = argc > 6 ? (uint32_t)atoi(argv[6]) : 4;
    if (version == 4 && (lanes != 32 || block % 512)) { fprintf(stderr, "version 4: lanes=32, block Vielfaches von 512\n"); return 1; }

    FILE *fp = fopen(argv[1], "rb"); if (!fp) { perror("in"); return 1; }
    fseek(fp, 0, SEEK_END); long fg = ftell(fp); fseek(fp, 0, SEEK_SET);
    if (fg & 1) { fprintf(stderr, "ungerade Byteanzahl\n"); return 1; }
    uint8_t *in = (uint8_t*)malloc(fg);
    if (fread(in, 1, fg, fp) != (size_t)fg) { fprintf(stderr, "read\n"); return 1; }
    fclose(fp);

    double t0 = jetzt();
    uint64_t n_high = (uint64_t)fg / 2;
    uint8_t *high = (uint8_t*)malloc(n_high), *low = (uint8_t*)malloc(n_high);
    for (uint64_t i = 0; i < n_high; i++) { low[i] = in[2*i]; high[i] = in[2*i+1]; }

    uint64_t z[256] = {0}; for (uint64_t i = 0; i < n_high; i++) z[high[i]]++;
    uint16_t freq[256]; normiere(z, freq);
    uint32_t kum[257]; kum[0] = 0; for (int i = 0; i < 256; i++) kum[i+1] = kum[i] + freq[i];
    uint8_t *slot2sym = (uint8_t*)malloc(TOT);
    for (int s = 0; s < 256; s++) for (uint32_t j = kum[s]; j < kum[s+1]; j++) slot2sym[j] = (uint8_t)s;

    uint32_t n_bloecke = (uint32_t)((n_high + block - 1) / block);
    uint32_t *basen = (uint32_t*)malloc(4ull * n_bloecke);
    uint16_t *subrel = (uint16_t*)malloc(2ull * n_bloecke * lanes);
    size_t kap = n_high + n_bloecke * (size_t)lanes * (align + 8) + 1024;
    uint8_t *daten = (uint8_t*)calloc(kap, 1);
    size_t pos = 0;
    uint8_t *tmp = (uint8_t*)malloc(2ull * block + 16), *symtmp = (uint8_t*)malloc(block + 16);

    for (uint32_t b = 0; b < n_bloecke; b++) {
        uint64_t basis = (uint64_t)b * block;
        uint32_t n_sym = (uint32_t)((n_high - basis > block) ? block : (n_high - basis));
        pos = (pos + align - 1) / align * align;
        basen[b] = (uint32_t)pos;
        for (uint32_t lane = 0; lane < lanes; lane++) {
            uint32_t n_lane = 0;
            for (uint32_t i = 0; ; i++) {                     /* Symbole der Lane einsammeln */
                uint32_t q = pos_im_block(version, lanes, lane, i);
                if (q >= n_sym) { if (version == 4 && (i & 15) != 15 && (i >> 4) * 512 < n_sym) continue; break; }
                symtmp[n_lane++] = high[basis + q];
                if (n_lane >= block) break;
            }
            size_t len = kodiere_lane(symtmp, n_lane, freq, kum, tmp + 2ull * block + 16);
            pos = (pos + align - 1) / align * align;
            size_t rel = pos - basen[b];
            if (rel > 65535) { fprintf(stderr, "Block %u: Offset %zu > 65535\n", b, rel); return 1; }
            subrel[(uint64_t)b * lanes + lane] = (uint16_t)rel;
            memcpy(daten + pos, tmp + 2ull * block + 16 - len, len);
            pos += len;
        }
    }
    size_t daten_len = (pos + align - 1) / align * align;
    double t1 = jetzt();

    /* Selbsttest */
    uint8_t *pruef = (uint8_t*)malloc(n_high); memset(pruef, 0, n_high);
    for (uint32_t b = 0; b < n_bloecke; b++) {
        uint64_t basis = (uint64_t)b * block;
        uint32_t n_sym = (uint32_t)((n_high - basis > block) ? block : (n_high - basis));
        for (uint32_t lane = 0; lane < lanes; lane++) {
            uint32_t n_lane = 0, idx[8192 + 16];
            for (uint32_t i = 0; ; i++) {
                uint32_t q = pos_im_block(version, lanes, lane, i);
                if (q >= n_sym) { if (version == 4 && (i & 15) != 15 && (i >> 4) * 512 < n_sym) continue; break; }
                idx[n_lane++] = q;
                if (n_lane >= block) break;
            }
            if (!n_lane) continue;
            dekodiere_lane(daten + basen[b] + subrel[(uint64_t)b * lanes + lane], n_lane, freq, kum, slot2sym, symtmp);
            for (uint32_t i = 0; i < n_lane; i++) pruef[basis + idx[i]] = symtmp[i];
        }
    }
    int ok = !memcmp(pruef, high, n_high);
    double t2 = jetzt();

    FILE *out = fopen(argv[2], "wb"); if (!out) { perror("out"); return 1; }
    uint8_t kopf[28] = { 'N','W','C','2', (uint8_t)version, (uint8_t)lanes, SCALE_BITS, 16 };
    memcpy(kopf + 8, &n_high, 8);
    uint32_t stride = 0; memcpy(kopf + 16, &stride, 4);
    memcpy(kopf + 20, &block, 4); memcpy(kopf + 24, &n_bloecke, 4);
    fwrite(kopf, 1, 28, out);
    fwrite(freq, 2, 256, out);
    fwrite(basen, 4, n_bloecke, out);
    fwrite(subrel, 2, (size_t)n_bloecke * lanes, out);
    fwrite(daten, 1, daten_len, out);
    fwrite(low, 1, n_high, out);
    long ausgroesse = ftell(out);
    fclose(out);

    printf("%s -> %s  (Formatversion %u)\n", argv[1], argv[2], version);
    printf("  %.1f MB -> %.1f MB   Rate %.4f   (block=%u lanes=%u align=%u, %u Bloecke)\n",
           fg / 1e6, ausgroesse / 1e6, (double)ausgroesse / fg, block, lanes, align, n_bloecke);
    printf("  kodieren %.2f s (%.1f MB/s), Selbsttest %.2f s: %s\n",
           t1 - t0, fg / 1e6 / (t1 - t0), t2 - t1, ok ? "bitexakt OK" : "FEHLER");
    return ok ? 0 : 1;
}
