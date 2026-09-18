/* NWC1-Dekoder in C -- misst realen Einzelkern-Durchsatz.
   Spiegelt exakt nwc.py/rans.py. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>

#define SCALE_BITS 12
#define TOT        (1u << SCALE_BITS)
#define RANS_L     (1u << 23)

static double jetzt(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "nutzung: nwcdec <datei.nwc> <referenz.raw>\n"); return 2; }

    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror("open"); return 1; }
    fseek(fp, 0, SEEK_END); long dateigroesse = ftell(fp); fseek(fp, 0, SEEK_SET);
    uint8_t *buf = (uint8_t*)malloc(dateigroesse);
    if (fread(buf, 1, dateigroesse, fp) != (size_t)dateigroesse) { fprintf(stderr,"read\n"); return 1; }
    fclose(fp);

    if (memcmp(buf, "NWC1", 4)) { fprintf(stderr, "kein NWC1\n"); return 1; }
    uint32_t n_streams = buf[5], skala = buf[6];
    uint64_t n_high; uint32_t stride, block, n_bloecke;
    memcpy(&n_high, buf + 8,  8);
    memcpy(&stride, buf + 16, 4);
    memcpy(&block,  buf + 20, 4);
    memcpy(&n_bloecke, buf + 24, 4);
    if (skala != SCALE_BITS) { fprintf(stderr, "skala != %d\n", SCALE_BITS); return 1; }

    const uint16_t *freq16 = (const uint16_t*)(buf + 28);
    const uint32_t *offsets = (const uint32_t*)(buf + 28 + 512);
    size_t daten_start = 28 + 512 + 4ull * n_bloecke;
    size_t low_start   = (size_t)dateigroesse - (size_t)n_high;

    /* Tabellen aufbauen */
    uint32_t freq[256], kum[257];
    kum[0] = 0;
    for (int i = 0; i < 256; i++) { freq[i] = freq16[i]; kum[i+1] = kum[i] + freq[i]; }
    uint8_t *slot2sym = (uint8_t*)malloc(TOT);
    for (int s = 0; s < 256; s++)
        for (uint32_t j = kum[s]; j < kum[s+1]; j++) slot2sym[j] = (uint8_t)s;

    uint8_t *high = (uint8_t*)malloc(n_high);
    uint8_t *aus  = (uint8_t*)malloc(2 * n_high);

    int wdh = 5;
    double beste = 1e30;
    for (int w = 0; w < wdh; w++) {
        double t0 = jetzt();

        /* --- Stufe A: rANS je Block (hier seriell; auf GPU ein Block pro Threadblock) --- */
        for (uint32_t b = 0; b < n_bloecke; b++) {
            const uint8_t *p = buf + daten_start + offsets[b];
            uint64_t basis = (uint64_t)b * block;
            uint64_t n_sym = n_high - basis; if (n_sym > block) n_sym = block;

            uint32_t zustand[64];
            for (int k = (int)n_streams - 1; k >= 0; k--) {
                zustand[k] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                             ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
                p += 4;
            }
            uint8_t *out = high + basis;
            uint32_t maske = n_streams - 1;          /* n_streams ist Zweierpotenz */
            for (uint64_t i = 0; i < n_sym; i++) {
                uint32_t k = (uint32_t)i & maske;
                uint32_t x = zustand[k];
                uint32_t slot = x & (TOT - 1);
                uint8_t s = slot2sym[slot];
                out[i] = s;
                x = freq[s] * (x >> SCALE_BITS) + slot - kum[s];
                while (x < RANS_L) x = (x << 8) | *p++;
                zustand[k] = x;
            }
        }

        /* --- Stufe B: XOR-Ketten (auf GPU: stride-fach parallel) --- */
        if (stride) for (uint64_t i = stride; i < n_high; i++) high[i] ^= high[i - stride];

        /* --- Stufe C: Ebenen verschraenken --- */
        const uint8_t *low = buf + low_start;
        for (uint64_t i = 0; i < n_high; i++) { aus[2*i] = low[i]; aus[2*i+1] = high[i]; }

        double dt = jetzt() - t0;
        if (dt < beste) beste = dt;
    }

    /* Verifikation */
    FILE *rf = fopen(argv[2], "rb");
    long refgroesse = 0;
    int ok = 0;
    if (rf) {
        fseek(rf, 0, SEEK_END); refgroesse = ftell(rf); fseek(rf, 0, SEEK_SET);
        uint8_t *ref = (uint8_t*)malloc(refgroesse);
        if (fread(ref, 1, refgroesse, rf) == (size_t)refgroesse)
            ok = (refgroesse == (long)(2*n_high)) && !memcmp(ref, aus, refgroesse);
        fclose(rf); free(ref);
    }

    double mb = (double)(2 * n_high) / 1e6;
    printf("  Ausgabe        : %.2f MB (%llu Gewichte, stride=%u, %u Bloecke)\n",
           mb, (unsigned long long)n_high, stride, n_bloecke);
    printf("  Verifikation   : %s\n", ok ? "bitexakt OK" : "FEHLER");
    printf("  Zeit (best v %d): %.4f s\n", wdh, beste);
    printf("  Durchsatz      : %.1f MB/s  (%.3f GB/s)\n", mb / beste, mb / beste / 1000.0);
    return ok ? 0 : 1;
}
