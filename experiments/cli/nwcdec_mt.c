/* NWC1-Dekoder, mehrkernig. Zeigt, ob die Blockunabhaengigkeit wirklich traegt. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define SCALE_BITS 12
#define TOT        (1u << SCALE_BITS)
#define RANS_L     (1u << 23)

static double jetzt(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

static uint32_t freq[256], kum[257];
static uint8_t *slot2sym, *high, *aus;

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "nutzung: %s <datei.nwc> <referenz.raw>\n", argv[0]); return 2; }

    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror("open"); return 1; }
    fseek(fp, 0, SEEK_END); long dateigroesse = ftell(fp); fseek(fp, 0, SEEK_SET);
    uint8_t *buf = (uint8_t*)malloc(dateigroesse);
    if (fread(buf, 1, dateigroesse, fp) != (size_t)dateigroesse) { fprintf(stderr, "read\n"); return 1; }
    fclose(fp);

    if (memcmp(buf, "NWC1", 4)) { fprintf(stderr, "kein NWC1\n"); return 1; }
    uint32_t n_streams = buf[5], skala = buf[6];
    uint64_t n_high; uint32_t stride, block, n_bloecke;
    memcpy(&n_high, buf + 8, 8);
    memcpy(&stride, buf + 16, 4);
    memcpy(&block,  buf + 20, 4);
    memcpy(&n_bloecke, buf + 24, 4);
    if (skala != SCALE_BITS) { fprintf(stderr, "skala != %d\n", SCALE_BITS); return 1; }

    const uint16_t *freq16 = (const uint16_t*)(buf + 28);
    const uint32_t *offsets = (const uint32_t*)(buf + 28 + 512);
    size_t daten_start = 28 + 512 + 4ull * n_bloecke;
    size_t low_start   = (size_t)dateigroesse - (size_t)n_high;

    kum[0] = 0;
    for (int i = 0; i < 256; i++) { freq[i] = freq16[i]; kum[i+1] = kum[i] + freq[i]; }
    slot2sym = (uint8_t*)malloc(TOT);
    for (int s = 0; s < 256; s++)
        for (uint32_t j = kum[s]; j < kum[s+1]; j++) slot2sym[j] = (uint8_t)s;

    high = (uint8_t*)malloc(n_high);
    aus  = (uint8_t*)malloc(2 * n_high);
    const uint8_t *low = buf + low_start;
    const int nb = (int)n_bloecke, ns = (int)n_streams, sd = (int)stride;
    const int nh = (int)n_high;

    double beste = 1e30;
    for (int w = 0; w < 5; w++) {
        double t0 = jetzt();

        /* Stufe A -- ein Block pro Thread, keine gemeinsamen Daten */
        int b;
        #pragma omp parallel for schedule(dynamic)
        for (b = 0; b < nb; b++) {
            const uint8_t *p = buf + daten_start + offsets[b];
            uint64_t basis = (uint64_t)b * block;
            uint64_t n_sym = n_high - basis; if (n_sym > block) n_sym = block;
            uint32_t zustand[64];
            for (int k = ns - 1; k >= 0; k--) {
                zustand[k] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                             ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
                p += 4;
            }
            uint8_t *out = high + basis;
            uint32_t maske = (uint32_t)ns - 1;
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

        /* Stufe B -- eine XOR-Kette je Restklasse mod stride */
        if (sd) {
            int r;
            #pragma omp parallel for schedule(static)
            for (r = 0; r < sd; r++) {
                int i;
                for (i = r + sd; i < nh; i += sd) high[i] ^= high[i - sd];
            }
        }

        /* Stufe C -- Ebenen verschraenken */
        int i;
        #pragma omp parallel for schedule(static)
        for (i = 0; i < nh; i++) { aus[2*i] = low[i]; aus[2*i+1] = high[i]; }

        double dt = jetzt() - t0;
        if (dt < beste) beste = dt;
    }

    FILE *rf = fopen(argv[2], "rb");
    int ok = 0;
    if (rf) {
        fseek(rf, 0, SEEK_END); long rg = ftell(rf); fseek(rf, 0, SEEK_SET);
        uint8_t *ref = (uint8_t*)malloc(rg);
        if (fread(ref, 1, rg, rf) == (size_t)rg)
            ok = (rg == (long)(2*n_high)) && !memcmp(ref, aus, rg);
        fclose(rf); free(ref);
    }

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    double mb = (double)(2 * n_high) / 1e6;
    printf("  Threads        : %d\n", threads);
    printf("  Verifikation   : %s\n", ok ? "bitexakt OK" : "FEHLER");
    printf("  Durchsatz      : %.1f MB/s  (%.3f GB/s)\n", mb / beste, mb / beste / 1000.0);
    return ok ? 0 : 1;
}
