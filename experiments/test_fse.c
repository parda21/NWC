/* Minimaltest fuer die FSE-Kodierung: eine Lane, erste Abweichung ausgeben. */
#define main nwcenc_main
#include "nwcenc_fse.c"
#undef main

int main(void) {
    static uint8_t sym[256], out[256], puf[8192];
    uint64_t z[256] = {0};
    /* schiefe Verteilung wie Exponentenbytes */
    uint32_t seed = 7;
    for (int i = 0; i < 256; i++) {
        seed = seed * 1664525u + 1013904223u;
        uint32_t r = (seed >> 8) % 100;
        sym[i] = (uint8_t)(r < 50 ? 60 : r < 80 ? 61 : r < 95 ? 62 : r < 99 ? 59 : 63);
    }
    for (int i = 0; i < 256; i++) z[sym[i]]++;
    z[7] = 1;                                  /* ein seltenes Symbol mit freq 1 erzwingen */
    uint16_t freq[256]; normiere(z, freq);
    baue_tabellen(freq);

    for (int n = 1; n <= 256; n *= 2) {
        size_t len = kodiere_lane(sym, n, puf + 8192);
        dekodiere_lane(puf + 8192 - len, n, out);
        int erste = -1;
        for (int i = 0; i < n; i++) if (out[i] != sym[i]) { erste = i; break; }
        printf("n=%3d  len=%3zu  %s", n, len, erste < 0 ? "OK" : "FEHLER ab Index");
        if (erste >= 0) printf(" %d  (soll %d, ist %d)", erste, sym[erste], out[erste]);
        printf("\n");
    }
    return 0;
}
