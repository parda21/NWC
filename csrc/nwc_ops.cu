/* nwc_ops -- NWC (Neural Weight Compression) for PyTorch via ctypes (Windows: nwc_ops.dll, Linux: nwc_ops.so).
   Format NWC2 version 9 (prefix code), any matrix shape M x K:
     - BF16 split into byte planes: mantissa raw (low[], row-major, row stride K16 = K rounded up to 16);
       exponent byte prefix-coded: rank r of the 7-bit exponent (by frequency, table exp_of_rank[16]) as
       r zeros and a one, then the raw sign bit; rank >= 15: escape = 15 zeros, a one, the raw byte (24 bits).
     - Block = 8 rows x 512 columns = 4096 weights = 32 lanes; lane L owns columns 16L..16L+15 of each of the
       8 rows (order: row 0..7, columns ascending). The matrix is conceptually padded to Mp = M rounded up to 8
       and Kp = K rounded up to 512; filler weights (row >= M or column >= K) cost 2 bits in the exponent stream
       (rank-0 code) and have no mantissa (the decoder loads nothing there and multiplies by x = 0).
       Blocks nb = (Mp/8) * (Kp/512), row-major.
     - Block layout [32 lane streams, each a multiple of 4 B], header 1 byte per lane (length in words),
       blocks 8-byte aligned (bases[]), lane offsets by warp prefix sum.
     - Decoder: 12-bit peek into a LUT (4096 x u32 in shared memory): [c | by0<<8 | by1<<16] = two complete
       codes (c = bits consumed, in byte 0 so that shf.l.wrap takes it directly as the shift amount), otherwise
       FLAG(bit 30) | c1 | by0<<8 | n<<24 -> slow path via clz. Window hi:lo (64 bits, MSB first) shifted with
       funnel shifts; fill level as a rotation M = rotl(M, c) (M = 2^(64-nb) mod 2^32): wrap-around M < M_old
       <=> refill (one pre-loaded 32-bit word). Weight from exponent byte + mantissa byte with one PRMT, zero
       bytes by sign replication. Token path: persistent blocks (1024 threads, one per SM), x fp32 in 16
       registers per lane, 8 row accumulators, transposed shuffle reduction, partial sums
       partial[row * (Kp/512) + column_block]. Cost model and measurements: docs/results.md, docs/format.md.
   Element types (parameter elem on every entry point): 0 = BF16 as above; 1 = FP8 e4m3 (weight-only, per-row scale
   applied in the finalize kernel): the exponent is halved into a 10-symbol alphabet (8 exponent pairs, small
   subnormals, zero; no escape), the low exponent bit and the 3 mantissa bits form a 4-bit raw nibble plane (row stride
   K16/2 bytes, nibble order per 16 weights chosen so one shift + one mask expands a word into 4 weight bytes). The
   decoder is the BF16 decoder: the LUT byte is the fp32 exponent >> 1, the nibble << 4 is byte 2 of the fp32 weight.
   Dequantization yields the unscaled fp8 values as BF16 (exact).
   Exports:
     nwc_info        host. 0: format version, 1: states per lane (1), 2: table words (u32),
                     3: header bytes per block (32), 4: block size (4096, fixed), 5: elem parameter present (1)
     nwc_layout      host. (M, K, elem) -> [blocks, bytes of the raw plane, floats of the scratch buffer, K16]
     nwc_encode      host. BF16 or FP8 bytes (n = M*K, K, elem) -> psym[0..15] = exp_of_rank, bases, hdr, data, low;
                     freq is unused. Returns the data length, -1 on bad shape / capacity, -2 on NaN in FP8.
     nwc_build_tab   host. psym (exp_of_rank), elem -> tab[4096 u32] (LUT)
     nwc_matvec      device, fp32 test path, y = W x via atomics; x fp32[K16]
     nwc_linear_bf16 device. y = scale * (W x) + b, x fp32[K16] (zeros beyond K), y BF16[M], decoded in registers
     nwc_dequant     device. compressed -> BF16[M x K16] (row-major, stride K16); needs K
     nwc_gather      device. rows ids[] -> BF16[n_ids x K16] (embedding lookup)
     nwc_ref_fp8     device. reference fp8 weight-only matvec (bandwidth baseline), y = scale * (W8 x)
     nwc_setup       host. SM count (called on load)
   Earlier formats: v8 (rANS, pair symbols) in experiments/nwc_ops_v8.cu, v7 in experiments/nwc_ops_v7.cu. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define VERSION    9
#define LANES      32u
#define ROWS       8u
#define COLS       512u
#define BLOCK      (ROWS * COLS)
#define HDR_BLOCK  LANES                 /* 1 byte per lane: stream length in 32-bit words */
#define ALIGN      8u
#define PEEK       12
#define LUTN       (1u << PEEK)
#define TAB_WORDS  LUTN
#define MAXR       15u                   /* BF16: ranks 0..14 unary, escape from 15 */
#define MAXR_FP8   10u                   /* FP8: 10 symbols, ranks 0..9 unary, no escape */
#define ELEM_BF16  0
#define ELEM_FP8   1                     /* fp8 e4m3 weights: 10-symbol code + 4-bit raw nibble plane */
#define FLAG       (1u << 30)
#define TBP        1024u                 /* threads per persistent block */
#define NPAIR      1024u                 /* size of freq/psym (Python bridge, compatible with v8) */
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

/* Rounded-up dimensions of an M x K matrix (identical on host and device) */
struct Shape {
    uint32_t M, K, K16, Kp, CB, Mp, nb;
    __host__ __device__ Shape(uint32_t M_, uint32_t K_) : M(M_), K(K_) {
        K16 = (K + 15u) & ~15u; Kp = (K + COLS - 1) / COLS * COLS; CB = Kp / COLS; Mp = (M + ROWS - 1) / ROWS * ROWS;
        nb = (Mp / ROWS) * CB;
    }
};

EXPORT int nwc_info(int what) {
    switch (what) { case 0: return VERSION; case 1: return 1; case 2: return (int)TAB_WORDS; case 3: return (int)HDR_BLOCK;
                    case 4: return (int)BLOCK; case 5: return 1; /* elem parameter (0 BF16, 1 FP8) on all entry points */ }
    return -1;
}
/* bytes of the raw plane: BF16 one mantissa byte per weight, FP8 one nibble (row stride K16/2 bytes) */
static uint64_t raw_bytes(const Shape &f, int elem) { return elem == ELEM_FP8 ? (uint64_t)f.M * (f.K16 / 2) : (uint64_t)f.M * f.K16; }
EXPORT int nwc_layout(uint64_t M, uint32_t K, int elem, uint64_t *out) {
    if (M == 0 || K == 0 || M > 0xFFFFFFFFull) return -1;
    Shape f((uint32_t)M, K);
    out[0] = f.nb; out[1] = raw_bytes(f, elem); out[2] = (uint64_t)f.Mp * f.CB; out[3] = f.K16;
    return 0;
}

/* ============================ Host: encoder ============================ */
struct BitWriter {                                        /* MSB-first bit writer into u32 words */
    uint32_t *w; uint32_t n_w; uint64_t acc; int n;
    void push(uint32_t v, int nb) { acc = (acc << nb) | v; n += nb; while (n >= 32) { w[n_w++] = (uint32_t)(acc >> (n - 32)); n -= 32; } }
    void finish() { if (n) { w[n_w++] = (uint32_t)(acc << (32 - n)); n = 0; } }
};

/* FP8 e4m3 byte -> (7-bit "exponent byte" the decoder gets from the LUT, 4-bit nibble for the raw plane).
   The decoded fp32 is [s | byte3 | nibble<<4 | 0 | 0]: byte3 = (fp32 exponent >> 1), nibble = (fp32 exponent & 1)<<3 | top
   3 mantissa bits. Normals: exponent e -> fp32 exponent e+120, symbol e>>1 (byte3 = 60 + (e>>1)). Subnormals m*2^-9
   with m >= 4 fit symbol 0 (fp32 exponent 120), m = 1..3 get symbol 8 (byte3 = 59, exponents 118/119), zero is
   symbol 9 (byte3 = 0). NaN (e = 15, m = 7) is rejected: the model would be broken anyway. */
static const uint8_t FP8_BYTE3[10] = {60, 61, 62, 63, 64, 65, 66, 67, 59, 0};
static int fp8_split(uint8_t b, uint8_t *sym, uint8_t *nib) {
    uint32_t e = (b >> 3) & 15u, m = b & 7u;
    if (e >= 1) { if (e == 15 && m == 7) return -1; *sym = (uint8_t)(e >> 1); *nib = (uint8_t)(((e & 1) << 3) | m); return 0; }
    if (m >= 4) { *sym = 0; *nib = (uint8_t)((m - 4) << 1); return 0; }               /* (1 + (m-4)/4) * 2^-7 = m * 2^-9 */
    if (m == 0) { *sym = 9; *nib = 0; return 0; }
    *sym = 8; *nib = (m == 1) ? 0 : (m == 2) ? 8 : 12;                                /* 2^-9, 2^-8, 1.5 * 2^-8 */
    return 0;
}
/* position of column j (0..15) of a lane's 16 weights inside its 8 nibble bytes: byte, high nibble? The order lets the
   decoder expand a raw word into 4 weight bytes with one shift and one mask (see raw_chunk). */
static inline uint32_t nib_byte(uint32_t j) { return (j & 3u) + ((j >> 3) << 2); }
static inline uint32_t nib_high(uint32_t j) { return (j >> 2) & 1u; }

EXPORT int64_t nwc_encode(const void *wv, uint64_t n, uint32_t K, int elem, uint16_t *freq, uint16_t *psym,
                          uint32_t *bases, uint8_t *hdr, uint8_t *data, uint64_t data_cap, uint8_t *low)
{
    if (K == 0 || n % K || n / K > 0xFFFFFFFFull) return -1;
    Shape f((uint32_t)(n / K), K);
    const int NSYM = elem == ELEM_FP8 ? 10 : 128, MR = elem == ELEM_FP8 ? (int)MAXR_FP8 : (int)MAXR;
    uint8_t *high = (uint8_t*)malloc(n);                                   /* per weight: symbol (bit 7 = sign) */
    if (elem == ELEM_FP8) {
        const uint8_t *w = (const uint8_t*)wv;
        memset(low, 0, raw_bytes(f, elem));
        for (uint32_t m = 0; m < f.M; m++) for (uint32_t k = 0; k < f.K; k++) {
            uint8_t b = w[(uint64_t)m * K + k], sym, nib;
            if (fp8_split(b, &sym, &nib)) { free(high); return -2; }
            high[(uint64_t)m * K + k] = (uint8_t)((b & 0x80u) | sym);
            uint64_t o = (uint64_t)m * (f.K16 / 2) + (k / 16) * 8 + nib_byte(k & 15);
            low[o] |= (uint8_t)(nib << (nib_high(k & 15) ? 4 : 0));
        }
    } else {
        const uint16_t *w = (const uint16_t*)wv;
        for (uint32_t m = 0; m < f.M; m++) {                               /* mantissa plane with stride K16 */
            for (uint32_t k = 0; k < f.K; k++) { uint16_t v = w[(uint64_t)m * K + k]; low[(uint64_t)m * f.K16 + k] = (uint8_t)v; high[(uint64_t)m * K + k] = (uint8_t)(v >> 8); }
            for (uint32_t k = f.K; k < f.K16; k++) low[(uint64_t)m * f.K16 + k] = 0;
        }
    }
    /* ranks of the symbols by frequency */
    uint64_t hist[128] = {0}; for (uint64_t i = 0; i < n; i++) hist[high[i] & 0x7F]++;
    int ord[128]; for (int i = 0; i < 128; i++) ord[i] = i;
    for (int a = 0; a < NSYM; a++) for (int b = a + 1; b < NSYM; b++) if (hist[ord[b]] > hist[ord[a]]) { int t = ord[a]; ord[a] = ord[b]; ord[b] = t; }
    uint8_t rank_of_exp[128];
    for (int r = 0; r < 128; r++) rank_of_exp[ord[r]] = (uint8_t)r;
    for (uint32_t i = 0; i < NPAIR; i++) {
        freq[i] = 0;
        psym[i] = (i < 16) ? (uint16_t)(elem == ELEM_FP8 ? (i < 10 ? FP8_BYTE3[ord[i]] : 0) : ord[i]) : 0;
    }

    uint32_t words[BLOCK / LANES + 2];
    uint64_t pos = 0; int64_t rc = -1;
    for (uint32_t b = 0; b < f.nb; b++) {
        const uint32_t rb = b / f.CB, cb = b % f.CB;
        pos = (pos + ALIGN - 1) / ALIGN * ALIGN;
        bases[b] = (uint32_t)pos;
        for (uint32_t L = 0; L < LANES; L++) {
            BitWriter bw; bw.w = words; bw.n_w = 0; bw.acc = 0; bw.n = 0;
            for (uint32_t i = 0; i < BLOCK / LANES; i++) {
                uint32_t row = rb * ROWS + (i >> 4), col = cb * COLS + L * 16 + (i & 15);
                if (row >= f.M || col >= f.K) { bw.push(2, 2); continue; }               /* filler: rank 0, sign 0 */
                uint8_t e = high[(uint64_t)row * K + col]; uint32_t r = rank_of_exp[e & 0x7F], s = e >> 7;
                if ((int)r < MR) { bw.push(1, r + 1); bw.push(s, 1); }
                else { bw.push(1, 16); bw.push(e, 8); }                                     /* BF16 only */
            }
            bw.finish();
            if (bw.n_w > 255 || pos + 4ull * bw.n_w + 64 > data_cap) goto done;
            hdr[(uint64_t)b * HDR_BLOCK + L] = (uint8_t)bw.n_w;
            memcpy(data + pos, words, 4ull * bw.n_w); pos += 4ull * bw.n_w;   /* little-endian, as ld.u32 reads it */
        }
    }
    rc = (int64_t)((pos + ALIGN - 1) / ALIGN * ALIGN + 64);                    /* 64 B slack: the window reads ahead */
done:
    free(high);
    return rc;
}

static uint32_t clz32(uint32_t x) { uint32_t u = 0; if (!x) return 32; while (!(x & 0x80000000u)) { x <<= 1; u++; } return u; }
/* LUT: for every 12-bit pattern the first <= 2 complete codes */
EXPORT void nwc_build_tab(const uint16_t *freq, const uint16_t *psym, int elem, uint32_t *lut) {
    (void)freq;
    const uint32_t MR = elem == ELEM_FP8 ? MAXR_FP8 : MAXR;
    for (uint32_t pat = 0; pat < LUTN; pat++) {
        uint32_t bits = pat << (32 - PEEK), pos = 0, n = 0, c1 = 0, by[2] = {0, 0};
        for (int k = 0; k < 2; k++) {
            uint32_t rest = PEEK - pos; if (rest == 0) break;
            uint32_t x = bits << pos, u = clz32(x);
            if (u >= MR || u + 2 > rest) break;
            uint32_t s = (x >> (30 - u)) & 1;
            by[k] = (s << 7) | (psym[u] & 0x7F); pos += u + 2; n++; if (k == 0) c1 = pos;
        }
        if (n == 2) lut[pat] = pos | (by[0] << 8) | (by[1] << 16);
        else        lut[pat] = FLAG | (n ? c1 : 0u) | (by[0] << 8) | (n << 24);
    }
    /* patterns 0 and 1 never hold a complete code: they carry exp_of_rank[11..14] (ranks 0..10 are read back
       from the patterns "r zeros, one"), so no second table is needed */
    lut[0] = FLAG | ((psym[11] & 0x7Fu) << 8) | ((psym[12] & 0x7Fu) << 16);
    lut[1] = FLAG | ((psym[13] & 0x7Fu) << 8) | ((psym[14] & 0x7Fu) << 16);
}

/* ============================ Device: decoder ============================ */
struct Window {
    uint64_t hl, p; uint32_t M, w;
    __device__ __forceinline__ uint32_t hi() const { return (uint32_t)(hl >> 32); }
    __device__ __forceinline__ void init(const uint32_t *start) {
        hl = ((uint64_t)__ldg(start) << 32) | __ldg(start + 1); w = __ldg(start + 2); M = 1; p = (uint64_t)(start + 3);
    }
    /* advance by c bits (c = low 5 bits of e) and refill */
    __device__ __forceinline__ void advance(uint32_t e) {
        uint32_t M_old = M, nh, nl, lo = (uint32_t)hl, h = hi();
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(nh) : "r"(lo), "r"(h), "r"(e));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(nl) : "r"(0u), "r"(lo), "r"(e));
        asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(M) : "r"(M_old), "r"(e));
        uint32_t mh = (M < M_old) ? M : 0u;                          /* 2^(32-nb) on wrap-around, else 0 */
        hl = ((uint64_t)nh << 32) | nl;
        hl = (uint64_t)w * mh + hl;                                  /* lo is 0 on wrap-around: inserts the word */
#ifndef KO_NOREFILL
        if (M < M_old) { asm("ld.global.nc.u32 %0, [%1];" : "=r"(w) : "l"(p)); p += 4; }
#else
        if (M < M_old) { w ^= (uint32_t)p; p += 4; }                  /* knockout: no refill load (timing experiments) */
#endif
    }
    __device__ __forceinline__ uint32_t peek4(uint32_t sh) const { uint32_t i; asm("shr.u32 %0, %1, %2;" : "=r"(i) : "r"(hi()), "r"(sh)); return i * 4u; }
    template<uint32_t MR>
    __device__ __forceinline__ uint32_t decode_slow(const uint8_t *exp_of_rank) {   /* one code via clz */
        uint32_t h = hi(), u = __clz(h);
        if (MR == MAXR_FP8 || u < MR) { uint32_t s = (h >> (30 - u)) & 1; uint32_t b = (s << 7) | exp_of_rank[u]; advance(u + 2); return b; }
        advance(16); uint32_t b = hi() >> 24; advance(8); return b;                 /* BF16 escape */
    }
    /* decode one pair: returns e = [. | by0 | by1 | 0] (bytes 1/2 = exponent bytes, byte 3 = 0) */
    template<uint32_t MR>
    __device__ __forceinline__ uint32_t pair(uint32_t sm_lut, uint32_t sh, const uint8_t *exp_of_rank) {
        uint32_t e;
asm("ld.shared.u32 %0, [%1];" : "=r"(e) : "r"(sm_lut + peek4(sh)));
        if (e < FLAG) advance(e);
        else {
            uint32_t n = e >> 24 & 1, b0 = (e >> 8) & 0xFFu, b1;
            if (n) advance(e);
            if (!n) b0 = decode_slow<MR>(exp_of_rank);
            b1 = decode_slow<MR>(exp_of_rank);
            e = (b0 << 8) | (b1 << 16);
        }
        return e;
    }
    /* set up the lane stream of a block (offsets by warp prefix sum), prefetch the block's data into L1 */
    __device__ __forceinline__ void setup(const uint8_t *data, const uint32_t *bases, const uint8_t *hdr, uint32_t b, uint32_t lane) {
        uint32_t l4 = __ldg(hdr + (uint64_t)b * HDR_BLOCK + lane), xx = l4;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { uint32_t y = __shfl_up_sync(0xffffffffu, xx, o); if (lane >= (uint32_t)o) xx += y; }
        const uint8_t *bs = data + bases[b];
        init((const uint32_t*)bs + (xx - l4));
        uint32_t words = __shfl_sync(0xffffffffu, xx, 31);
        if (lane * 32 < words) asm volatile("prefetch.global.L1 [%0];" :: "l"(bs + lane * 128));
    }
};

/* LUT into shared memory, followed by exp_of_rank[16] (reconstructed from the LUT) */
__device__ __forceinline__ void load_lut(uint32_t *smem, const uint32_t *__restrict__ lut, uint32_t tid, uint32_t nth) {
    for (uint32_t i = tid; i < LUTN; i += nth) smem[i] = __ldg(lut + i);
    if (tid < 16) {
        uint8_t *exp_of_rank = (uint8_t*)(smem + LUTN);
        uint32_t e = (tid < 11) ? __ldg(lut + (1u << (PEEK - 1 - tid))) : (tid < 15) ? __ldg(lut + ((tid - 11) >> 1)) : 0u;
        exp_of_rank[tid] = (uint8_t)((e >> ((tid >= 11 && !(tid & 1)) ? 16 : 8)) & 0x7Fu);   /* 11,13: byte 1; 12,14: byte 2 */
    }
    __syncthreads();
}

/* PRMT with sign replication: nibble 0xF = byte 7 (e byte 3, MSB 0) -> 0x00 */
__device__ __forceinline__ float weight_f32(uint32_t cw, uint32_t e, uint32_t sel) {
    uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(cw), "r"(e), "r"(sel)); return __uint_as_float(r);
}
/* two BF16 as u32: [m0][by0][m1][by1] */
__device__ __forceinline__ uint32_t two_bf16(uint32_t cw, uint32_t e, int jj) {
    return __byte_perm(cw, e, (6 << 12) | (((jj & 3) + 1) << 8) | (5 << 4) | (jj & 3));
}
__device__ __forceinline__ uint32_t u4_word(const uint4 &c, int jj) { return (jj < 4) ? c.x : (jj < 8) ? c.y : (jj < 12) ? c.z : c.w; }
/* the raw bytes (byte 2 of the fp32 weight) of a lane's 16 weights (row, column col0), zeros outside the matrix.
   BF16: 16 mantissa bytes. FP8: 8 nibble bytes, expanded so that word j>>2, byte j&3 holds weight j's nibble << 4
   (nibble order chosen by the encoder: word 0 low nibbles = weights 0..3, high = 4..7, word 1 likewise 8..15). */
template<int ELEM>
__device__ __forceinline__ uint4 raw_chunk(const uint8_t *__restrict__ low, const Shape &f, uint32_t row, uint32_t col0, bool col_ok) {
#ifdef KO_NOLOW
    if (row < f.M && col_ok) return make_uint4(0x12345678u ^ row, 0x9abcdef0u ^ col0, 0x0f0f0f0fu, 0xf0f0f0f0u);   /* knockout: no raw loads */
#endif
    if (row < f.M && col_ok) {
        if (ELEM == ELEM_FP8) {
            uint2 c = __ldcs((const uint2*)(low + (uint64_t)row * (f.K16 / 2) + col0 / 2));
            return make_uint4((c.x << 4) & 0xF0F0F0F0u, c.x & 0xF0F0F0F0u, (c.y << 4) & 0xF0F0F0F0u, c.y & 0xF0F0F0F0u);
        }
        return __ldcs((const uint4*)(low + (uint64_t)row * f.K16 + col0));
    }
    return make_uint4(0u, 0u, 0u, 0u);
}

/* Persistent block kernel. MODE 0: partial sums partial[row*CB + cb] (x fp32), MODE 1: dequantization
   (BF16 row-major to out, stride K16), MODE 2: fp32 matvec via atomics on y (tests) */
template<int MODE, int ELEM>
__global__ void __launch_bounds__(TBP, 1)
kern_block(const uint8_t *__restrict__ data, const uint32_t *__restrict__ bases, const uint8_t *__restrict__ hdr,
           const uint32_t *__restrict__ lut, const uint8_t *__restrict__ low, const float *__restrict__ x,
           float *__restrict__ partial, uint16_t *__restrict__ out, uint32_t M, uint32_t K, uint32_t sh)
{
    extern __shared__ uint32_t smem[];                          /* [LUT 16 KB][exp_of_rank 16 B] */
    load_lut(smem, lut, threadIdx.x, TBP);
    const uint8_t *exp_of_rank = (const uint8_t*)(smem + LUTN);
    const uint32_t sm_lut = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t lane = threadIdx.x & 31;
    const Shape f(M, K);

    for (uint32_t b = blockIdx.x * (TBP / 32) + (threadIdx.x >> 5); b < f.nb; b += gridDim.x * (TBP / 32)) {
        const uint32_t rb = b / f.CB, cb = b - rb * f.CB, col0 = cb * COLS + lane * 16, row0 = rb * ROWS;
        const bool col_ok = col0 < f.K16;                        /* lane inside the matrix (rounded up to 16) */
        Window W; W.setup(data, bases, hdr, b, lane);
#ifndef PF_ROWS_L1
#define PF_ROWS_L1 0
#endif
#ifndef PF_NEXT_L2
#define PF_NEXT_L2 0
#endif
        /* prefetch experiments (RTX 4070: neither helps, the L2 one costs 5-10 %; kept as compile switches):
           this block's raw-plane rows into L1, the next block of this warp (stream + raw rows) into L2 */
        const uint32_t rstride = (ELEM == ELEM_FP8) ? f.K16 / 2 : f.K16, coff = (ELEM == ELEM_FP8) ? col0 / 2 : col0;
        if (PF_ROWS_L1 && col_ok) {
            #pragma unroll
            for (int r = 0; r < (int)ROWS; r++)
                if (row0 + r < f.M) asm volatile("prefetch.global.L1 [%0];" :: "l"(low + (uint64_t)(row0 + r) * rstride + coff));
        }
        if (PF_NEXT_L2) {
            const uint32_t bn = b + gridDim.x * (TBP / 32);
            if (bn < f.nb) {
                asm volatile("prefetch.global.L2 [%0];" :: "l"(data + __ldg(bases + bn) + lane * 128));
                const uint32_t rbn = bn / f.CB, cbn = bn - rbn * f.CB, coln = cbn * COLS + lane * 16;
                if (coln < f.K16) {
                    const uint8_t *pn = low + (uint64_t)(rbn * ROWS) * rstride + ((ELEM == ELEM_FP8) ? coln / 2 : coln);
                    #pragma unroll
                    for (int r = 0; r < (int)ROWS; r++)
                        if (rbn * ROWS + r < f.M) asm volatile("prefetch.global.L2 [%0];" :: "l"(pn + (uint64_t)r * rstride));
                }
            }
        }
        float xw[16];
        if (MODE != 1) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float4 v = col_ok ? __ldg((const float4*)(x + col0) + i) : make_float4(0.f, 0.f, 0.f, 0.f);
                xw[4 * i] = v.x; xw[4 * i + 1] = v.y; xw[4 * i + 2] = v.z; xw[4 * i + 3] = v.w;
            }
        }
        float acc[ROWS];
        #pragma unroll
        for (int r = 0; r < (int)ROWS; r++) acc[r] = 0.f;
        constexpr uint32_t MR = ELEM == ELEM_FP8 ? MAXR_FP8 : MAXR;
        uint4 chunk_next = raw_chunk<ELEM>(low, f, row0, col0, col_ok);
        #pragma unroll 1
        for (int r = 0; r < (int)ROWS; r++) {
            uint4 chunk = chunk_next;
            if (r + 1 < (int)ROWS) chunk_next = raw_chunk<ELEM>(low, f, row0 + r + 1, col0, col_ok);
            uint32_t w16[8]; float row_sum = 0.f;
            #pragma unroll
            for (int q = 0; q < 8; q++) {
                uint32_t e = W.pair<MR>(sm_lut, sh, exp_of_rank);
                const int jj = 2 * q; const uint32_t cw = u4_word(chunk, jj);
                if (MODE != 1) {
                    row_sum = fmaf(weight_f32(cw, e, (5 << 12) | ((jj & 3) << 8) | 0xFF), xw[jj], row_sum);
                    row_sum = fmaf(weight_f32(cw, e, (6 << 12) | (((jj + 1) & 3) << 8) | 0xFF), xw[jj + 1], row_sum);
                } else w16[q] = two_bf16(cw, e, jj);
            }
            if (MODE == 1) {
                if (row0 + r < f.M && col_ok) {
                    uint4 *o = (uint4*)(out + (uint64_t)(row0 + r) * f.K16 + col0);
                    o[0] = make_uint4(w16[0], w16[1], w16[2], w16[3]); o[1] = make_uint4(w16[4], w16[5], w16[6], w16[7]);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < (int)ROWS - 1; i++) acc[i] = acc[i + 1];   /* rotation: at the end acc[i] = row i */
                acc[ROWS - 1] = row_sum;
            }
        }
        if (MODE != 1) {                                          /* transposed reduction: lane 4r receives row r */
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
            if ((lane & 3) == 0) {
                uint32_t row = row0 + (lane >> 2);
                if (MODE == 0) partial[(uint64_t)row * f.CB + cb] = v;         /* scratch has Mp rows */
                else if (row < f.M) atomicAdd(partial + row, v);
            }
        }
    }
}

/* partial sums (* per-row scale) (+ bias) -> BF16, round to nearest even */
__global__ void kern_f32_to_bf16_bias(const float *__restrict__ partial, const uint16_t *__restrict__ bias,
                                      const float *__restrict__ scale, uint16_t *__restrict__ yb, uint32_t M, uint32_t P)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    float v = 0.f;
    for (uint32_t s = 0; s < P; s++) v += partial[(uint64_t)i * P + s];
    if (scale) v *= scale[i];
    if (bias) v += __uint_as_float((uint32_t)bias[i] << 16);
    uint32_t u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    yb[i] = (uint16_t)(u >> 16);
}

/* Row gather (embedding lookup): one warp per requested row; per column block it decodes rows 0..r%8 */
template<int ELEM>
__global__ void __launch_bounds__(128)
kern_gather(const uint8_t *__restrict__ data, const uint32_t *__restrict__ bases, const uint8_t *__restrict__ hdr,
            const uint32_t *__restrict__ lut, const uint8_t *__restrict__ low, const int64_t *__restrict__ ids,
            uint32_t n_ids, uint16_t *__restrict__ out, uint32_t M, uint32_t K, uint32_t sh)
{
    __shared__ uint32_t smem[LUTN + 4];
    load_lut(smem, lut, threadIdx.x, blockDim.x);
    const uint8_t *exp_of_rank = (const uint8_t*)(smem + LUTN);
    const uint32_t sm_lut = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= n_ids) return;
    int64_t r = ids[warp];
    if (r < 0 || r >= (int64_t)M) return;
    const Shape f(M, K);
    constexpr uint32_t MR = ELEM == ELEM_FP8 ? MAXR_FP8 : MAXR;
    const uint32_t rb = (uint32_t)r / ROWS, rr = (uint32_t)r % ROWS;
    uint16_t *o = out + (uint64_t)warp * f.K16;
    for (uint32_t cb = 0; cb < f.CB; cb++) {
        const uint32_t col0 = cb * COLS + lane * 16; const bool col_ok = col0 < f.K16;
        Window W; W.setup(data, bases, hdr, rb * f.CB + cb, lane);
        uint4 chunk = raw_chunk<ELEM>(low, f, (uint32_t)r, col0, col_ok);
        for (uint32_t z = 0; z < rr; z++)                         /* skip the rows in front (decode, discard) */
            #pragma unroll
            for (int q = 0; q < 8; q++) W.pair<MR>(sm_lut, sh, exp_of_rank);
        uint32_t w16[8];
        #pragma unroll
        for (int q = 0; q < 8; q++) { uint32_t e = W.pair<MR>(sm_lut, sh, exp_of_rank); w16[q] = two_bf16(u4_word(chunk, 2 * q), e, 2 * q); }
        if (col_ok) {
            uint4 *o4 = (uint4*)(o + col0);
            o4[0] = make_uint4(w16[0], w16[1], w16[2], w16[3]); o4[1] = make_uint4(w16[4], w16[5], w16[6], w16[7]);
        }
    }
}

/* Reference "native" fp8 weight-only matvec: y[i] = scale[i] * sum_k fp8[i,k] * x[k]. The bandwidth baseline NWC-FP8 is
   compared with. One warp per 4 rows, 16 weights per lane and iteration (uint4 loads), x fp32[K16] (zeros beyond K)
   converted to half2 once per 16-column chunk and shared by the 4 rows. sm_89+: hardware e4m3x2 -> f16x2 conversion and
   HFMA2 over the 16 products of a chunk, chunk sums accumulated in fp32 (the usual fast fp8 weight-only recipe);
   older architectures: a 256-entry fp32 table in shared memory. y BF16[M]. */
__device__ __forceinline__ float fp8_to_f32(uint32_t b) {
    uint32_t s = b >> 7, e = (b >> 3) & 15u, m = b & 7u;
    float v = (e == 0) ? (float)m * 0.001953125f : ldexpf(1.f + (float)m * 0.125f, (int)e - 7);
    return s ? -v : v;
}
__device__ __forceinline__ uint32_t h2_bits(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __half2 bits_h2(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }
__device__ __forceinline__ uint32_t bf16_bits(float v) { uint32_t u = __float_as_uint(v); u += 0x7FFFu + ((u >> 16) & 1u); return u >> 16; }
__global__ void __launch_bounds__(128)
kern_ref_fp8(const uint8_t *__restrict__ w, const float *__restrict__ scale, const float *__restrict__ x,
             uint16_t *__restrict__ y, uint32_t M, uint32_t K)
{
    const uint32_t row0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * 4, lane = threadIdx.x & 31;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (K % 16 == 0) {
#if __CUDA_ARCH__ >= 890
        for (uint32_t k = lane * 16; k < K; k += 512) {
            const float4 *xp = (const float4*)(x + k);
            uint32_t xh[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float4 v = __ldg(xp + i);
                xh[2 * i] = h2_bits(__floats2half2_rn(v.x, v.y)); xh[2 * i + 1] = h2_bits(__floats2half2_rn(v.z, v.w));
            }
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const uint32_t row = row0 + r;
                if (row >= M) break;
                uint4 c = __ldcs((const uint4*)(w + (uint64_t)row * K + k));
                const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
                uint32_t h2 = 0;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint32_t lo, hi;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(lo) : "h"((unsigned short)(cw[j] & 0xFFFFu)));
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hi) : "h"((unsigned short)(cw[j] >> 16)));
                    asm("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(h2) : "r"(lo), "r"(xh[2 * j]));
                    asm("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(h2) : "r"(hi), "r"(xh[2 * j + 1]));
                }
                float2 f = __half22float2(bits_h2(h2));
                acc[r] += f.x + f.y;
            }
        }
#else
        __shared__ float tab[256];
        for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) tab[i] = fp8_to_f32(i);
        __syncthreads();
        for (uint32_t k = lane * 16; k < K; k += 512) {
            float xv[16];
            #pragma unroll
            for (int i = 0; i < 4; i++) { float4 v = __ldg((const float4*)(x + k) + i); xv[4*i] = v.x; xv[4*i+1] = v.y; xv[4*i+2] = v.z; xv[4*i+3] = v.w; }
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const uint32_t row = row0 + r;
                if (row >= M) break;
                uint4 c = __ldcs((const uint4*)(w + (uint64_t)row * K + k));
                const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
                #pragma unroll
                for (int j = 0; j < 16; j++) acc[r] = fmaf(tab[(cw[j >> 2] >> (8 * (j & 3))) & 0xFFu], xv[j], acc[r]);
            }
        }
#endif
    } else {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const uint32_t row = row0 + r;
            if (row >= M) break;
            for (uint32_t k = lane; k < K; k += 32) acc[r] = fmaf(fp8_to_f32(w[(uint64_t)row * K + k]), __ldg(x + k), acc[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        float v = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && row0 + r < M) y[row0 + r] = (uint16_t)bf16_bits(v * scale[row0 + r]);
    }
}

/* ============================ Host: launchers ============================ */
static int g_sms = 0;
EXPORT int nwc_setup(void);
#define SMEM (4 * LUTN + 16)
#define SH   (32 - PEEK)

static int shape_ok(uint64_t n, uint32_t block, uint32_t K) { return block == BLOCK && K != 0 && n % K == 0 && n / K <= 0xFFFFFFFFull; }

template<int MODE, int ELEM>
static void launch_mode(int bl, cudaStream_t st, const uint8_t *data, const uint32_t *bases, const uint8_t *hdr,
                        const uint32_t *tab, const uint8_t *low, const float *x, float *partial, uint16_t *out, const Shape &f) {
    kern_block<MODE, ELEM><<<bl, TBP, SMEM, st>>>(data, bases, hdr, tab, low, x, partial, out, f.M, f.K, SH);
}
static int launch_block(int mode, int elem, cudaStream_t st, const uint8_t *data, const uint32_t *bases, const uint8_t *hdr,
                        const uint32_t *tab, const uint8_t *low, const float *x, float *partial, uint16_t *out, uint64_t n, uint32_t K)
{
    if (g_sms == 0) nwc_setup();
    Shape f((uint32_t)(n / K), K);
    int bl = (int)((f.nb + TBP / 32 - 1) / (TBP / 32)); if (bl > g_sms) bl = g_sms;
    if (elem == ELEM_FP8) {
        if (mode == 0)      launch_mode<0, ELEM_FP8>(bl, st, data, bases, hdr, tab, low, x, partial, out, f);
        else if (mode == 1) launch_mode<1, ELEM_FP8>(bl, st, data, bases, hdr, tab, low, x, partial, out, f);
        else                launch_mode<2, ELEM_FP8>(bl, st, data, bases, hdr, tab, low, x, partial, out, f);
    } else {
        if (mode == 0)      launch_mode<0, ELEM_BF16>(bl, st, data, bases, hdr, tab, low, x, partial, out, f);
        else if (mode == 1) launch_mode<1, ELEM_BF16>(bl, st, data, bases, hdr, tab, low, x, partial, out, f);
        else                launch_mode<2, ELEM_BF16>(bl, st, data, bases, hdr, tab, low, x, partial, out, f);
    }
    return (int)cudaGetLastError();
}

EXPORT int nwc_matvec(void *stream, const uint8_t *data, const uint32_t *bases, const uint8_t *hdr,
                      const uint32_t *tab, const uint8_t *low, const float *x, float *y,
                      uint64_t n, uint32_t block, uint32_t K, int elem)
{
    if (!shape_ok(n, block, K)) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    cudaMemsetAsync(y, 0, 4ull * (n / K), st);
    return launch_block(2, elem, st, data, bases, hdr, tab, low, x, y, NULL, n, K);
}

/* compressed -> BF16[M x K16]. FP8: the unscaled fp8 values as BF16 (exact) */
EXPORT int nwc_dequant(void *stream, const uint8_t *data, const uint32_t *bases, const uint8_t *hdr,
                       const uint32_t *tab, const uint8_t *low, uint16_t *out, uint64_t n, uint32_t block, uint32_t K, int elem)
{
    if (!shape_ok(n, block, K)) return -1;
    return launch_block(1, elem, (cudaStream_t)stream, data, bases, hdr, tab, low, NULL, NULL, out, n, K);
}

/* x fp32 (K16, zeros beyond K), bias BF16 (M) or NULL, scale fp32 (M) or NULL (FP8 per-row scale), y BF16 (M);
   scratch fp32[Mp * CB] */
EXPORT int nwc_linear_bf16(void *stream, const uint8_t *data, const uint32_t *bases, const uint8_t *hdr,
                           const uint32_t *tab, const uint8_t *low, const float *x, const uint16_t *bias, const float *scale,
                           float *scratch, uint16_t *y, uint64_t n, uint32_t block, uint32_t K, int elem)
{
    if (!shape_ok(n, block, K)) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    Shape f((uint32_t)(n / K), K);
    int rc = launch_block(0, elem, st, data, bases, hdr, tab, low, x, scratch, NULL, n, K);
    if (rc) return rc;
    kern_f32_to_bf16_bias<<<(f.M + 255) / 256, 256, 0, st>>>(scratch, bias, scale, y, f.M, f.CB);
    return (int)cudaGetLastError();
}

EXPORT int nwc_gather(void *stream, const uint8_t *data, const uint32_t *bases, const uint8_t *hdr,
                      const uint32_t *tab, const uint8_t *low, const int64_t *ids, uint32_t n_ids,
                      uint16_t *out, uint64_t n, uint32_t block, uint32_t K, int elem)
{
    if (!shape_ok(n, block, K)) return -1;
    if (n_ids == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    if (elem == ELEM_FP8) kern_gather<ELEM_FP8><<<(n_ids * 32 + 127) / 128, 128, 0, st>>>(data, bases, hdr, tab, low, ids, n_ids, out, (uint32_t)(n / K), K, SH);
    else                  kern_gather<ELEM_BF16><<<(n_ids * 32 + 127) / 128, 128, 0, st>>>(data, bases, hdr, tab, low, ids, n_ids, out, (uint32_t)(n / K), K, SH);
    return (int)cudaGetLastError();
}

/* reference fp8 weight-only matvec (see kern_ref_fp8): w fp8[M x K] row-major, scale fp32[M], x fp32[K16], y BF16[M] */
EXPORT int nwc_ref_fp8(void *stream, const uint8_t *w, const float *scale, const float *x, uint16_t *y, uint64_t M, uint32_t K)
{
    if (M == 0 || K == 0 || M > 0xFFFFFFFFull) return -1;
    kern_ref_fp8<<<(unsigned)((M + 15) / 16), 128, 0, (cudaStream_t)stream>>>(w, scale, x, y, (uint32_t)M, K);
    return (int)cudaGetLastError();
}

EXPORT int nwc_setup(void) {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(kern_block<0, ELEM_BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(kern_block<1, ELEM_BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(kern_block<2, ELEM_BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(kern_block<0, ELEM_FP8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(kern_block<1, ELEM_FP8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(kern_block<2, ELEM_FP8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    return g_sms;
}
