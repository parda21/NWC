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
   Exports:
     nwc_info        host. 0: format version, 1: states per lane (1), 2: table words (u32),
                     3: header bytes per block (32), 4: block size (4096, fixed)
     nwc_layout      host. (M, K) -> [blocks, bytes of the mantissa plane, floats of the scratch buffer, K16]
     nwc_encode      host. BF16 (n = M*K) -> psym[0..15] = exp_of_rank, bases, hdr, data, low (M x K16);
                     parameter 3 is K (the block size is fixed). freq is unused.
     nwc_build_tab   host. psym (exp_of_rank) -> tab[4096 u32] (LUT)
     nwc_matvec      device, fp32 test path, y = W x via atomics; x fp32[K16]
     nwc_linear_bf16 device. y = W x + b, x fp32[K16] (zeros beyond K), y BF16[M], decoded in registers
     nwc_dequant     device. compressed -> BF16[M x K16] (row-major, stride K16); needs K
     nwc_gather      device. rows ids[] -> BF16[n_ids x K16] (embedding lookup)
     nwc_setup       host. SM count (called on load)
   Earlier formats: v8 (rANS, pair symbols) in experiments/nwc_ops_v8.cu, v7 in experiments/nwc_ops_v7.cu. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

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
#define MAXR       15u                   /* ranks 0..14 unary, escape from 15 */
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
                    case 4: return (int)BLOCK; }
    return -1;
}
EXPORT int nwc_layout(uint64_t M, uint32_t K, uint64_t *out) {
    if (M == 0 || K == 0 || M > 0xFFFFFFFFull) return -1;
    Shape f((uint32_t)M, K);
    out[0] = f.nb; out[1] = (uint64_t)f.M * f.K16; out[2] = (uint64_t)f.Mp * f.CB; out[3] = f.K16;
    return 0;
}

/* ============================ Host: encoder ============================ */
struct BitWriter {                                        /* MSB-first bit writer into u32 words */
    uint32_t *w; uint32_t n_w; uint64_t acc; int n;
    void push(uint32_t v, int nb) { acc = (acc << nb) | v; n += nb; while (n >= 32) { w[n_w++] = (uint32_t)(acc >> (n - 32)); n -= 32; } }
    void finish() { if (n) { w[n_w++] = (uint32_t)(acc << (32 - n)); n = 0; } }
};

EXPORT int64_t nwc_encode(const uint16_t *w, uint64_t n, uint32_t K, uint16_t *freq, uint16_t *psym,
                          uint32_t *bases, uint8_t *hdr, uint8_t *data, uint64_t data_cap, uint8_t *low)
{
    if (K == 0 || n % K || n / K > 0xFFFFFFFFull) return -1;
    Shape f((uint32_t)(n / K), K);
    uint8_t *high = (uint8_t*)malloc(n);
    for (uint32_t m = 0; m < f.M; m++) {                                   /* mantissa plane with stride K16 */
        for (uint32_t k = 0; k < f.K; k++) { uint16_t v = w[(uint64_t)m * K + k]; low[(uint64_t)m * f.K16 + k] = (uint8_t)v; high[(uint64_t)m * K + k] = (uint8_t)(v >> 8); }
        for (uint32_t k = f.K; k < f.K16; k++) low[(uint64_t)m * f.K16 + k] = 0;
    }
    /* ranks of the 7-bit exponents by frequency */
    uint64_t hist[128] = {0}; for (uint64_t i = 0; i < n; i++) hist[high[i] & 0x7F]++;
    int ord[128]; for (int i = 0; i < 128; i++) ord[i] = i;
    for (int a = 0; a < 128; a++) for (int b = a + 1; b < 128; b++) if (hist[ord[b]] > hist[ord[a]]) { int t = ord[a]; ord[a] = ord[b]; ord[b] = t; }
    uint8_t rank_of_exp[128];
    for (int r = 0; r < 128; r++) rank_of_exp[ord[r]] = (uint8_t)r;
    for (uint32_t i = 0; i < NPAIR; i++) { freq[i] = 0; psym[i] = (i < 16) ? (uint16_t)ord[i] : 0; }

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
                if (r < MAXR) { bw.push(1, r + 1); bw.push(s, 1); }
                else { bw.push(1, 16); bw.push(e, 8); }
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
EXPORT void nwc_build_tab(const uint16_t *freq, const uint16_t *psym, uint32_t *lut) {
    (void)freq;
    for (uint32_t pat = 0; pat < LUTN; pat++) {
        uint32_t bits = pat << (32 - PEEK), pos = 0, n = 0, c1 = 0, by[2] = {0, 0};
        for (int k = 0; k < 2; k++) {
            uint32_t rest = PEEK - pos; if (rest == 0) break;
            uint32_t x = bits << pos, u = clz32(x);
            if (u >= MAXR || u + 2 > rest) break;
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
        if (M < M_old) { asm("ld.global.nc.u32 %0, [%1];" : "=r"(w) : "l"(p)); p += 4; }
    }
    __device__ __forceinline__ uint32_t peek4(uint32_t sh) const { uint32_t i; asm("shr.u32 %0, %1, %2;" : "=r"(i) : "r"(hi()), "r"(sh)); return i * 4u; }
    __device__ __forceinline__ uint32_t decode_slow(const uint8_t *exp_of_rank) {   /* one code via clz */
        uint32_t h = hi(), u = __clz(h);
        if (u < MAXR) { uint32_t s = (h >> (30 - u)) & 1; uint32_t b = (s << 7) | exp_of_rank[u]; advance(u + 2); return b; }
        advance(16); uint32_t b = hi() >> 24; advance(8); return b;
    }
    /* decode one pair: returns e = [. | by0 | by1 | 0] (bytes 1/2 = exponent bytes, byte 3 = 0) */
    __device__ __forceinline__ uint32_t pair(uint32_t sm_lut, uint32_t sh, const uint8_t *exp_of_rank) {
        uint32_t e;
        asm("ld.shared.u32 %0, [%1];" : "=r"(e) : "r"(sm_lut + peek4(sh)));
        if (e < FLAG) advance(e);
        else {
            uint32_t n = e >> 24 & 1, b0 = (e >> 8) & 0xFFu, b1;
            if (n) advance(e);
            if (!n) b0 = decode_slow(exp_of_rank);
            b1 = decode_slow(exp_of_rank);
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
/* 16 mantissa bytes of a lane (row, column col0), zeros outside the matrix */
__device__ __forceinline__ uint4 mantissas(const uint8_t *__restrict__ low, const Shape &f, uint32_t row, uint32_t col0, bool col_ok) {
    if (row < f.M && col_ok) return __ldcs((const uint4*)(low + (uint64_t)row * f.K16 + col0));
    return make_uint4(0u, 0u, 0u, 0u);
}

/* Persistent block kernel. MODE 0: partial sums partial[row*CB + cb] (x fp32), MODE 1: dequantization
   (BF16 row-major to out, stride K16), MODE 2: fp32 matvec via atomics on y (tests) */
template<int MODE>
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
        uint4 chunk_next = mantissas(low, f, row0, col0, col_ok);
        #pragma unroll 1
        for (int r = 0; r < (int)ROWS; r++) {
            uint4 chunk = chunk_next;
            if (r + 1 < (int)ROWS) chunk_next = mantissas(low, f, row0 + r + 1, col0, col_ok);
            uint32_t w16[8]; float row_sum = 0.f;
            #pragma unroll
            for (int q = 0; q < 8; q++) {
                uint32_t e = W.pair(sm_lut, sh, exp_of_rank);
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

/* partial sums -> BF16 (+ bias), round to nearest even */
__global__ void kern_f32_to_bf16_bias(const float *__restrict__ partial, const uint16_t *__restrict__ bias,
                                      uint16_t *__restrict__ yb, uint32_t M, uint32_t P)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    float v = 0.f;
    for (uint32_t s = 0; s < P; s++) v += partial[(uint64_t)i * P + s];
    if (bias) v += __uint_as_float((uint32_t)bias[i] << 16);
    uint32_t u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    yb[i] = (uint16_t)(u >> 16);
}

/* Row gather (embedding lookup): one warp per requested row; per column block it decodes rows 0..r%8 */
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
    const uint32_t rb = (uint32_t)r / ROWS, rr = (uint32_t)r % ROWS;
    uint16_t *o = out + (uint64_t)warp * f.K16;
    for (uint32_t cb = 0; cb < f.CB; cb++) {
        const uint32_t col0 = cb * COLS + lane * 16; const bool col_ok = col0 < f.K16;
        Window W; W.setup(data, bases, hdr, rb * f.CB + cb, lane);
        uint4 chunk = col_ok ? __ldg((const uint4*)(low + (uint64_t)r * f.K16 + col0)) : make_uint4(0u, 0u, 0u, 0u);
        for (uint32_t z = 0; z < rr; z++)                         /* skip the rows in front (decode, discard) */
            #pragma unroll
            for (int q = 0; q < 8; q++) W.pair(sm_lut, sh, exp_of_rank);
        uint32_t w16[8];
        #pragma unroll
        for (int q = 0; q < 8; q++) { uint32_t e = W.pair(sm_lut, sh, exp_of_rank); w16[q] = two_bf16(u4_word(chunk, 2 * q), e, 2 * q); }
        if (col_ok) {
            uint4 *o4 = (uint4*)(o + col0);
            o4[0] = make_uint4(w16[0], w16[1], w16[2], w16[3]); o4[1] = make_uint4(w16[4], w16[5], w16[6], w16[7]);
        }
    }
}

/* ============================ Host: launchers ============================ */
static int g_sms = 0;
EXPORT int nwc_setup(void);
#define SMEM (4 * LUTN + 16)
#define SH   (32 - PEEK)

static int shape_ok(uint64_t n, uint32_t block, uint32_t K) { return block == BLOCK && K != 0 && n % K == 0 && n / K <= 0xFFFFFFFFull; }

static int launch_block(int mode, cudaStream_t st, const uint8_t *data, const uint32_t *bases, const uint8_t *hdr,
                        const uint32_t *tab, const uint8_t *low, const float *x, float *partial, uint16_t *out, uint64_t n, uint32_t K)
{
    if (g_sms == 0) nwc_setup();
    Shape f((uint32_t)(n / K), K);
    int bl = (int)((f.nb + TBP / 32 - 1) / (TBP / 32)); if (bl > g_sms) bl = g_sms;
    if (mode == 0)      kern_block<0><<<bl, TBP, SMEM, st>>>(data, bases, hdr, tab, low, x, partial, out, f.M, K, SH);
    else if (mode == 1) kern_block<1><<<bl, TBP, SMEM, st>>>(data, bases, hdr, tab, low, x, partial, out, f.M, K, SH);
    else                kern_block<2><<<bl, TBP, SMEM, st>>>(data, bases, hdr, tab, low, x, partial, out, f.M, K, SH);
    return (int)cudaGetLastError();
}

EXPORT int nwc_matvec(void *stream, const uint8_t *data, const uint32_t *bases, const uint8_t *hdr,
                      const uint32_t *tab, const uint8_t *low, const float *x, float *y,
                      uint64_t n, uint32_t block, uint32_t K)
{
    if (!shape_ok(n, block, K)) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    cudaMemsetAsync(y, 0, 4ull * (n / K), st);
    return launch_block(2, st, data, bases, hdr, tab, low, x, y, NULL, n, K);
}

EXPORT int nwc_dequant(void *stream, const uint8_t *data, const uint32_t *bases, const uint8_t *hdr,
                       const uint32_t *tab, const uint8_t *low, uint16_t *out, uint64_t n, uint32_t block, uint32_t K)
{
    if (!shape_ok(n, block, K)) return -1;
    return launch_block(1, (cudaStream_t)stream, data, bases, hdr, tab, low, NULL, NULL, out, n, K);
}

/* x fp32 (K16, zeros beyond K), bias BF16 (M) or NULL, y BF16 (M); scratch fp32[Mp * CB] */
EXPORT int nwc_linear_bf16(void *stream, const uint8_t *data, const uint32_t *bases, const uint8_t *hdr,
                           const uint32_t *tab, const uint8_t *low, const float *x, const uint16_t *bias,
                           float *scratch, uint16_t *y, uint64_t n, uint32_t block, uint32_t K)
{
    if (!shape_ok(n, block, K)) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    Shape f((uint32_t)(n / K), K);
    int rc = launch_block(0, st, data, bases, hdr, tab, low, x, scratch, NULL, n, K);
    if (rc) return rc;
    kern_f32_to_bf16_bias<<<(f.M + 255) / 256, 256, 0, st>>>(scratch, bias, y, f.M, f.CB);
    return (int)cudaGetLastError();
}

EXPORT int nwc_gather(void *stream, const uint8_t *data, const uint32_t *bases, const uint8_t *hdr,
                      const uint32_t *tab, const uint8_t *low, const int64_t *ids, uint32_t n_ids,
                      uint16_t *out, uint64_t n, uint32_t block, uint32_t K)
{
    if (!shape_ok(n, block, K)) return -1;
    if (n_ids == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    kern_gather<<<(n_ids * 32 + 127) / 128, 128, 0, st>>>(data, bases, hdr, tab, low, ids, n_ids, out, (uint32_t)(n / K), K, SH);
    return (int)cudaGetLastError();
}

EXPORT int nwc_setup(void) {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(kern_block<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(kern_block<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(kern_block<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    return g_sms;
}
