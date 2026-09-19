# Format and decoder design

This document describes NWC format version 9, the one implemented in `csrc/nwc_ops.cu`, and the decoder
that goes with it. Earlier versions (rANS-based) are summarized at the end; their sources are kept under
`experiments/`.

## 1. What is compressed

A BF16 weight is 16 bits: sign (1), exponent (8), mantissa (7). Measured over real checkpoints, the mantissa
byte carries 7.97 bits of entropy and is stored raw. The remaining byte (sign + 7 exponent bits) carries
2.6–2.9 bits of entropy and is entropy-coded. Context modelling (neighbours, strides, per-row statistics)
buys less than 0.4 % and costs more in table traffic than it saves, so the coder is memoryless. The
resulting size is 68.5–69.5 % of BF16; this is the entropy of the data, not a property of the coder.

## 2. Code for the exponent byte

The 128 possible exponent values are ranked by frequency (table `exp_of_rank[16]`, stored with the weights).

| rank r | code |
|---|---|
| 0 … 14 | r zeros, a one, then the sign bit raw (r + 2 bits) |
| ≥ 15 | 15 zeros, a one, then the whole byte raw (24 bits) |

Mean cost 2.85–2.99 bits per weight on Qwen models; escapes are 0.01–0.4 % of the weights. A rANS coder
with 12 scale bits would save about 1 % of the exponent stream, at roughly twice the decoding cost.

## 3. Block layout

The matrix (M rows × K columns, row-major) is cut into blocks of **8 rows × 512 columns**. A block is
decoded by one warp; lane L owns columns 16L … 16L+15 of every row of the block, in the order row 0 … 7,
columns ascending. Each lane has its own bit stream (MSB first, packed into little-endian 32-bit words).

Any M and K are allowed. The matrix is conceptually padded to a multiple of 8 rows and 512 columns; a
filler weight costs 2 bits (rank-0 code) in the exponent stream and has no mantissa. The mantissa plane is
stored row-major with a row stride of K rounded up to 16 (`K16`); the activation vector is padded to `K16`
with zeros. For K = 11008 the padding costs 0.3 % of the compressed size, for shapes that are already
multiples it costs nothing.

Per block: 32 lane streams, each a multiple of 4 bytes, concatenated; blocks are 8-byte aligned
(`bases[]` holds the byte offset of each block); a header byte per lane holds its stream length in words.
The lane's start offset is the warp prefix sum of the header bytes.

## 4. Decoder

### Lookup table

A 4096-entry table (16 KB, copied to shared memory once per kernel) is indexed by the next 12 bits of the
stream. For 98.9 % of the patterns two complete codes fit in 12 bits and the entry is

```
byte 0: c    = bits consumed by both codes (put in the low byte so shf.l.wrap uses it as the shift amount)
byte 1: exponent byte of weight 0
byte 2: exponent byte of weight 1
byte 3: 0    (so the MSB is 0 — see "weights")
```

Otherwise the entry has bit 30 set and carries the first code (if complete) plus its length; the slow path
finishes the pair with `clz`. The slow path runs for 0.06–0.4 % of lane steps, i.e. 2–6 % of warp steps.
The values `exp_of_rank[11..14]` are stored in the two table entries that can never hold a complete code
(patterns 0 and 1), ranks 0–10 are read back from the patterns "r zeros, one"; the kernel therefore needs
no second table.

### Bit window

Each lane keeps a 64-bit window `hi:lo`, MSB first. Advancing by `c` bits is two funnel shifts. The fill
level is not a counter but a rotated power of two: `M = 2^(64 − nb) mod 2^32`; a step rotates `M` left by
`c`. When the rotation wraps (`M < M_old`) fewer than 32 valid bits are left and the pre-loaded next word
`w` is inserted with one wide multiply-add, `hi:lo += w · M` — for lanes that did not wrap the multiplier
is zero and the instruction is a no-op. Only the load of the following word is predicated. The block's
stream data is prefetched into L1 at block start (one `prefetch.global.L1` per lane).

### Weights and accumulation

A decoded pair yields the two exponent bytes in bytes 1 and 2 of a register; the two mantissa bytes sit in
a `uint4` loaded once per row per lane. One `prmt` builds the fp32 bit pattern of a weight from the
exponent byte and the mantissa byte, using sign replication of the zero MSB of byte 3 for the two zero
bytes, so no masking instruction is needed. The activation values (fp32, 16 per lane) are loaded once per
block into registers; each row is accumulated in a register, rotated through eight accumulators, and at
block end a transposed shuffle reduction leaves the sum of row r in lane 4r. Partial sums go to
`scratch[row · K/512 + column_block]`; a small finalize kernel adds them, adds the bias and rounds to BF16.

The token path runs one persistent block of 1024 threads per SM, grid-striding over the warp blocks.
Dequantization (row-major BF16 output) and the embedding gather (one warp per requested row, decoding the
rows in front of it and discarding them) use the same decoder.

### Instruction budget (A16, measured)

The decoder's cost is issue-bound, not memory-bound. Measured per SASS instruction on the A16 (Ampere
GA107, SM clocks per warp instruction; `experiments/pipe_bench.cu`, independent chains with data-dependent
operands — with constant operands ptxas folds the chains and reports twice the rate):

| instruction | cost | pipe |
|---|---|---|
| FFMA | 0.27 | FMA |
| IADD3, LOP3 | 0.30 | either |
| IMAD, IMUL, IMAD.WIDE, SHF, ISETP, SEL, IMNMX | 0.51 | IMAD on FMA, the rest on ALU; ALU and FMA overlap |
| PRMT | 0.51 | blocks both pipes |
| IMAD.HI | 1.0 | blocks both pipes |
| FLO (clz) | 2.0 | |

The v9 fast path is 21–23 instructions per pair of weights, of which 4–5 are register shuffling that
ptxas produces around 64-bit pairs and predicated loads. Issue alone bounds it at 5.5 clocks per warp step;
measured are 11.0 (gate_up) and 12.6 (lm_head) clocks per 64 weights against 13.6 needed to match cuBLAS on
the A16. The next lever is a tensor-core path: one `prmt` per pair produces the bf16x2 A-fragment of an
`mma.m16n8k16` directly, the activations become the B-fragment, accumulation happens in the D-fragment,
and the reduction and the activation registers disappear.

## 4b. FP8 element type (weight-only fp8 e4m3, format v9 with `elem = 1`)

The same block layout, streams, LUT and decoder serve fp8 weights (`quantize_fp8`: symmetric absmax scale per
output channel, activations stay BF16; the per-row scale is applied in the finalize kernel). The fp8 byte
`s e4 m3` is split so that the BF16 decoder needs no new instructions in the hot loop:

| fp8 value | coded symbol (rank prefix code, 10 symbols, no escape) | raw nibble (4-bit plane) |
|---|---|---|
| normal, exponent e = 1..15 (NaN rejected) | e >> 1 (8 symbols), LUT byte = 60 + (e >> 1) | (e & 1) << 3 \| mantissa |
| subnormal m · 2⁻⁹, m ≥ 4 | symbol 0 (fp32 exponent 120) | (m − 4) << 1 |
| subnormal m = 1, 2, 3 | symbol 8, LUT byte 59 (fp32 exponents 118/119) | 0, 8, 12 |
| zero | symbol 9, LUT byte 0 | 0 |

The decoded fp32 weight is `[sign | LUT byte | nibble << 4 | 0 | 0]`, i.e. the LUT byte is the fp32 exponent
field shifted right by one and the nibble carries its low bit plus the three mantissa bits; the sign is coded
behind the rank code as for BF16. Subnormals and zeros are ordinary symbols (with per-channel scaling they are
3 % of the weights, far too many for a slow path). The nibble plane has a row stride of `K16 / 2` bytes; within a
lane's 16 weights the nibbles are ordered so that one shift and one mask turn a raw word into the four weight
bytes the PRMT expects (word 0 low nibbles = weights 0..3, high nibbles = 4..7, word 1 likewise 8..15).
Dequantization yields the unscaled fp8 values as BF16 (exact), so `dequant_fp8` recovers the fp8 tensor bit
for bit.

Size on Qwen3-4B: 6.9 bits per weight, 0.866 of fp8 (`scripts/entropy_quant.py`; an ideal coder on the same
split reaches 0.862, on the full 4-bit exponent 0.828). int8 was measured and rejected: the rank code reaches
only 0.95 of int8 because the alphabet is flat; that needs a rANS-class coder.

## 5. Library interface (`nwc_ops.dll` / `nwc_ops.so`)

| export | purpose |
|---|---|
| `nwc_info(i)` | format version, states per lane, table words, header bytes per block, block size |
| `nwc_layout(M, K, elem, out[4])` | block count, raw-plane bytes, scratch floats, `K16` for a shape |
| `nwc_encode(w, n, K, elem, freq, psym, bases, hdr, data, cap, low)` | host encoder (BF16 or fp8 bytes); `psym[0..15]` receives `exp_of_rank` |
| `nwc_build_tab(freq, psym, elem, tab)` | builds the 4096-entry lookup table |
| `nwc_ref_fp8(...)` | reference fp8 weight-only matvec (bandwidth baseline for the fp8 comparison) |
| `nwc_linear_bf16(...)` | y = scale · (W x) + b, x fp32 (`K16`), y BF16; persistent block kernel + finalize |
| `nwc_matvec(...)` | fp32 test path (atomics) |
| `nwc_dequant(...)` | BF16 row-major output (stride `K16`) |
| `nwc_gather(...)` | rows by index (embedding lookup) |

The Python bridge (`nwc/nwc_torch.py`) reads the format parameters through `nwc_info` and keeps loading
older libraries (v7/v8) for A/B comparisons via the `NWC_DLL` environment variable.

## 6. Earlier formats

| version | coder | block | notes |
|---|---|---|---|
| 4 | rANS, 12 scale bits, 16-bit renorm | 8192 weights, 32 lane streams | first fused kernel; chunk layout with 16 consecutive weights per lane |
| 5 | tANS/FSE | same | same speed, 1.1 % worse rate, simpler kernel; CLI only |
| 6 | rANS | 2048 (capped) | streams without padding, 1-byte lengths, prefix-sum offsets; gather kernel |
| 7 | rANS on exponent pairs, two states per lane | | halves gathers per weight; A16 0.53× → 0.74× |
| 8 | rANS pairs, one 64-bit state, 32-bit renorm | 4096 | 32-bit table `[pair 16 \| freq 8 \| bias 8]`, escape side stream, persistent blocks with table and x in shared memory; A16 0.73–0.78× |
| **9** | **prefix code + LUT** | **8 rows × 512 columns** | **A16 1.06–1.18×** |
