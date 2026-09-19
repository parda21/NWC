# Results

All numbers are measured, medians over many runs, never right after compilation (clock state). Two GPUs:

- **RTX 4070**, 12 GB GDDR6X, 452 GB/s measured read bandwidth, 46 SMs, CUDA 13.4 / MSVC 14.44.
- **NVIDIA A16**, vGPU profile 16Q (16 GB, 10 SMs, 2 MB L2, GDDR6; cuBLAS reaches 160–165 GB/s), Ampere
  sm_86, built with nvcc 12.6 in a container. The same library, only compiled for sm_86.

Model runs use Qwen3-4B (HF Transformers, BF16), fused q/k/v and gate/up projections, tied `lm_head`
compressed and served through the gather kernel.

## 1. Headline (format v9)

| | RTX 4070 | A16 |
|---|---|---|
| weights, native → NWC | 8.04 GB → 5.55 GB (rate 0.689) | same |
| VRAM in use, native → NWC | 8.10 GB → 5.67 GB | 8.10 GB → 5.67 GB |
| tokens/s as CUDA graph, native → NWC | 45.0 → **55.2** | 16.8 → **18.2** |
| GPU time per token, native → NWC | 20.4 → **18.9 ms** | 54.1 → **52.1 ms** |
| token sequence vs HF `generate` (greedy, 64 tokens) | identical | identical |
| logits vs native | max Δ = 0 | max Δ = 0.20, argmax identical¹ |

¹ cuBLAS itself differs by max Δ = 0.19 between the two GPUs (different fp32 summation order). Bit-identical
are the *weights* (dequantization is bit-exact on both architectures); logits are bit-identical only where
the summation order matches, which it does on the 4070.

Kernels per layer shape (`scripts/kernbench.py`, GPU time by CUDA events, copies rotated against the L2):

| layer (M × K) | 4070: cuBLAS / NWC | ratio | A16: cuBLAS / NWC | ratio |
|---|---|---|---|---|
| qkv 6144 × 2560 | 0.117 / 0.100 ms | 1.17× | 0.200 / 0.178 ms | 1.12× |
| o 2560 × 4096 | 0.086 / 0.073 ms | 1.18× | 0.135 / 0.128 ms | 1.06× |
| gate_up 19456 × 2560 | 0.252 / 0.163 ms | 1.55× | 0.605 / 0.513 ms | 1.18× |
| down 2560 × 9728 | 0.146 / 0.110 ms | 1.34× | 0.306 / 0.271 ms | 1.13× |
| lm_head 151936 × 2560 | 1.675 / 1.241 ms | 1.35× | 4.787 / 4.538 ms | 1.06× |
| sum per token (36 layers + lm_head) | 23.4 / 17.3 ms | 1.35× | 49.6 / 43.8 ms | 1.13× |

On the 4070 the kernel is at the physical bandwidth (compressed bytes read at 430–450 GB/s); on the A16 it is
issue-bound in the decoder. The 4070 kernel figures were taken with the fixed-shape build; the shape-generic
padding costs ≤ 3 % (A16: sum 43.0 → 43.8 ms).

Correctness tests (`tests/test_k.py`, `tests/test_gather.py`) pass on both GPUs for K = 11008, 3600, 2064,
1000, 40, 16, 4097 and M = 1001, 13, 3: dequantization and gather bit-exact, fp32 path within 2·10⁻⁵ relative
of a double reference (summation order only).

## 2. Comparison with DFloat11

[DFloat11](https://github.com/LeanModels/DFloat11) (NeurIPS 2025) uses Huffman coding of the exponent bits
and decodes block-wise into a buffer before the matmul. Same model (Qwen/Qwen3-4B vs DFloat11/Qwen3-4B-DF11,
dfloat11 0.5.0), same prompt, same measurement (torch.profiler, HF eager), RTX 4070:

| | native BF16 | DFloat11 | NWC (v6 at the time) | NWC v9 |
|---|---|---|---|---|
| VRAM | 8.10 GB | 5.73 GB | 5.78 GB | 5.67 GB |
| GPU time per token | 20.4 ms | 56.1 ms (2.74×) | 20.0 ms | 18.9 ms |
| weight kernels per token | gemv 17.2 ms | decode 36.6 + gemv 16.2 ms | 18.0 ms | 17.3 ms |
| logits vs native | — | bit-identical (paper) | max Δ = 0 | max Δ = 0 |

DFloat11's decode kernel alone takes longer than the whole native forward pass because the decoded weights
pass through memory twice (write, then read by the matmul). NWC decodes inside the matvec and only reads the
compressed bytes. Both sit at the same size, which is the entropy of the mantissa; the coder choice moves the
rate by 1–2 %. DFloat11 was not measured under a CUDA graph (its decode runs in separate cupy kernels); the
GPU-time comparison is the robust figure.

Earlier end-to-end runs on the 4070 (format v4, HF eager): Qwen2.5-7B, 14.14 → 9.69 GB of linear weights,
fits a 12 GB card; Qwen2.5-3B perplexity on WikiText-2 identical to native (10.3032).

## 3. Why it works, and where the limit is

Token generation is memory-bound: the matvec reads every weight once per token. NWC reads 69 % of the bytes
and decodes them with compute units that would otherwise idle. Whether that is a win depends on the ratio of
compute to bandwidth:

- The decoder has a fixed cost per weight; its throughput scales with SM count × clock, not with memory
  bandwidth. Format v9 delivers **7–8 GB/s of compressed data per SM·GHz** (A16: 11.0–12.6 SM clocks per 64
  weights; v8 5.0, v7 5.1, v6 3.5–3.9).
- NWC is faster than native when decoder throughput exceeds *bandwidth × 0.69*.

| GPU | SMs × GHz | decoder, v9 (GB/s compressed) | bandwidth × 0.69 | expectation |
|---|---|---|---|---|
| A16 (per GPU) | 10 × 1.76 | 123–140 (measured) | 165 × 0.69 = 114 | 1.06–1.18× (measured) |
| RTX 4070 | 46 × 2.5 | > 800 | 452 × 0.69 = 312 | bandwidth-bound, 1.2–1.55× (measured) |
| RTX 4080 Super | 80 × 2.55 | ~1500 | 736 × 0.69 = 508 | bandwidth-bound, ~1.4× |
| A100 SXM (2.0 TB/s) | 108 × 1.41 | ~1100 | ~1750 × 0.69 = 1200 | ~0.9× |
| H100 PCIe (2.0 TB/s) | 114 × 1.76 | ~1400–1600 | ~1850 × 0.69 = 1280 | ~1.1–1.2× |
| H100 SXM (3.35 TB/s) | 132 × 1.83 | ~1700–1900 | ~3000 × 0.69 = 2070 | ~0.8–0.9× |

The projections assume Ampere issue rates; Hopper has not been measured. The A100/H100-SXM gap is the
motivation for the tensor-core accumulation path described in `docs/format.md`.

## 4. Development history

Fused matvec on the RTX 4070, 451 M weights (K = 4096–14336), median of 60 runs, vs cuBLAS:

| kernel | speedup | change |
|---|---|---|
| v1 | 0.44× | single-byte loads in the decode chain |
| v2 | 0.72× | stream prefetch, mantissa via warp shuffle |
| v3 | 1.06× | chunk layout (16 consecutive weights per lane) |
| v4 | 1.16× | 40 registers (launch bounds), x as float4, L1 carve-out |
| v6 | 1.12–1.18× | validated against cuBLAS over 5 values of K |

Model-level and second-architecture progression (Qwen3-4B shapes):

| format | 4070 gate_up / lm_head vs cuBLAS | A16 gate_up / lm_head vs cuBLAS | A16 tokens/s (native 16.8) |
|---|---|---|---|
| v6, block 2048 | 1.18 / 1.26× | 0.52 / 0.53× | 9.5 |
| v7, exponent pairs, two rANS states | 1.41 / 1.36× | 0.74 / 0.71× | 12.4 |
| v8, one 64-bit state, persistent blocks | 1.41 / 1.33× | 0.75 / 0.74× | 12.9 |
| **v9, prefix code + LUT** | **1.55 / 1.35×** | **1.18 / 1.06×** | **18.2** |

The v9 decoder itself, measured as SM clocks per warp step of 64 weights on the A16 (parity = 13.6):

| stage | gate_up | lm_head | change |
|---|---|---|---|
| v8 | 17.7 | 18.3 | — |
| prefix code, 12-bit LUT, IMAD bit window (`experiments/v9_rice.cu`) | 17.5 | 19.8 | table gather gone, but INT pipe busy |
| 8 × 512 block, x in registers, multiplicative fill level (`v9b_rice.cu`) | 16.0 | 17.2 | no x loads, no row-change logic |
| slow path out of line | 12.5 | 14.3 | **code size**: 147 KB → 52 KB |
| row loop not unrolled | 12.5 | 14.0 | 23 KB, slow path inline again |
| funnel-shift window, rotation fill level (`v9c_rice.cu`) | 12.1 | 13.3 | work moved from the FMA to the ALU pipe |
| block stream prefetch into L1 | **11.0** | **12.6** | one prefetch per lane and block |

Knockouts of the final kernel (gate_up, without prefetch 12.1): no slow path 11.0, no refill 9.7, no LUT
access 10.9, floor (window shift, weights, loop, block overhead) 9.7.

## 5. What did not work

Kept as compile switches in the respective sources so they can be re-measured.

- **rANS tuning (v8):** 64-bit table entries (+4 %), four states per lane (−5 to −15 %), L1 prefetch of the
  renorm stream (0 to −7 %), 8192-weight blocks (always slower: 32–40 warps' stream data thrash the L1),
  IMAD.HI instead of SHF for the slot (±0), escape via `__any_sync` (±0), half occupancy (±0). Quad symbols:
  exponent bytes are nearly independent (2.815 / 2.811 / 2.803 bits per weight for 1/2/4-byte symbols),
  49 000 distinct quads, escape mass 23–28 %.
- **v9 decoder:** warp-uniform branch via `__any_sync` (+1 clock; ptxas emits BRA.DIV + VOTE + BSSY),
  unrolling the row loop 2×/4× (±0 / +0.3), 13/14-bit peek (slower), IMAD.HI instead of SHF (±0), LOP3 mask
  instead of ISETP+SEL (+0.3: ptxas then splits the pointer pair).
- **Earlier layout experiments (v6):** row-interleaved streams, chunk-interleaved mantissa, prefetch depth 2,
  SM-local work queue, 1–2 warps per CTA, 4/8 KB tables, two-level table, table in shared memory with the rANS
  layout (bank conflicts), L2 prefetch hints, 16-byte refills. Kept: mantissa loads evict-first (`__ldcs`,
  +3–5 %).
- **Compiler behaviour worth knowing:** ptxas folds instruction chains with constant operands (a benchmark
  artefact), splits `mad.wide` with a 64-bit addend unless the pair arises naturally, turns a predicated
  `mad.wide` into SEL + split; `#pragma unroll N` does not expand macros under Linux nvcc.

## 6. Open items

Tracked as [roadmap issues](https://github.com/parda21/NWC/issues?q=is%3Aissue+label%3Aroadmap) on GitHub.

1. Tensor-core accumulation path (see `docs/format.md`) for A100 / H100 SXM parity; measure `pipe_bench` on
   Hopper first ([#3](https://github.com/parda21/NWC/issues/3)).
2. Measure on RTX 4080 Super ([#1](https://github.com/parda21/NWC/issues/1)) and H100
   ([#2](https://github.com/parda21/NWC/issues/2)); community results ([#8](https://github.com/parda21/NWC/issues/8)).
3. Small matrices (qkv, o) reach 1.06–1.18× instead of 1.5×: launch and tail effects; Nsight Compute needs
   the performance-counter permission. Qwen3-0.6B (matrices ≤ 1024 × 2048) is launch-bound and 0.83× native on
   the 4070 (162 vs 196 tokens/s as a CUDA graph) while saving 30 % of VRAM.
4. Second model family (Llama 3.1 8B / Mistral) and perplexity with a common window count
   ([#4](https://github.com/parda21/NWC/issues/4)).
5. llama.cpp port: new GGML tensor type, CPU dequantization, CUDA mmv kernel, converter
   ([#6](https://github.com/parda21/NWC/issues/6)).
6. Prefill (batch > 1) dequantizes into a scratch buffer — no speed gain there.
7. Turing (sm_75) builds without spills (56 registers) but has not been run ([#5](https://github.com/parda21/NWC/issues/5)).

## 7. Lossless coding on top of quantized weights (measured, no kernel)

`scripts/entropy_quant.py` quantizes the 253 linear matrices of Qwen3-4B (4.02 G weights) and measures the
zero-order entropy of the quantized symbols, i.e. what an ideal memoryless coder would reach on top of each
format. RTX 4070, per-matrix, weighted by size:

| format | stored bits/weight | entropy bits/weight | ideal coder | v9 prefix code | 4B model |
|---|---|---|---|---|---|
| BF16 (NWC today) | 16.00 | 10.76 | 0.673 | 0.676 | 5.41 GB |
| int8, per output channel | 8.01 | 6.97 | 0.871 | 2.55 (unusable) | 3.51 GB |
| fp8 e4m3, per output channel | 8.01 | 6.57 | 0.820 | 1.03 (unusable) | 3.30 GB |
| int4, groups of 128 | 4.12 | 3.35 | 0.812 | 1.08 (unusable) | 1.68 GB |

Reading: on BF16 the rank prefix code is within 0.5 % of the entropy because the coded alphabet (128 exponents)
is steep. Quantized alphabets are flat (int8: 256 symbols at ~7 bits), so the unary rank code is the wrong tool
and a rANS/tANS-class coder is needed to collect the 13 % (int8), 18 % (fp8) or 19 % (int4). At the same time
the decoder's time budget per weight shrinks by the same factor as the stored size (2× for int8/fp8, 4× for
int4), because the uncompressed competitor is already that much faster than BF16. Consequence: on top of
quantized formats NWC is a memory saving of 13–19 %, not a speed-up, until the decoder is several times faster
than v9 (tensor-core path, [#3](https://github.com/parda21/NWC/issues/3)).

## 8. FP8 element type (weight-only fp8 e4m3, the fp8 values stored lossless)

Format v9 with `elem = 1` (docs/format.md, section 4b): Qwen3-4B quantized to weight-only fp8 with a per-channel
scale (`quantize_fp8`), the fp8 bytes entropy-coded to 0.866 of their size and decoded by the same kernel.
Baseline is the library's own reference fp8 matvec (`nwc_ref_fp8`: hardware e4m3x2 conversion and HFMA2 on
sm_89+, a shared-memory table on older parts), which reads exactly one byte per weight; cuBLAS BF16 is shown for
context (`scripts/kernbench.py --elem fp8`).

| layer (M × K) | 4070: fp8 ref / NWC-fp8 | ratio | cuBLAS BF16 | A16: fp8 ref / NWC-fp8 | ratio | cuBLAS BF16 |
|---|---|---|---|---|---|---|
| qkv 6144 × 2560 | 0.071 / 0.074 ms | 0.96× | 0.108 ms | 0.128 / 0.179 ms | 0.72× | 0.200 ms |
| o 2560 × 4096 | 0.059 / 0.070 ms | 0.85× | 0.085 ms | 0.090 / 0.129 ms | 0.70× | 0.135 ms |
| gate_up 19456 × 2560 | 0.151 / 0.138 ms | 1.09× | 0.253 ms | 0.388 / 0.538 ms | 0.72× | 0.605 ms |
| down 2560 × 9728 | 0.098 / 0.100 ms | 0.98× | 0.147 ms | 0.193 / 0.275 ms | 0.70× | 0.306 ms |
| lm_head 151936 × 2560 | 0.849 / 0.842 ms | 1.01× | 1.765 ms | 3.053 / 4.222 ms | 0.72× | 4.797 ms |
| sum per token | 14.5 / 14.6 ms | 0.99× | ~26 ms | 31.8 / 44.6 ms | 0.71× | 49.6 ms |

Reading: NWC-fp8 costs the decoder the same per weight as NWC-BF16 (the hot loop is identical), so its time
equals the BF16 kernel's (4070 gate_up 0.138 vs 0.163 ms, A16 0.538 vs 0.522 ms) while the native fp8 competitor
is twice as fast as cuBLAS BF16. Where the BF16 kernel had headroom (4070) fp8 lands at parity with native fp8
and 1.2–2.1× cuBLAS BF16 at 0.43 of the BF16 size; where it was at parity (A16) fp8 is 0.7× native fp8 and
still 1.1× cuBLAS BF16. The fp8 reference reaches 459 GB/s on the 4070 (lm_head) and 128 GB/s on the A16.

**The Ada finding.** The 4070 executes the same SASS at about 17 SM clocks per warp step (2.7 GHz verified
under load with `nvidia-smi -lms 250`), the A16 at 11. Knockouts on the 4070 (gate_up fp8, 0.146 ms): no raw-plane
loads 0.112 ms, no stream refill 0.144, neither 0.140; L2-resident data is as fast as DRAM; prefetching this
block's rows into L1 or the next block into L2 (compile switches `PF_ROWS_L1`, `PF_NEXT_L2`) is neutral to
−10 %. So the kernel is issue- or latency-bound in the SM, not in memory, and Ada needs 1.5× the clocks of
Ampere for it; the cause is open (Nsight Compute is not installed here). Closing that gap would make fp8 on
the 4070 bandwidth-bound (1.14× native fp8) and BF16 on the 4080 Super / 4090 more comfortable.

Model level, Qwen3-4B on the RTX 4070 (`python -m nwc.demo models/Qwen3-4B --fp8 --graph --tokens 128`):

| | native BF16 | NWC (BF16, lossless) | fp8 weight-only, NWC-coded |
|---|---|---|---|
| linear weights | 8.04 GB | 5.55 GB | **3.54 GB** (0.88 of fp8, 0.44 of BF16) |
| VRAM in use | 8.10 GB | 5.67 GB | **3.61 GB** |
| tokens/s, CUDA graph, greedy | 45 | 55 | **73.6** |
| perplexity WikiText-2 (16 × 1024) | 18.03 | = native | 18.15 (+0.7 %, the fp8 quantization) |

Quantized entropies (section 7) put the ceiling for fp8 at 0.828 with an ideal coder on the 4-bit exponent;
the decoder-friendly split costs 0.866 (+4.6 %); with the LUTs, headers and scales the checkpoint lands at 0.88.

## 9. Tensor-core accumulation (layout 1)

Format v10 adds a second block layout, selected per matrix (`NWCWeight(layout=1)`, checkpoints record it; v9
checkpoints keep loading). Blocks are 16 rows × 512 columns and each lane's stream is in the order of the A
fragments of `mma.m16n8k16` (bf16 → fp32): lane (r = lane >> 2, q = lane & 3) holds, per 32-column super-chunk,
columns 8q..8q+7 of rows r and r+8. One PRMT per pair builds a bf16x2 fragment register, x enters as the B
fragment (one 16-byte load per lane and super-chunk, every lane loading its own columns so that all eight columns
of D carry the same sums), the D fragment accumulates over the block, and the shuffle reduction, the eight
accumulators and the sixteen x registers of layout 0 disappear. The raw plane is `[row group][super-chunk][lane]
[16 bytes]` (fp8: 8 nibble bytes), loaded one super-chunk ahead without a bounds check (512 bytes of slack).
sm_75 gets an FFMA fallback in the same order. `tests/test_format_cpu.py` reproduces both layouts.

Kernels per layer shape (`scripts/kernbench.py --layout 0|1`), GPU time, medians:

| layer | 4070 BF16: L0 / L1 | 4070 fp8: L0 / L1 | A16 BF16: L0 / L1 | A16 fp8: L0 / L1 |
|---|---|---|---|---|
| gate_up 19456 × 2560 | 0.163 / 0.170 ms | 0.137 / 0.140 ms | 0.521 / 0.526 ms | 0.536 / **0.468 ms** |
| down 2560 × 9728 | 0.106 / 0.127 ms | 0.096 / 0.102 ms | 0.270 / 0.287 ms | 0.276 / **0.253 ms** |
| lm_head 151936 × 2560 | 1.253 / 1.305 ms | 0.819 / 0.853 ms | 4.538 / 4.511 ms | 4.222 / **3.834 ms** |
| sum per token | 17.9 / 18.6 ms | 14.1 / 15.7 ms | 44.1 / 45.5 ms | 44.6 / **40.0 ms** |

Reading: on the A16 (issue-bound) the tensor-core layout makes fp8 9–13 % faster (0.72× → 0.80× of the native
fp8 matvec) and leaves BF16 at parity; on the 4070 it is at parity for large layers and behind on small ones
(half as many blocks per matrix, so the tail of the persistent grid weighs more). At model level on the 4070 the
fp8 Qwen3-4B runs at 66.9 tokens/s with layout 1 against 73.6 with layout 0 (checkpoint 3.49 instead of 3.54 GB).
Layout 1 therefore stays opt-in (`NWCWeight(layout=1)`, `NWC_LAYOUT=1`) until more GPUs are measured; the A16
model-level numbers are below.

What it took to get there, all measured on the 4070 with fp8 lm_head (layout 0: 0.819 ms): the first version with
16-column chunks, a bounds-checked raw load and fp32 x converted per chunk ran at 0.992 ms; the unconditional
raw load with slack 0.892; x as BF16 without conversion 0.888; 32-column super-chunks with one 16-byte x load and
one 16-byte raw load per two mma 0.853. Knockouts showed the mma and its WARPSYNC cost nothing and the x loads
5 %: what matters on both architectures is the number of memory instructions per pair, not the ALU count. The
expected two clocks per step from removing the PRMT/FFMA pairs did not materialise for BF16, which suggests the
BF16 kernel on the A16 is now limited by its memory access pattern (32 scattered 4-byte stream refills per warp)
rather than by issue; the next lever there is wider refills.

Edge handling worth knowing: a 16-column chunk is not one mma (the super-chunk column mapping interleaves both),
so matrices whose `K16` is an odd multiple of 16 code whole super-chunks with fillers, and the dequantization and
gather kernels guard their stores with `column < K16`; the first version wrote filler columns past the row end
and corrupted neighbouring tensors.
