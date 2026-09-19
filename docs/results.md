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
