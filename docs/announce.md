# Announcement drafts

Use after the 4080 Super and H100 numbers are in; replace the bracketed placeholders.

## Show HN

**Title** (≤ 80 chars):
Show HN: Lossless BF16 LLM weight compression that runs faster than cuBLAS

**Text:**

NWC keeps a transformer's BF16 weights in VRAM compressed by 31 % and decodes them inside the matvec
kernel, in registers. No decompressed weight is ever written to memory. Because token generation is
memory-bound, reading fewer bytes makes the fused kernel faster than the plain cuBLAS matvec.

Numbers (Qwen3-4B, greedy decoding as a CUDA graph, output tokens identical to the reference):
- RTX 4070: 8.10 → 5.67 GB, 45 → 55 tokens/s, logits bit-identical to native
- NVIDIA A16 (10 SMs, 165 GB/s): 16.8 → 18.2 tokens/s
- [RTX 4080 Super: …] [H100 PCIe/SXM: …]
- Qwen2.5-7B in full BF16 fits a 12 GB card

How: the mantissa byte of a BF16 is incompressible (7.97 bits of entropy) and stays raw; the
exponent/sign byte has ~2.8 bits of entropy and gets a rank prefix code. A 12-bit lookup table in shared
memory yields two codes per step, a 64-bit window is shifted with funnel shifts and refilled with one wide
multiply-add. On Ampere that is 11–13 SM clocks per 64 weights, which is below what the memory system
needs. The interesting part was measuring where the clocks go: per-instruction issue rates, pipe overlap
(PRMT and IMAD.HI block both pipes on GA10x), and code size — moving a rarely taken slow path out of the
hot loop was worth 3.5 clocks per step.

Compared with DFloat11 (same 30 %, Huffman, decode into a buffer then matmul): same size, but DFloat11
takes 2.7× the native GPU time per token on the 4070 because the decoded weights pass through memory
twice.

Apache-2.0, `pip install neural-weight-compression`, then `python -m nwc.doctor` and
`python -m nwc.demo Parda21/Qwen3-4B-NWC --load --graph` (ready-made checkpoint; a 0.8 GB one for a one-minute
smoke test is there too). No lock-in: `python -m nwc.export` gives the BF16 checkpoint back bit for bit.
Limits: batch-1 decoding is the fast path; prefill only saves memory. HBM parts need more decoder
throughput per SM; a tensor-core accumulation path is the next step.

## r/LocalLLaMA

**Title:** Run a 7B model in full BF16 on 12 GB, faster than native: lossless weight compression decoded inside the matvec (open source, Apache-2.0)

**Body:**

I built NWC: the weights stay in VRAM losslessly compressed (−31 %), and the CUDA kernel decodes them in
registers while it multiplies. No quantization, no quality loss — the decompressed weights are bit-exact
and on my 4070 the logits are bit-identical to the unmodified model.

- Qwen3-4B: 8.10 → 5.67 GB, 45 → 55 tokens/s (CUDA graph, greedy), identical tokens
- Qwen2.5-7B BF16 on a 12 GB card: 14.1 → 9.7 GB of weights
- Also faster than native on a bandwidth-starved NVIDIA A16 (10 SMs); [4080 Super: …]
- vs DFloat11: same size, but 2.7× faster on the 4070 (they decode into a buffer, NWC decodes in the kernel)

Works with HF Transformers: `fuse(model); convert(model)` replaces every nn.Linear. Compressed checkpoints
can be saved and loaded without the original weights (`save_pretrained` / `load_pretrained`), and exported back
to plain BF16 bit for bit, so there is no lock-in. Ready-made checkpoints on HF (Qwen3-4B, and a 0.6B for a
one-minute smoke test); `python -m nwc.doctor` tells you whether your setup works before you download anything.
Windows and Linux wheels with prebuilt kernels for sm_75–sm_90 (Blackwell via PTX).

Not yet: LM Studio / Ollama (needs a llama.cpp port, on the roadmap), vLLM.

Caveats, honestly: batch 1 only (prefill dequantizes into a buffer, no speedup there), and on HBM cards
(A100, H100 SXM) the decoder is not yet fast enough to beat native; that is the next piece of work.

Repo: https://github.com/parda21/NWC — measurements, negative results and the full cost model are in docs/.

## X / Twitter (thread opener)

Lossless BF16 weight compression that is *faster* than cuBLAS: NWC decodes the weights inside the
matvec, in registers, never writing them back. Qwen3-4B: 8.1 → 5.7 GB and 45 → 55 tok/s on a 4070,
tokens identical. Apache-2.0. github.com/parda21/NWC

(reply) Same 31 % as DFloat11, opposite speed: their decode kernel alone takes longer than the whole
native forward on a 4070 because the decoded weights pass through memory twice. NWC reads only the
compressed bytes. Head-to-head numbers in the README.

(reply) The fun part was the cost model: on Ampere GA10x only FFMA issues at full rate, IMAD/PRMT/SHF are
half rate, PRMT and IMAD.HI block both pipes, and code size matters — a 147 KB kernel vs 23 KB was 3.5
clocks per step. Details: docs/format.md

## Issue / discussion for the DFloat11 repository (optional, be polite)

Title: Comparison with fused in-kernel decoding (NWC)

Hi, thanks for DFloat11 — the exponent-entropy observation is what started this. I built a variant that
decodes inside the matvec instead of into a buffer, so batch-1 decoding gets faster rather than slower,
and measured both head-to-head on the same model and GPU (Qwen3-4B, RTX 4070, torch.profiler GPU time
per token: native 20.4 ms, DFloat11 56.1 ms, NWC 18.9 ms; VRAM 5.73 vs 5.67 GB). Write-up and code:
https://github.com/parda21/NWC. If I measured DFloat11 unfairly anywhere (settings, version 0.5.0, eager
mode), I'd like to fix that — happy to rerun with whatever configuration you consider representative.
