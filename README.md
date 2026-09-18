# NWC — Neural Weight Compression

Lossless BF16 weight compression with a fused GPU matvec. NWC keeps the weights of a transformer in VRAM **losslessly compressed by ~31 %** and decodes them
**inside the matrix-vector kernel, in registers**. No decompressed weight ever touches memory. Because token
generation is memory-bound, reading fewer bytes makes the kernel faster than the uncompressed cuBLAS
matvec — on consumer GPUs by a wide margin, and since format v9 also on a bandwidth-starved Ampere part.

Qwen3-4B, greedy decoding as a CUDA graph, same tokens as the reference (64/64):

| | RTX 4070 (452 GB/s) | NVIDIA A16 (vGPU 16Q, 10 SMs, 165 GB/s) |
|---|---|---|
| VRAM, native BF16 → NWC | 8.10 GB → **5.67 GB** | 8.10 GB → **5.67 GB** |
| tokens/s, native → NWC | 45 → **55** (1.22×) | 16.8 → **18.2** (1.08×) |
| GPU time per token, native → NWC | 20.4 ms → **18.9 ms** | 54.1 ms → **52.1 ms** |
| weight kernels vs cuBLAS (5 layer shapes) | 1.17–1.55× | 1.06–1.18× |
| logits vs native | bit-identical | within cuBLAS' own cross-GPU spread (max 0.20; argmax identical) |

Same model, same GPU, against [DFloat11](https://github.com/LeanModels/DFloat11) (NeurIPS 2025), which
compresses the same exponent bits with Huffman coding but decodes into a buffer before the matmul
(RTX 4070, GPU time per token measured with `torch.profiler`, HF eager; `scripts/compare_df11.py`):

| | native BF16 | DFloat11 | NWC |
|---|---|---|---|
| VRAM | 8.10 GB | 5.73 GB | **5.67 GB** |
| GPU time per token | 20.4 ms | 56.1 ms (2.74× native) | **18.9 ms** (0.93× native) |
| weight kernels per token | gemv 17.2 ms | decode 36.6 + gemv 16.2 ms | **17.3 ms**, decode fused |
| logits vs native | — | bit-identical (paper) | bit-identical (measured) |

Same size, opposite speed: DFloat11's decoded weights pass through memory twice, NWC reads only the
compressed bytes. DFloat11 has not been measured under a CUDA graph (its decode runs in separate cupy
kernels), so GPU time is the comparable figure.

Decompressed weights are bit-exact on both GPUs (`tests/test_k.py`, `tests/test_gather.py`). Qwen2.5-7B in
full BF16 fits a 12 GB card (14.1 → 9.7 GB of weights). Measurements, negative results and the cost model:
[docs/results.md](docs/results.md). Format and decoder design: [docs/format.md](docs/format.md).

## How it works

- **Byte planes.** A BF16 weight is split into its mantissa byte (7.97 bits of entropy, stored raw) and its
  exponent/sign byte (2.6–2.9 bits of entropy, entropy-coded). That is where the 31 % come from; ZipNN and
  DFloat11 land at the same figure because it is the entropy of the data.
- **Prefix code instead of rANS (format v9).** The exponent is ranked by frequency and written as a unary
  rank code plus a raw sign bit (escape for rare values). A 12-bit peek into a 16 KB lookup table in shared
  memory yields two complete codes per step; the bit window is shifted with funnel shifts and refilled with a
  single wide multiply-add. Measured on the A16: 11–13 SM clocks per 64 weights, below the 13.6 needed to
  match cuBLAS.
- **Fused kernel.** One persistent block per SM (1024 threads); each lane decodes its own bit stream for a
  block of 8 rows × 512 columns, keeps the 16 matching activations in registers and accumulates eight row
  sums; a transposed shuffle reduction writes partial sums per (row, column block). Works for any matrix
  shape (padding costs 2 bits per filler weight, no mantissa).
- **Dequantization and embedding lookup** (tied `lm_head`) run on the same decoder; prefill (batch > 1)
  dequantizes into a temporary BF16 buffer and calls cuBLAS.

## Install

```bash
pip install neural-weight-compression            # wheels for Windows and Linux with prebuilt kernels (sm_80–sm_90, PTX for newer)
pip install transformers accelerate              # for the model helpers and the demo
python -m nwc.demo Qwen/Qwen3-4B --graph         # load, compress, generate, report VRAM and tokens/s
```

```python
from nwc import fuse, convert, save_pretrained, load_pretrained
fuse(model); convert(model)                      # nn.Linear -> NWCLinear, model to the GPU
save_pretrained(model, "Qwen3-4B-NWC", tokenizer=tok, base_model="Qwen/Qwen3-4B")
model = load_pretrained("Qwen3-4B-NWC")          # later: straight to the GPU, no BF16 originals needed
```

A ready-made compressed checkpoint is on Hugging Face: [Parda21/Qwen3-4B-NWC](https://huggingface.co/Parda21/Qwen3-4B-NWC)
(`python -m nwc.demo Parda21/Qwen3-4B-NWC --load --graph` after `hf download Parda21/Qwen3-4B-NWC --local-dir Qwen3-4B-NWC`).

Requirements: an NVIDIA GPU with compute capability 8.0 or newer, a driver for CUDA 12.6+, PyTorch with CUDA.
Installing from source (`pip install git+https://github.com/parda21/NWC`) needs `nvcc`; run
`python -m nwc.build` once to compile the kernel into the package.

## Build and test from the repository

Requirements: CUDA 12.6+ (13.x on Windows), a GPU with sm_80 or newer, Python 3.12 with PyTorch (CUDA build)
and `transformers`. Windows: Visual Studio 2022 Build Tools.

```powershell
.\build.ps1                       # Windows: build/nwc_ops.dll (add -Fatbin for the package library)
```

```bash
./build.sh sm_86                  # Linux: build/nwc_ops.so (nvcc in PATH, pip wheel, or a CUDA container)
```

```bash
python scripts/download_models.py Qwen/Qwen3-4B [--df11 DFloat11/Qwen3-4B-DF11]
python scripts/make_wraw.py       # test matrix data/W.raw from a local safetensors model
python tests/test_k.py            # dequantization bit-exact, token path, fp32 path — odd shapes included
python tests/test_gather.py       # embedding lookup bit-exact
python tests/test_checkpoint.py   # save/load of a compressed checkpoint on a small random model, no download
python tests/test_nwc_torch.py    # PyTorch bridge, speed vs cuBLAS on one matrix
```

## Run a model

```bash
python scripts/kernbench.py --runs 30                              # kernel vs cuBLAS per layer shape
python scripts/graph_decode.py --mode nwc --fusion --tokens 256    # tokens/s as a CUDA graph, vs HF generate
python scripts/compare_df11.py --mode native|df11|nwc --fusion     # VRAM, GPU time per token, logits
python scripts/bench_model.py --model <path> [--native]            # any HF causal LM in BF16, plus perplexity
```

## Repository layout

```
csrc/nwc_ops.cu   the kernel library: encoder, fused matvec, dequantization, gather
nwc/              Python package: nwc_torch (NWCLinear, NWCEmbedding, convert, fuse), checkpoint (save/load),
                  demo, build; lib/ holds the compiled kernel
tests/            bit-exactness and numerical tests
scripts/          kernbench, graph_decode, compare_df11, bench_model, make_wraw, download_models, bench_all.sh
docs/             results.md (measurements, cost model), format.md (format and decoder design), paper/, announce.md
experiments/      kernel iterations v1–v9, micro-benchmarks (gather_bench.cu, pipe_bench.cu), format experiments,
                  reference implementations and CLI tools of earlier formats; history, not product
build/ data/ models/ dist/   build artefacts, test matrices, checkpoints, wheels — git-ignored
```

## Status and limitations

- Batch 1 (token generation) is the fast path. Prefill is not faster than native, it only saves memory.
- Faster than native requires the decoder to keep up with the memory system: the v9 decoder delivers
  7–8 GB/s of compressed data per SM·GHz. That is enough for GDDR cards and for the A16, projected enough
  for an H100 PCIe, and probably not yet for an H100 SXM (see docs/results.md). The planned next step is a
  tensor-core accumulation path.
- Tested with Qwen2.5 and Qwen3; any HF causal LM in BF16 should work (the encoder only needs the weights).

## Sponsor

This project is sponsored by [cloo GmbH](https://github.com/cloogmbh).

## License

Apache License 2.0, Copyright 2026 Paul Otto. See [LICENSE](LICENSE).

## About this project

Was this project created with the help of AI? Yes.
Do I care, as long as it works? No.
