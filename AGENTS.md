# Working in this repository (for coding agents and new contributors)

NWC keeps BF16 transformer weights losslessly compressed in VRAM and decodes them inside the CUDA matrix-vector
kernel. Everything here serves one measured claim: the compressed matvec is faster than cuBLAS on the
uncompressed weights, with bit-identical results. Read this file before changing anything.

## Map

| path | what | touch it? |
|---|---|---|
| `csrc/nwc_ops.cu` | the whole kernel library: host encoder, lookup table, fused matvec, dequantization, gather | yes, carefully; every change is measured |
| `nwc/nwc_torch.py` | ctypes bridge, `NWCWeight`, `NWCLinear`, `NWCEmbedding`, `convert`, `fuse` | yes |
| `nwc/checkpoint.py` | `save_pretrained`, `load_pretrained`, `export_bf16` (checkpoint format) | yes; keep old checkpoints loadable |
| `nwc/demo.py`, `nwc/doctor.py`, `nwc/export.py`, `nwc/build.py` | user-facing entry points (`python -m nwc.<name>`) | yes |
| `tests/` | correctness; see the table below | add a test with every feature |
| `scripts/` | benchmarks; the numbers in the README come from here | yes |
| `docs/` | `results.md` (measurements, cost model), `format.md` (format spec), `paper/`, `announce.md` | keep consistent with the code |
| `experiments/` | kernel iterations v1–v9, micro-benchmarks, earlier formats | **history, do not edit or "clean up"** |
| `build/`, `data/`, `models/`, `dist/`, `nwc/lib/*.dll|so` | git-ignored artefacts | — |

## Commands

```bash
python -m nwc.doctor                      # is this machine ready? (GPU, driver, library, round trip)
.\build.ps1                               # Windows: build/nwc_ops.dll for the local GPU (sm_89 by default)
./build.sh sm_86                          # Linux: build/nwc_ops.so
python -m nwc.build                       # fatbin into nwc/lib (what the wheel ships)
```

| test | needs | checks |
|---|---|---|
| `python tests/test_format_cpu.py` | the library only, **no GPU** | pure-Python reference decoder reproduces the encoder's output bit-exactly (format spec) |
| `python tests/test_k.py` | GPU, `data/W.raw` (`scripts/make_wraw.py`) | dequantization bit-exact, token path, fp32 path, odd shapes |
| `python tests/test_gather.py` | GPU, `data/W.raw` | embedding lookup bit-exact |
| `python tests/test_checkpoint.py` | GPU, `transformers` | save → load → export round trip on a small random model (BF16 and fp8), no download |
| `python tests/test_fp8.py` | GPU | fp8 element type: fp8 values bit-exact (dequant, gather), matvec vs float reference, reference kernel, NaN rejected |
| `python tests/test_nwc_torch.py` | GPU, `data/W.raw` | PyTorch bridge, speed vs cuBLAS on one matrix |

Benchmarks: `scripts/kernbench.py` (kernel vs cuBLAS per layer shape; `--elem fp8` against the reference fp8 matvec), `scripts/graph_decode.py --mode nwc --fusion`
(tokens/s as a CUDA graph), `scripts/compare_df11.py` (against DFloat11). CI (`.github/workflows/ci.yml`) compiles
the library for every architecture and runs the CPU test; GPU tests are run by hand before a release and the
output is pasted into the release notes.

## Rules

1. **Compute in C/CUDA, orchestration in Python.** No hot loop in Python; the Python layer moves tensors and calls
   the library.
2. **Any matrix shape must work.** The format pads to 8 rows × 512 columns; never add a `K % 512 == 0`-style
   requirement. `tests/test_k.py` covers shapes like 13 × 40 and 3 × 4097 for that reason.
3. **Lossless means bit-identical.** Dequantized weights must equal the originals; tests compare with
   `torch.equal`, not with a tolerance. Only fp32 summation order may differ in matvec outputs.
4. **Numbers are measured, never estimated in the text.** Every figure in `README.md` and `docs/results.md` comes
   from a script in `scripts/` on named hardware. Projections are labelled as projections. Measure after the GPU
   has warmed up, as medians, never right after compilation.
5. **Kernel changes are judged by `scripts/kernbench.py`** on the A16 (issue-bound, the hard case) and by clocks per
   warp step (`docs/results.md`, section 4). A change that is not faster is a compile switch or an entry in
   "what did not work", not a revert of history.
6. **Keep the format version honest.** A change to the bit stream or the tensor layout bumps `VERSION` in
   `csrc/nwc_ops.cu`; `load_pretrained` refuses checkpoints of other versions.
7. **English everywhere** in code, comments, docs and commit messages.
8. **Commits**: one topic per commit, imperative subject (`nwc: add doctor command`), no attribution trailers,
   no "cleanup" commits that rewrite history. Never force-push `main`.

## Things that bit us (so you do not repeat them)

- ptxas folds instruction chains with constant operands: micro-benchmarks need data-dependent operands.
- `mad.wide` with a 64-bit addend gets split unless the 64-bit pair arises naturally; a predicated `mad.wide`
  becomes SEL + split.
- Kernel code size is a first-order effect on Ampere (147 KB → 23 KB was 3.5 clocks per step): do not unroll the
  row loop.
- `#pragma unroll N` does not expand macros under Linux nvcc; use `_Pragma`.
- `__byte_perm` ignores the sign-replication bit of the selector; use inline `prmt.b32`.
- Building a random test model with `.to(torch.bfloat16)` converts `inv_freq` buffers too; use
  `from_config(cfg, dtype=torch.bfloat16)`.

## Definition of done

All tests in the table pass on a GPU, `tests/test_format_cpu.py` passes, `scripts/kernbench.py` shows no regression
on the shapes it prints, and the docs say what the code does.
