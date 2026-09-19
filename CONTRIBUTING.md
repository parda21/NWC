# Contributing

Thanks for looking at NWC. The most valuable contributions right now, in this order:

1. **Benchmark results on GPUs we do not have.** Especially H100 (PCIe and SXM), A100, RTX 4080/4090, RTX 30xx,
   Turing (T4, RTX 20xx) and Blackwell. Two commands, then paste the output into a
   [benchmark issue](https://github.com/parda21/NWC/issues/new?template=benchmark_result.yml):

   ```bash
   pip install neural-weight-compression transformers accelerate
   python -m nwc.doctor
   python -m nwc.demo Parda21/Qwen3-4B-NWC --load --graph      # and: python -m nwc.demo Qwen/Qwen3-4B --native --graph
   ```

   From a clone, `python scripts/kernbench.py --runs 30` gives the per-layer kernel comparison as well.
2. **Other model families.** Anything HF Transformers loads in BF16 should work; reports (working or not) with
   the model id are welcome.
3. **Kernel work.** See the roadmap in the README and the open items in `docs/results.md`. The tensor-core
   accumulation path and the llama.cpp port are the two big ones.

## Before opening a pull request

- Read [AGENTS.md](AGENTS.md); it is written for coding agents but it is the rule book for humans too.
- `python tests/test_format_cpu.py` passes (no GPU needed) and, on a GPU, `tests/test_k.py`, `tests/test_gather.py`
  and `tests/test_checkpoint.py` pass.
- A kernel change comes with `scripts/kernbench.py` output before and after, on a named GPU.
- Bit-exactness is not negotiable: dequantized weights equal the originals.
- One topic per PR, English, no attribution trailers in commit messages.

## Reporting a problem

Run `python -m nwc.doctor` and paste its output into the
[bug report](https://github.com/parda21/NWC/issues/new?template=bug_report.yml). It tells us the GPU, the driver,
the torch build and whether the kernel round trip works, which answers most questions before they are asked.
