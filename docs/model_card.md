---
license: apache-2.0
base_model: Qwen/Qwen3-4B
library_name: nwc
tags:
- lossless
- compression
- bf16
- cuda
- text-generation
pipeline_tag: text-generation
---

# Qwen3-4B-NWC

Qwen/Qwen3-4B with all linear layers and the tied embedding stored in **NWC** (Neural Weight Compression)
format: losslessly compressed BF16, decoded inside the CUDA matvec kernel. The weights are bit-exact
with the original checkpoint; no quantization is involved.

| | Qwen3-4B (BF16) | **Qwen3-4B-NWC** |
|---|---|---|
| weights | 8.04 GB | **5.55 GB** (rate 0.689) |
| VRAM in use, batch 1 | 8.10 GB | **5.67 GB** |
| tokens/s, RTX 4070, CUDA graph, greedy | 45 | **55** |
| tokens/s, NVIDIA A16 (vGPU 16Q) | 16.8 | **18.2** |
| output tokens vs original (greedy, 64) | — | identical |

## Usage

```bash
pip install neural-weight-compression transformers accelerate
```

```python
from nwc import load_pretrained
from transformers import AutoTokenizer

model = load_pretrained("Parda21/Qwen3-4B-NWC")        # no BF16 originals needed, loads straight to the GPU
tok = AutoTokenizer.from_pretrained("Parda21/Qwen3-4B-NWC")
ids = tok("Question: What is a binary tree? Answer:", return_tensors="pt").input_ids.cuda()
print(tok.decode(model.generate(ids, max_new_tokens=64, do_sample=False)[0]))
```

Or from the command line: `python -m nwc.demo Parda21/Qwen3-4B-NWC --load --graph`. A 0.8 GB smoke test of the same
setup: [Parda21/Qwen3-0.6B-NWC](https://huggingface.co/Parda21/Qwen3-0.6B-NWC).

Back to a plain BF16 checkpoint (bit-identical, for tools that do not know NWC):
`python -m nwc.export Parda21/Qwen3-4B-NWC Qwen3-4B`

Requirements: NVIDIA GPU with compute capability 7.5 or newer (8.0+ measured; Blackwell via PTX JIT), CUDA driver
for CUDA 12.6+, PyTorch with CUDA. `python -m nwc.doctor` checks the setup. Batch-1 generation runs through the fused kernel;
prefill (batch > 1) dequantizes into a temporary BF16 buffer and uses cuBLAS.

## How it was made

```python
from nwc import fuse, convert, save_pretrained
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cpu")
fuse(model); convert(model)
save_pretrained(model, "Qwen3-4B-NWC", tokenizer=tok, base_model="Qwen/Qwen3-4B")
```

Format and measurements: https://github.com/parda21/NWC (docs/format.md, docs/results.md).
License of the weights: Apache-2.0, as the base model.
