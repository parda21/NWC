---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: nwc
tags:
- lossless
- compression
- bf16
- cuda
- text-generation
pipeline_tag: text-generation
---

# Qwen3-0.6B-NWC

The small one: Qwen/Qwen3-0.6B with all linear layers and the tied embedding stored in **NWC** (Neural Weight
Compression) format, losslessly compressed BF16 that is decoded inside the CUDA matvec kernel. It exists so that
you can check your setup in a minute (0.8 GB download) before pulling
[Parda21/Qwen3-4B-NWC](https://huggingface.co/Parda21/Qwen3-4B-NWC), where the speed gain shows.

| | Qwen3-0.6B (BF16) | **Qwen3-0.6B-NWC** |
|---|---|---|
| weights | 1.19 GB | **0.82 GB** (rate 0.690) |
| VRAM in use, batch 1 | 1.19 GB | **0.83 GB** |
| tokens/s, RTX 4070, CUDA graph, greedy | 196 | 162 |
| weights vs original | — | bit-identical |

The 0.6B model is **not faster** than native on a large GPU: its matrices are small (1024 × 2048 and smaller) and
decoding is bound by kernel launches, not by memory bandwidth, so the fused kernel has nothing to win. It saves
memory and proves the setup. Greedy output matches the BF16 model for the first 29 tokens of the reference
prompt, then the two diverge on an exact tie of the top two logits (fp32 summation order; max logit difference
0.36 on a range of 28 before the tie).

## Usage

```bash
pip install neural-weight-compression transformers accelerate
python -m nwc.doctor                                          # GPU, driver, library, kernel round trip
python -m nwc.demo Parda21/Qwen3-0.6B-NWC --load --graph      # downloads, loads, generates, reports tokens/s
```

```python
from nwc import load_pretrained
from transformers import AutoTokenizer

model = load_pretrained("Parda21/Qwen3-0.6B-NWC")
tok = AutoTokenizer.from_pretrained("Parda21/Qwen3-0.6B-NWC")
ids = tok("Question: What is a binary tree? Answer:", return_tensors="pt").input_ids.cuda()
print(tok.decode(model.generate(ids, max_new_tokens=64, do_sample=False)[0]))
```

Back to a plain BF16 checkpoint (bit-identical, for tools that do not know NWC):
`python -m nwc.export Parda21/Qwen3-0.6B-NWC Qwen3-0.6B`

Requirements: NVIDIA GPU with compute capability 7.5 or newer (8.0+ measured), CUDA driver for CUDA 12.6+,
PyTorch with CUDA.

## How it was made

```python
from nwc import fuse, convert, save_pretrained
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cpu")
fuse(model); convert(model)
save_pretrained(model, "Qwen3-0.6B-NWC", tokenizer=tok, base_model="Qwen/Qwen3-0.6B")
```

Format and measurements: https://github.com/parda21/NWC. License of the weights: Apache-2.0, as the base model.
