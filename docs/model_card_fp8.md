---
license: apache-2.0
base_model: Qwen/Qwen3-4B
library_name: nwc
tags:
- fp8
- compression
- cuda
- text-generation
pipeline_tag: text-generation
---

# Qwen3-4B-NWC-fp8

Qwen/Qwen3-4B quantized to **weight-only fp8 e4m3** (symmetric per-output-channel scale, activations BF16) and
then stored in **NWC** (Neural Weight Compression) format: the fp8 values are entropy-coded to 0.87 of their
size and decoded inside the CUDA matvec kernel. Relative to the fp8 checkpoint nothing is lost (the fp8 bytes
come back bit for bit); relative to the BF16 original it is an ordinary fp8 quantization.

| | Qwen3-4B (BF16) | fp8 weight-only | **Qwen3-4B-NWC-fp8** |
|---|---|---|---|
| weights | 8.04 GB | 4.02 GB | **3.54 GB** (0.88 of fp8, 0.44 of BF16) |
| VRAM in use, batch 1 | 8.10 GB | | **3.61 GB** |
| tokens/s, RTX 4070, CUDA graph, greedy | 45 | | **73.6** |
| weight kernels vs a native fp8 matvec (RTX 4070) | | 1× | 0.85–1.09× (parity) |
| perplexity WikiText-2 (16 × 1024 tokens) | 18.03 | | 18.15 |

Perplexity WikiText-2, 16 windows × 1024 tokens, RTX 4070: BF16 original 18.03, this checkpoint 18.15 (+0.7 %; the effect of the fp8 quantization, NWC itself changes nothing).

## Usage

```bash
pip install neural-weight-compression transformers accelerate
python -m nwc.demo Parda21/Qwen3-4B-NWC-fp8 --load --graph
```

```python
from nwc import load_pretrained
from transformers import AutoTokenizer

model = load_pretrained("Parda21/Qwen3-4B-NWC-fp8")
tok = AutoTokenizer.from_pretrained("Parda21/Qwen3-4B-NWC-fp8")
ids = tok("Question: What is a binary tree? Answer:", return_tensors="pt").input_ids.cuda()
print(tok.decode(model.generate(ids, max_new_tokens=64, do_sample=False)[0]))
```

Requirements: NVIDIA GPU with compute capability 7.5 or newer (8.0+ measured), CUDA driver for CUDA 12.6+,
PyTorch with CUDA, `neural-weight-compression >= 0.10`. Batch-1 generation runs through the fused kernel with
the per-channel scale applied at the end; prefill dequantizes to BF16 and calls cuBLAS.

Speed: on the RTX 4070 the fused kernel matches a native fp8 weight-only matvec (same GPU time, 13 % fewer
bytes); on an NVIDIA A16 it reaches 0.7× of it (the decoder is the limit there) while still beating cuBLAS on
the BF16 model. Details: https://github.com/parda21/NWC, docs/results.md section 8.

## How it was made

```python
from nwc import fuse, convert, save_pretrained
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cpu")
fuse(model, elem="fp8"); convert(model, elem="fp8")     # quantize_fp8 per matrix, then NWC-encode the fp8 bytes
save_pretrained(model, "Qwen3-4B-NWC-fp8", tokenizer=tok, base_model="Qwen/Qwen3-4B")
```

`python -m nwc.export Parda21/Qwen3-4B-NWC-fp8 OUT` writes a BF16 checkpoint of fp8 × scale (rounded to BF16).
License of the weights: Apache-2.0, as the base model.
