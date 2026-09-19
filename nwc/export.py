"""Export an NWC checkpoint back to a plain BF16 Transformers checkpoint (bit-identical weights).
usage: python -m nwc.export CHECKPOINT OUT_DIR      # CHECKPOINT: local directory or Hugging Face repo id"""
import sys, argparse
from .checkpoint import export_bf16

ap = argparse.ArgumentParser(description="NWC checkpoint -> BF16 checkpoint")
ap.add_argument("checkpoint", help="NWC checkpoint: local directory or HF repo id, e.g. Parda21/Qwen3-4B-NWC")
ap.add_argument("out", help="output directory (HF format, model.safetensors + config + tokenizer)")
ap.add_argument("--device", default="cuda", help="GPU used for decoding (default cuda)")
a = ap.parse_args()
export_bf16(a.checkpoint, a.out, device=a.device)
