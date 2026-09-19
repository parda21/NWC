"""Perplexity with the llama.cpp method, so NWC / native / fp8 points can sit on one curve with GGUF models scored by
`llama-perplexity` on the same text file: the text is tokenized once (no special tokens), cut into consecutive chunks
of n_ctx tokens, and in every chunk the tokens of the second half are scored (logits at positions n_ctx/2 .. n_ctx-2
predict tokens n_ctx/2+1 .. n_ctx-1), perplexity = exp(sum nll / count). Prefill path (dequantization + cuBLAS).
usage: python scripts/ppl_curve.py --model Qwen/Qwen3-8B --text data/wiki.test.raw [--elem bf16|fp8] [--native]
       [--native-fp8] [--load] [--save DIR] [--n-ctx 2048] [--chunks N] [--out results.json]"""
import os, sys, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="HF model id / path (BF16), or an NWC checkpoint with --load")
ap.add_argument("--text", default=os.path.join(ROOT, "data", "wiki.test.raw"))
ap.add_argument("--elem", default="bf16", choices=["bf16", "fp8"])
ap.add_argument("--native", action="store_true", help="uncompressed BF16 (needs the whole model in VRAM, or --offload)")
ap.add_argument("--offload", type=float, default=0, help="with --native: GB of VRAM for the weights, the rest streams from CPU RAM (accelerate)")
ap.add_argument("--native-fp8", action="store_true", help="uncompressed weight-only fp8 (same quantization as --elem fp8)")
ap.add_argument("--load", action="store_true", help="--model is an NWC checkpoint")
ap.add_argument("--save", default=None, help="write the converted model as an NWC checkpoint")
ap.add_argument("--n-ctx", type=int, default=2048)
ap.add_argument("--chunks", type=int, default=0, help="limit the number of chunks (0: all)")
ap.add_argument("--out", default=None, help="append the result to this JSON file")
ap.add_argument("--label", default=None)
a = ap.parse_args()

from transformers import AutoTokenizer, AutoModelForCausalLM
from nwc.checkpoint import resolve

t0 = time.time()
path = resolve(a.model)
if not a.load and os.path.exists(os.path.join(path, "nwc_config.json")): a.load = True
tok = AutoTokenizer.from_pretrained(path)
bytes_w = None
if a.load:
    from nwc.checkpoint import load_pretrained
    model = load_pretrained(path)
    variant = "nwc-" + ("fp8" if any(getattr(getattr(m, "w", None), "elem", 0) for m in model.modules()) else "bf16")
elif a.native and a.offload:
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
                                                 max_memory={0: f"{a.offload}GiB", "cpu": "48GiB"})
    variant = "bf16"
else:
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    if a.native:
        model.cuda(); variant = "bf16"
    elif a.native_fp8:
        from nwc.nwc_torch import convert_ref_fp8
        convert_ref_fp8(model); variant = "fp8"
    else:
        from nwc.nwc_torch import convert, fuse
        fuse(model, elem=a.elem)
        bytes_w, _ = convert(model, elem=a.elem); variant = "nwc-" + a.elem
model.eval()
if a.save and variant.startswith("nwc"):
    from nwc.checkpoint import save_pretrained
    c = save_pretrained(model, a.save, tokenizer=tok, base_model=None if a.load else a.model)
    print(f"checkpoint written: {a.save} ({c['bytes_nwc']/1e9:.2f} GB of compressed matrices)")
# size of everything on the GPU: compressed matrices + the rest (norms, biases, native tensors)
from nwc.nwc_torch import NWCWeight
seen, total = set(), 0
for m in model.modules():
    w = getattr(m, "w", None)
    if isinstance(m, torch.nn.Module) and hasattr(m, "g") and isinstance(m.g, dict): w = m.g["lin"].w   # fused projection heads
    if isinstance(w, NWCWeight) and id(w) not in seen: seen.add(id(w)); total += w.bytes
    if m.__class__.__name__ == "RefFP8Linear": total += m.w8.numel() + m.scale.numel() * 4
for p in model.parameters(): total += p.numel() * p.element_size()
for b in model.buffers(): total += b.numel() * b.element_size() if b.dtype != torch.bool else 0
print(f"{variant}: loaded in {time.time()-t0:.0f} s, {total/1e9:.2f} GB of weights, VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB")

text = open(a.text, encoding="utf-8").read()
ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
n_ctx = a.n_ctx; first = n_ctx // 2
n_chunk = ids.numel() // n_ctx
if a.chunks: n_chunk = min(n_chunk, a.chunks)
print(f"{ids.numel()} tokens, {n_chunk} chunks of {n_ctx}, scoring positions {first}..{n_ctx-2} of each")
nll, count = 0.0, 0
t1 = time.time()
with torch.inference_mode():
    for c in range(n_chunk):
        x = ids[c * n_ctx:(c + 1) * n_ctx].unsqueeze(0).cuda()
        logits = model(x, use_cache=False).logits[0, first:n_ctx - 1].float()
        tgt = x[0, first + 1:n_ctx]
        nll += F.cross_entropy(logits, tgt, reduction="sum").item(); count += tgt.numel()
        if (c + 1) % 8 == 0 or c + 1 == n_chunk:
            print(f"[{c+1}/{n_chunk}] ppl {torch.exp(torch.tensor(nll/count)).item():.4f}  ({(time.time()-t1)/(c+1):.1f} s/chunk)", flush=True)
ppl = float(torch.exp(torch.tensor(nll / count)))
res = {"label": a.label or f"{a.model} {variant}", "model": a.model, "variant": variant, "elem": a.elem, "bytes": total,
       "bytes_matrices": bytes_w, "ppl": ppl, "nll": nll, "count": count, "n_ctx": n_ctx, "chunks": n_chunk, "text": os.path.basename(a.text),
       "gpu": torch.cuda.get_device_name(0)}
print(json.dumps(res))
if a.out:
    prev = json.load(open(a.out)) if os.path.exists(a.out) else []
    prev.append(res); json.dump(prev, open(a.out, "w"), indent=1)
