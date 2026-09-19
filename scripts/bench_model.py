"""Any HF causal LM in BF16: load NWC-compressed (or native), generate, GPU time per token, perplexity on WikiText-2.
usage: python scripts/bench_model.py --model PATH [--native] [--tokens N] [--ppl-windows N] [--cpu-compare]"""
import os, sys, time, argparse, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from transformers import AutoTokenizer, AutoModelForCausalLM
from nwc.nwc_torch import convert, fuse

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=f"{ROOT}/models/Qwen2.5-7B")
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--ppl-windows", type=int, default=16)
ap.add_argument("--window", type=int, default=1024, help="window length for the perplexity")
ap.add_argument("--cpu-compare", action="store_true", help="compare logits against native BF16 on the CPU (slow)")
ap.add_argument("--native", action="store_true", help="without NWC: uncompressed model on the GPU (baseline)")
ap.add_argument("--fusion", action="store_true", help="fuse q/k/v and gate/up")
ap.add_argument("--elem", default="bf16", choices=["bf16", "fp8"], help="fp8: weight-only fp8 e4m3 (per-channel scale) before compression")
ap.add_argument("--logits-file", default=None, help="save/compare the logits of the first prompt (.pt)")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
model.eval()
print(f"loaded (CPU, BF16) in {time.time()-t0:.0f} s")

prompt = "The compression of model weights"
ids = tok(prompt, return_tensors="pt").input_ids

ref_logits = None
if a.cpu_compare:
    with torch.no_grad(): ref_logits = model(ids).logits[0, -1].float().clone()
    print("reference logits (native BF16, CPU) computed")

t0 = time.time()
if a.native:
    model.cuda(); torch.cuda.synchronize()
    print(f"native on the GPU in {time.time()-t0:.0f} s; VRAM in use: {torch.cuda.memory_allocated()/1e9:.2f} GB")
else:
    if a.fusion: fuse(model, elem=a.elem)
    nwc_b, bf16_b = convert(model, elem=a.elem)
    torch.cuda.synchronize()
    print(f"converted in {time.time()-t0:.0f} s; VRAM in use: {torch.cuda.memory_allocated()/1e9:.2f} GB "
          f"(uncompressed the model would take {sum(p.numel() for p in model.parameters())*2/1e9 + bf16_b/1e9:.2f} GB)")

ids = ids.cuda()
with torch.no_grad(): lg = model(ids).logits[0, -1].float()
if ref_logits is not None:
    print(f"logits native (CPU) vs now: max |d| = {(lg-ref_logits.cuda()).abs().max().item():.3e}, "
          f"argmax identical: {lg.argmax().item() == ref_logits.argmax().item()}")
if a.logits_file:
    if os.path.exists(a.logits_file):
        old = torch.load(a.logits_file).cuda()
        print(f"logits vs {a.logits_file}: max |d| = {(lg-old).abs().max().item():.3e}, "
              f"mean |d| = {(lg-old).abs().mean().item():.3e}, argmax identical: {lg.argmax().item() == old.argmax().item()}")
    else:
        torch.save(lg.cpu(), a.logits_file); print(f"logits saved: {a.logits_file}")

# generation: warm-up, then timing
with torch.no_grad():
    model.generate(ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize(); t0 = time.time()
    out = model.generate(ids, max_new_tokens=a.tokens, do_sample=False)
    torch.cuda.synchronize(); dt = time.time() - t0
new = out.shape[1] - ids.shape[1]
print(f"\ngeneration: {new} tokens in {dt:.2f} s = {new/dt:.1f} tokens/s (wall clock, HF eager)")

# pure GPU time per token (profiler): the part that depends on the kernels
from torch.profiler import profile, ProfilerActivity
with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
    model.generate(ids, max_new_tokens=32, do_sample=False)
gpu_us = sum(e.self_device_time_total for e in prof.key_averages())
print(f"GPU time per token: {gpu_us/32/1000:.2f} ms  (-> {32*1000/(gpu_us/1000):.1f} tokens/s if the GPU were the only bottleneck)")
print("text:", tok.decode(out[0], skip_special_tokens=True)[:300].replace("\n", " "))
print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# perplexity on WikiText-2 (prefill path)
try:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt").input_ids
    L, stride = a.window, a.window // 2
    nll, n_tok = 0.0, 0
    with torch.no_grad():
        for i, start in enumerate(range(0, enc.shape[1] - L, stride)):
            if i >= a.ppl_windows: break
            x = enc[:, start:start + L].cuda()
            out = model(x, labels=x)
            nll += out.loss.item() * (L - 1); n_tok += L - 1
    print(f"perplexity WikiText-2 ({a.ppl_windows} windows x {L}): {torch.exp(torch.tensor(nll/n_tok)).item():.4f}")
except Exception as e:
    print("perplexity skipped:", e)
