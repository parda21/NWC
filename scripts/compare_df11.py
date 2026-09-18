"""Three-way comparison: native BF16 vs DFloat11 vs NWC -- same model, same GPU, same measurement.
usage: python scripts/compare_df11.py --mode native|df11|nwc [--model PATH] [--df11 PATH] [--tokens N] [--fusion]
Measures: VRAM, tokens/s (wall clock), GPU time per token (profiler, top kernels), logits of the prompt (saved/compared)."""
import os, sys, time, argparse, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["native", "df11", "nwc"])
ap.add_argument("--model", default=f"{ROOT}/models/Qwen3-4B")
ap.add_argument("--df11", default=f"{ROOT}/models/Qwen3-4B-DF11")
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--fusion", action="store_true")
a = ap.parse_args()

prompt = "Question: What is a binary tree and its applications? Answer:"
tok = AutoTokenizer.from_pretrained(a.model if os.path.exists(os.path.join(a.model, "tokenizer.json")) else a.df11)
ids = tok(prompt, return_tensors="pt").input_ids

t0 = time.time()
if a.mode == "df11":
    import transformers.modeling_utils as _mu                        # shim: dfloat11 0.5 vs transformers 5.x
    if not hasattr(_mu, 'no_init_weights'):
        from transformers.initialization import no_init_weights as _niw; _mu.no_init_weights = _niw
    from dfloat11 import DFloat11Model
    model = DFloat11Model.from_pretrained(a.df11, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    if a.mode == "nwc":
        from nwc.nwc_torch import convert, fuse
        if a.fusion: fuse(model)
        convert(model)
    else:
        model.cuda()
model.eval(); torch.cuda.synchronize()
print(f"[{a.mode}] loaded in {time.time()-t0:.0f} s; VRAM in use {torch.cuda.memory_allocated()/1e9:.2f} GB, "
      f"reserved {torch.cuda.memory_reserved()/1e9:.2f} GB")

ids = ids.cuda()
with torch.no_grad(): lg = model(ids).logits[0, -1].float()
pf = os.path.join(ROOT, "data", "logits_qwen3_4b_native.pt")
if a.mode == "native":
    torch.save(lg.cpu(), pf); print("logits (native) saved")
elif os.path.exists(pf):
    ref = torch.load(pf).cuda()
    print(f"logits vs native: max |d| = {(lg-ref).abs().max().item():.3e}, argmax identical: {lg.argmax().item()==ref.argmax().item()}")

with torch.no_grad():
    model.generate(ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize(); t0 = time.time()
    out = model.generate(ids, max_new_tokens=a.tokens, do_sample=False)
    torch.cuda.synchronize(); dt = time.time() - t0
new = out.shape[1] - ids.shape[1]
print(f"generation: {new} tokens in {dt:.2f} s = {new/dt:.1f} tokens/s (wall clock)")
print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

N = 32
with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
    model.generate(ids, max_new_tokens=N, do_sample=False)
ev = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)
total = sum(e.self_device_time_total for e in ev)
print(f"GPU time per token: {total/N/1000:.2f} ms")
for e in ev[:6]:
    print(f"   {e.key[:60]:60s} {e.self_device_time_total/N/1000:7.3f} ms  {e.count/N:6.1f}/token")
print("text:", tok.decode(out[0], skip_special_tokens=True)[:200].replace("\n", " "))
