"""Decode step as a CUDA graph: StaticCache, one forward is captured and replayed per token.
Measures the real wall clock without Python overhead, native vs NWC, and checks the token sequence against HF generate (greedy).
usage: python scripts/graph_decode.py --mode native|nwc [--fusion] [--tokens 256] [--model PATH]"""
import os, sys, time, argparse, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["native", "nwc"])
ap.add_argument("--model", default=f"{ROOT}/models/Qwen3-4B")
ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--fusion", action="store_true")
ap.add_argument("--check", type=int, default=64, help="compare this many tokens against HF generate (0 = off)")
a = ap.parse_args()

prompt = "Question: What is a binary tree and its applications? Answer:"
tok = AutoTokenizer.from_pretrained(a.model)
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
L = ids.shape[1]

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
if a.mode == "nwc":
    from nwc.nwc_torch import convert, fuse
    if a.fusion: fuse(model)
    convert(model)
else:
    model.cuda()
model.eval()
print(f"[{a.mode}] VRAM in use {torch.cuda.memory_allocated()/1e9:.2f} GB")

ref = None
if a.check:
    with torch.no_grad():
        ref = model.generate(ids, max_new_tokens=a.check, do_sample=False)[0, L:].clone()

max_len = L + a.tokens + 8
cache = StaticCache(config=model.config, max_cache_len=max_len)
with torch.no_grad():
    out = model(input_ids=ids, past_key_values=cache, cache_position=torch.arange(L, device="cuda"), use_cache=True)
cur = out.logits[:, -1].argmax(-1).view(1, 1).clone()          # static buffers
pos = torch.tensor([L], device="cuda")
res = torch.zeros(a.tokens + 1, dtype=torch.long, device="cuda")
res[0] = cur[0, 0]

def step():
    o = model(input_ids=cur, past_key_values=cache, cache_position=pos, use_cache=True)
    nxt = o.logits[:, -1].argmax(-1)
    pos.add_(1)
    res.index_copy_(0, pos - L, nxt)
    cur.copy_(nxt.view(1, 1))

# warm-up on a side stream (required before capture), then capture
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s), torch.no_grad():
    for _ in range(3): step()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g), torch.no_grad():
    step()
# after warm-up + capture pos has advanced and the cache is partly written: restart cleanly for the timing
with torch.no_grad():
    cache.reset()
    out = model(input_ids=ids, past_key_values=cache, cache_position=torch.arange(L, device="cuda"), use_cache=True)
cur.copy_(out.logits[:, -1].argmax(-1).view(1, 1)); pos.fill_(L); res.zero_(); res[0] = cur[0, 0]

torch.cuda.synchronize(); t0 = time.time()
for _ in range(a.tokens): g.replay()
torch.cuda.synchronize(); dt = time.time() - t0
print(f"CUDA graph: {a.tokens} tokens in {dt:.3f} s = {a.tokens/dt:.1f} tokens/s = {dt/a.tokens*1e3:.2f} ms/token (wall clock)")
if ref is not None:
    same = int((res[:a.check] == ref[:a.check]).sum())
    print(f"token sequence vs HF generate (greedy, {a.check} tokens): {same}/{a.check} identical")
print("text:", tok.decode(res[:a.tokens], skip_special_tokens=True)[:300].replace("\n", " "))
