"""Kernel benchmark with real model shapes (Qwen3-4B, fused): NWC token path vs cuBLAS per layer.
GPU time by CUDA events, median. NWC_DLL selects the library (A/B tests).
usage: python scripts/kernbench.py [--runs 30] [--only gate_up,lm_head] [--all]   (needs data/W.raw)"""
import os, sys, statistics, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from nwc.nwc_torch import NWCWeight, choose_block, X_FP32

ap = argparse.ArgumentParser()
ap.add_argument("--ncu", action="store_true", help="few runs only, for the profiler")
ap.add_argument("--block", type=int, default=0)
ap.add_argument("--runs", type=int, default=100)
ap.add_argument("--only", default="", help="comma list of layer names")
ap.add_argument("--warm", default="", help="data|low|both: load into the L2 before every NWC run (diagnosis)")
ap.add_argument("--all", action="store_true", help="also time dequant and the fp32 matvec")
ap.add_argument("--copies-mb", type=int, default=160, help="working set per shape (L2 of the 4070 = 36 MB)")
a = ap.parse_args()

# (name, M = out, K = in): Qwen3-4B with fused qkv / gate+up, plus lm_head
SHAPES = [("qkv", 6144, 2560), ("o", 2560, 4096), ("gate_up", 19456, 2560), ("down", 2560, 9728), ("lm_head", 151936, 2560)]
raw = open(os.path.join(ROOT, "data", "W.raw"), "rb").read()

def gpu_ms(f, n, before=None):
    for _ in range(3 if a.ncu else 10): f()
    torch.cuda.synchronize()
    z = []
    for _ in range(n):
        if before: before(); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.cuda.synchronize(); z.append(s.elapsed_time(e))
    return statistics.median(z)

n_runs = 2 if a.ncu else a.runs
print(f"{'layer':8s} {'M':>7s} {'K':>6s} {'MB':>6s} {'block':>5s} | {'cuBLAS ms':>9s} {'GB/s':>5s} | {'NWC ms':>8s} {'read':>7s} {'equiv':>6s} | speedup")
sum_c = sum_n = 0.0
for name, M, K in SHAPES:
    if a.only and name not in a.only.split(","): continue
    n = M * K
    w = torch.frombuffer(bytearray(raw[:n * 2]), dtype=torch.int16).view(M, K).view(torch.bfloat16)
    block = a.block or choose_block(n)
    nk = max(1, min(8, a.copies_mb * 1000000 // (2 * n)))
    nws = [NWCWeight(w, block=block) for _ in range(nk)]; nw = nws[0]
    wcs = [w.cuda() for _ in range(nk)]; wc = wcs[0]
    xb = (torch.rand(K) - 0.5).cuda().to(torch.bfloat16); bias = torch.zeros(M, dtype=torch.bfloat16).cuda()
    ref = torch.nn.functional.linear(xb, wc, bias)
    out = nw.linear_bf16(xb, bias)
    it = [0]
    def fc(): it[0] += 1; return torch.nn.functional.linear(xb, wcs[it[0] % nk], bias)
    xn = xb.float() if X_FP32 else xb            # v8+: convert x beforehand (otherwise the event times the CPU dispatch of the cast)
    def fn(): it[0] += 1; return nws[it[0] % nk].linear_bf16(xn, bias)
    tc = gpu_ms(fc, n_runs)
    def warm():
        w_ = nws[(it[0] + 1) % nk]
        if a.warm in ("data", "both"): w_.data.sum()
        if a.warm in ("low", "both"): w_.low.sum()
    tn = gpu_ms(fn, n_runs, warm if a.warm else None)
    if a.all:
        x32 = xb.float()
        def fd(): it[0] += 1; return nws[it[0] % nk].dequant()
        def f32(): it[0] += 1; return nws[it[0] % nk].matvec(x32)
        td = gpu_ms(fd, n_runs); t32 = gpu_ms(f32, n_runs)
        print(f"   {name}: dequant {td:.3f} ms ({2*n/1e6/td:.0f} GB/s written, {nw.bytes/1e6/td:.0f} read) | fp32 matvec {t32:.3f} ms ({nw.bytes/1e6/t32:.0f} GB/s read)")
    mb_n, mb_c = 2 * n / 1e6, nw.bytes / 1e6
    sum_c += tc; sum_n += tn
    print(f"{name:8s} {M:7d} {K:6d} {mb_n:6.0f}x{nk} {block:5d} | {tc:9.3f} {mb_n/tc:5.0f} | {tn:8.3f} {mb_c/tn:7.0f} {mb_n/tn:6.0f} | {tc/tn:5.3f}x  max|dy|={(out.float()-ref.float()).abs().max().item():.2e}")
    del nw, wc, nws, wcs; torch.cuda.empty_cache()
print(f"sum per token (36 layers x 4 + lm_head): cuBLAS {36*(sum_c-tc)+tc:.2f} ms, NWC {36*(sum_n-tn)+tn:.2f} ms")
