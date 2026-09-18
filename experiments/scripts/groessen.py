"""Groessenkurve: NWC-Token-Pfad vs cuBLAS (F.linear) fuer Matrizen von 4 MB bis 900 MB, K = 3584 wie Qwen-7B.
GPU-Zeit per CUDA-Events, Median ueber viele Laeufe. nutzung: python scripts/groessen.py [block]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch, statistics
from nwc.nwc_torch import NWCWeight, waehle_block

block_fest = int(sys.argv[1]) if len(sys.argv) > 1 else None
K = 3584
raw = open(os.path.join(ROOT, "data", "W.raw"), "rb").read()
n_max = len(raw) // 2

def gpu_ms(f, n=60):
    for _ in range(8): f()
    torch.cuda.synchronize()
    zeiten = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.cuda.synchronize()
        zeiten.append(s.elapsed_time(e))
    return statistics.median(zeiten)

print(f"K={K}; Groesse | Block | Threads | cuBLAS ms  GB/s | NWC ms  GB/s(gelesen) GB/s(aequiv) | Speedup")
for M in (512, 3584, 7168, 18944, 37888, 75776, 120000):
    n = M * K
    if n > n_max: break
    w = torch.frombuffer(bytearray(raw[:n * 2]), dtype=torch.int16).view(M, K).view(torch.bfloat16)
    block = block_fest or waehle_block(n)
    nw = NWCWeight(w, block=block)
    wc = w.cuda(); xb = (torch.rand(K) - 0.5).cuda().to(torch.bfloat16); bias = torch.zeros(M, dtype=torch.bfloat16).cuda()
    tc = gpu_ms(lambda: torch.nn.functional.linear(xb, wc, bias))
    tn = gpu_ms(lambda: nw.linear_bf16(xb, bias))
    x32 = xb.float(); t32 = gpu_ms(lambda: nw.matvec(x32))
    mb_n, mb_c = 2 * n / 1e6, nw.bytes / 1e6
    print(f"{mb_n:7.0f} MB | {block:5d} | {n // block * 32:7d} | {tc:8.3f} {mb_n/tc:6.0f} | {tn:7.3f} {mb_c/tn:6.0f} {mb_n/tn:6.0f} | {tc/tn:5.3f}x | fp32-Kernel {t32:7.3f} ms {mb_n/t32:4.0f} GB/s {tc/t32:5.3f}x")
    del nw, wc; torch.cuda.empty_cache()
