"""Gather kernel (embedding lookup from the compressed matrix) against the original rows: bit-exact?
usage: python tests/test_gather.py   (needs data/W.raw)"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from nwc.nwc_torch import NWCWeight, fits, choose_block, V9

torch.manual_seed(0)
raw = open(os.path.join(ROOT, "data", "W.raw"), "rb").read()
ok_all = True
for M, K, block in ((4096, 2560, 2048), (8192, 4608, 8192), (151936 // 8, 2560, 2048), (2048, 2064, 512), (1001, 1000, 512), (13, 40, 512)):
    n = M * K
    if not fits(M, K): print(f"M={M:6d} K={K:5d}: shape not supported by this format, skipped"); continue
    if V9: block = choose_block(n)
    w = torch.frombuffer(bytearray(raw[:n * 2]), dtype=torch.int16).view(M, K).view(torch.bfloat16)
    nw = NWCWeight(w, block=block)
    ids = torch.cat([torch.tensor([0, M - 1, 1, M // 2]), torch.randint(0, M, (60,))]).view(4, 16)
    out = nw.gather(ids.cuda())
    ref = w.cuda()[ids.cuda()]
    ok = torch.equal(out.view(torch.int16), ref.view(torch.int16))
    ok_all &= ok
    print(f"M={M:6d} K={K:5d} block={block:4d}: gather bit-exact={ok}  ({tuple(out.shape)})")
print("ALL OK" if ok_all else "FAIL")
