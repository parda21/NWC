"""Token path, dequantization and fp32 path for odd shapes (K not a multiple of 512/16, M not a multiple of 8).
usage: python tests/test_k.py   (needs data/W.raw, see scripts/make_wraw.py)"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from nwc.nwc_torch import NWCWeight, fits, VERSION
raw = open(os.path.join(ROOT, "data", "W.raw"), "rb").read(96 * 1024 * 1024)
ok_all = True
for M, K in ((8192, 4608), (4096, 11008), (1024, 3600), (2048, 2064), (512, 16384), (19456, 2560), (1024, 9728),
             (1001, 1000), (13, 40), (8, 16), (3, 4097)):
    if not fits(M, K): print(f"M={M:5d} K={K:5d}: shape not supported by format v{VERSION}, skipped"); continue
    n = M * K
    w = torch.frombuffer(bytearray(raw[:n * 2]), dtype=torch.int16).view(M, K).view(torch.bfloat16)
    nw = NWCWeight(w)
    torch.manual_seed(1)
    xb = (torch.rand(K) - 0.5).cuda().to(torch.bfloat16)
    bias = (torch.rand(M) - 0.5).cuda().to(torch.bfloat16)
    y = nw.linear_bf16(xb, bias).float()
    ref = torch.nn.functional.linear(xb, w.cuda(), bias).float()
    d = (y - ref).abs().max().item()
    exact = torch.equal(nw.dequant().cpu().view(torch.int16), w.view(torch.int16))
    # fp32 path against an fp32 reference: only the summation order may differ (~1e-6), no weight error
    d32 = 0.0
    if K % 512 == 0 or VERSION >= 9:
        y32 = nw.matvec(xb.float())
        ref32 = w.cuda().float() @ xb.float()
        d32 = ((y32 - ref32).abs() / (ref32.abs() + 1e-2)).max().item()
    good = exact and d < 2e-2 and d32 < 1e-4
    ok_all &= good
    print(f"M={M:5d} K={K:5d} (K%512={K%512:3d}): dequant bit-exact={exact}  max|dy|={d:.2e}  fp32 rel {d32:.1e}  {'OK' if good else 'FAIL'}")
print("ALL OK" if ok_all else "FAIL")
