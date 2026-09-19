"""fp8 element type: the fp8 values come back bit-exact (dequant, dequant_fp8, gather), the token path matches a
float reference, the library's reference fp8 matvec agrees, NaN is rejected. Odd shapes included.
usage: python tests/test_fp8.py   (GPU)"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from nwc.nwc_torch import NWCWeight, quantize_fp8, ref_fp8_matvec, ELEM

assert ELEM, "library without fp8 support"
torch.manual_seed(0)
ok_all = True


def check(M, K, seed=0):
    global ok_all
    torch.manual_seed(seed)
    w = torch.randn(M, K) * 0.02
    w.view(-1)[:: 97] *= 40                                   # outliers: per-channel absmax far above sigma -> many subnormals
    w = w.to(torch.bfloat16)
    q, scale = quantize_fp8(w)
    ref_vals = q.to(torch.bfloat16)                            # unscaled fp8 values, exact in BF16
    c = NWCWeight(w, elem="fp8")
    back = c.dequant().cpu()
    q2, s2 = c.dequant_fp8()
    x = torch.randn(K)
    y = c.linear_bf16(x.cuda().float(), None).cpu().float()
    ref = (q.float() * scale[:, None]) @ x
    tol = 2 ** -7 * ref.abs().max().item() + 1e-6                # BF16 output rounding + fp32 summation order
    y_ref_kernel = ref_fp8_matvec(q.cuda().contiguous(), scale.cuda(), x.cuda()).cpu().float()
    ids = torch.randint(0, M, (7,))
    rows = c.gather(ids.cuda()).cpu()
    sub = ((q.view(torch.uint8) >> 3) & 15 == 0).float().mean().item()
    r = {"dequant": torch.equal(back, ref_vals), "fp8 bytes": torch.equal(q2.cpu().view(torch.uint8), q.view(torch.uint8)),
         "scale": torch.equal(s2.cpu(), scale), "gather": torch.equal(rows, ref_vals[ids]),
         "matvec": (y - ref).abs().max().item() <= tol, "ref kernel": (y_ref_kernel - ref).abs().max().item() <= tol}
    ok = all(r.values()); ok_all &= ok
    print(f"{M:>5} x {K:<5} bits/weight {8*c.bytes/c.n:5.2f} (fp8 8.0, {100*sub:.1f}% e=0)  " +
          "  ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r.items()) +
          f"  max|dy| {(y-ref).abs().max().item():.2e} / tol {tol:.2e}")


for M, K in [(8, 512), (13, 40), (3, 4097), (1001, 1000), (2560, 4096), (6144, 2560)]:
    check(M, K, seed=M)
try:
    bad = torch.full((8, 512), float("nan")).to(torch.float8_e4m3fn)
    NWCWeight(bad, elem="fp8", scale=torch.ones(8)); print("NaN accepted: FAIL"); ok_all = False
except ValueError as e:
    print(f"NaN rejected: ok ({e})")
print("OK" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
