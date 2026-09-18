"""Self-test of the PyTorch bridge on real weights (W.raw): dequantization bit-exact, matvec correct, speed vs cuBLAS."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import time, torch
from nwc.nwc_torch import NWCWeight, NWCLinear

M, K = 8192, 4608
raw = open(os.path.join(ROOT, "data", "W.raw"), "rb").read(M * K * 2)
w = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(M, K).view(torch.bfloat16)

nw = NWCWeight(w)
print(f"W {M}x{K}: BF16 {nw.bytes_bf16/1e6:.1f} MB -> NWC {nw.bytes/1e6:.1f} MB (ratio {nw.bytes/nw.bytes_bf16:.4f})")

# 1. dequantization bit-exact
wd = nw.dequant().cpu()
print("dequantization bit-exact:", torch.equal(wd.view(torch.int16), w.view(torch.int16)))

# 2. matvec against an fp32 reference
torch.manual_seed(0)
x = (torch.rand(K) - 0.5).cuda()
y = nw.matvec(x)
ref = w.cuda().float() @ x
err = ((y - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
print(f"matvec max relative error: {err:.2e}")

# 3. NWCLinear, both paths
lin = torch.nn.Linear(K, M, bias=True, dtype=torch.bfloat16); lin.weight.data = w.clone()
nl = NWCLinear(lin)
xb = x.to(torch.bfloat16)
y1 = nl(xb.view(1, 1, K)); r1 = torch.nn.functional.linear(xb, w.cuda(), lin.bias.cuda())
y2 = nl(xb.view(1, 1, K).repeat(1, 4, 1)); r2 = torch.nn.functional.linear(xb.repeat(4, 1), w.cuda(), lin.bias.cuda())
print(f"NWCLinear token path   max |dy|: {(y1.reshape(-1).float()-r1.float()).abs().max().item():.3e}")
print(f"NWCLinear prefill path max |dy|: {(y2.reshape(4,-1).float()-r2.float()).abs().max().item():.3e}")

# 4. speed: NWC vs torch (cuBLAS) on BF16
wc = w.cuda(); xb = x.to(torch.bfloat16)
for _ in range(5): nw.matvec(x); torch.matmul(wc, xb)
torch.cuda.synchronize()
def timeit(f, n=50):
    t0 = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3
tn, tt = timeit(lambda: nw.matvec(x)), timeit(lambda: torch.matmul(wc, xb))
print(f"matvec {M}x{K}: NWC (fp32 path) {tn:.3f} ms, torch/cuBLAS BF16 {tt:.3f} ms -> {tt/tn:.3f}x")
bias = lin.bias.data.cuda(); xb1 = xb.view(1, 1, K)
tl = timeit(lambda: nl(xb1)); tf = timeit(lambda: torch.nn.functional.linear(xb1, wc, bias))
print(f"linear {M}x{K} (BF16 in/out + bias): NWCLinear {tl:.3f} ms, torch F.linear {tf:.3f} ms -> {tf/tl:.3f}x")
yl = nl(xb1).reshape(-1).float(); yf = torch.nn.functional.linear(xb1, wc, bias).reshape(-1).float()
print(f"  max |dy| NWCLinear vs F.linear: {(yl-yf).abs().max().item():.3e}  (BF16 output)")
