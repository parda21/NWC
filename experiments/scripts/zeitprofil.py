"""GPU-Zeit (CUDA-Events) vs. Python-sichtbare Zeit fuer die drei Pfade auf einer kleinen Matrix."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import time, torch
from nwc.nwc_torch import NWCWeight, NWCLinear
M, K = 8192, 4608
raw = open(os.path.join(ROOT, "data", "W.raw"), "rb").read(M * K * 2)
w = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(M, K).view(torch.bfloat16)
nw = NWCWeight(w); wc = w.cuda()
x = (torch.rand(K) - 0.5).cuda(); xb = x.to(torch.bfloat16); bias = torch.zeros(M, dtype=torch.bfloat16).cuda()
lin = torch.nn.Linear(K, M, bias=True, dtype=torch.bfloat16); lin.weight.data = w.clone(); nl = NWCLinear(lin)
xb1 = xb.view(1, 1, K)
def gpu_ms(f, n=100):
    for _ in range(10): f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n
def cpu_ms(f, n=100):
    for _ in range(10): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3
for name, f in [("torch F.linear BF16", lambda: torch.nn.functional.linear(xb1, wc, bias)),
                ("NWC matvec fp32-Pfad", lambda: nw.matvec(x)),
                ("NWC linear_bf16 (DLL)", lambda: nw.linear_bf16(xb, bias)),
                ("NWCLinear.forward", lambda: nl(xb1))]:
    print(f"{name:26s} GPU-Zeit {gpu_ms(f):.3f} ms   Python-Zeit {cpu_ms(f):.3f} ms")
