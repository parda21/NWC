"""Testdaten im NWC2-Format mit 16-Bit-Renormalisierung, ohne Stride."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sys, time
from nwc.gguf import lies_gguf
from nwc import nwc_gpu

P = "models/mmproj-Bonsai-27B-BF16.gguf"
f, v, m, t, s = lies_gguf(P)
x = next(k for k in t if k["name"] == "mm.2.weight")
f.seek(s + x["off"]); d = f.read(40 * 1024 * 1024)

for block in (8192, 32768):
    t0 = time.time()
    p = nwc_gpu.komprimiere(d, x["dims"], 32, block, stride_fest=0, renorm=16)
    assert nwc_gpu.dekomprimiere(p) == d, "Python-Roundtrip kaputt"
    name = f"v2_{block}.nwc"
    open(name, "wb").write(p)
    print(f"{name}  Rate {len(p)/len(d):.4f}  ({time.time()-t0:.0f}s)", flush=True)
