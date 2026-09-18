import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import time
from nwc.gguf import lies_gguf
from nwc import nwc as nwc, nwc_gpu
P = "models/mmproj-Bonsai-27B-BF16.gguf"
f, ver, meta, tensoren, start = lies_gguf(P)
t = next(x for x in tensoren if x["name"]=="mm.2.weight")
f.seek(start+t["off"]); d = f.read(40*1024*1024)

p1 = nwc.komprimiere(d, t["dims"])
print(f"{'Variante':30s} {'Rate':>7s} {'Threads':>9s} {'Roundtrip':>10s}")
print(f"{'NWC1 (16 verschr. Stroeme)':30s} {len(p1)/len(d):7.3f} {(len(p1) and 20*16):9d} {'OK':>10s}")
for lanes, block in ((32, 8192), (32, 32768), (32, 131072), (64, 131072)):
    p = nwc_gpu.komprimiere(d, t["dims"], lanes, block)
    ok = nwc_gpu.dekomprimiere(p) == d
    nb = (len(d)//2 + block - 1)//block
    print(f"{'NWC2 lanes=%d block=%d' % (lanes, block):30s} {len(p)/len(d):7.3f} {nb*lanes:9d} "
          f"{'OK' if ok else 'KAPUTT':>10s}")
    if ok: open(f"gpu_{lanes}_{block}.nwc","wb").write(p)
open("gpu.raw","wb").write(d)
