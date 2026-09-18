import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import zlib, math
from collections import Counter
from nwc.gguf import lies_gguf

P = "models/mmproj-Bonsai-27B-BF16.gguf"
f, ver, meta, tensoren, start = lies_gguf(P)

def entropie(b):
    c = Counter(b); n = len(b)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

print(f"{'Tensor':26s} {'MB':>6s} {'H_roh':>6s} {'H_high':>7s} {'H_low':>6s} "
      f"{'zlib_roh':>9s} {'zlib_eb':>8s} {'Grenze':>7s}")
print("-"*82)

proben = [t for t in tensoren if t["typ"]=="BF16"]
auswahl = [("mm.2.weight",None),("mm.0.weight",None),("v.blk.0.ffn_up.weight",None),
           ("v.blk.0.attn_q.weight",None),("v.blk.10.ffn_down.weight",None)]
GRENZE = 8*1024*1024   # 8 MB pro Tensor sampeln

gesamt_roh = gesamt_eb = gesamt_grenze = gesamt_n = 0
for name,_ in auswahl:
    t = next((x for x in proben if x["name"]==name), None)
    if t is None: continue
    n_bytes = min(t["n"]*2, GRENZE)
    n_bytes -= n_bytes % 2
    f.seek(start + t["off"]); d = f.read(n_bytes)
    low, high = d[0::2], d[1::2]
    zr  = len(zlib.compress(d, 6))
    zeb = len(zlib.compress(high, 6)) + len(zlib.compress(low, 6))
    gr  = int((entropie(high)+entropie(low))/2/8 * len(d))
    print(f"{name[:26]:26s} {len(d)/1e6:6.1f} {entropie(d):6.3f} {entropie(high):7.3f} "
          f"{entropie(low):6.3f} {zr/len(d):9.3f} {zeb/len(d):8.3f} {gr/len(d):7.3f}")
    gesamt_roh += zr; gesamt_eb += zeb; gesamt_grenze += gr; gesamt_n += len(d)

print("-"*82)
print(f"{'GESAMT':26s} {gesamt_n/1e6:6.1f} {'':6s} {'':7s} {'':6s} "
      f"{gesamt_roh/gesamt_n:9.3f} {gesamt_eb/gesamt_n:8.3f} {gesamt_grenze/gesamt_n:7.3f}")
