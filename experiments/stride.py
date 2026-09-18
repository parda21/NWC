import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import math
from collections import Counter, defaultdict
from nwc.gguf import lies_gguf
P = "models/mmproj-Bonsai-27B-BF16.gguf"
f, ver, meta, tensoren, start = lies_gguf(P)

def H_bedingt(b, s):
    tab = defaultdict(Counter)
    for i in range(s, len(b)): tab[b[i-s]][b[i]] += 1
    n = len(b)-s; bits = 0.0
    for _, c in tab.items():
        m = sum(c.values()); bits += -sum(v*math.log2(v/m) for v in c.values())
    return bits/n

for name in ("mm.2.weight","v.blk.0.ffn_up.weight"):
    t = next(x for x in tensoren if x["name"]==name)
    f.seek(start+t["off"]); d = f.read(6*1024*1024)
    high = d[1::2]
    c = Counter(high); n=len(high)
    h0 = -sum((v/n)*math.log2(v/n) for v in c.values())
    print(f"\n{name}  dims={t['dims']}  (Zeilenlaenge ne0={t['dims'][0]})")
    print(f"  Order-0: {h0:.4f} Bits")
    kand = [1,2,3,4,8,16,64,256,1024, t['dims'][0]//2, t['dims'][0]-1, t['dims'][0], t['dims'][0]+1, t['dims'][0]*2]
    for s in sorted(set(x for x in kand if 0 < x < len(high)//4)):
        h = H_bedingt(high, s)
        marke = "  <== Zeilenlaenge" if s == t['dims'][0] else ""
        print(f"  stride {s:6d}: {h:.4f}  ({(h0-h)/h0*100:+5.2f} %){marke}")
