import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import zlib, math
from collections import Counter
from nwc.gguf import lies_gguf
P = "models/mmproj-Bonsai-27B-BF16.gguf"
f, ver, meta, tensoren, start = lies_gguf(P)
for name in ("mm.2.weight","v.blk.0.ffn_up.weight"):
    t = next(x for x in tensoren if x["name"]==name)
    f.seek(start+t["off"]); d = f.read(8*1024*1024)
    low, high = d[0::2], d[1::2]
    print(f"\n=== {name} ===")
    for nm, e in (("high",high),("low",low)):
        z = len(zlib.compress(e,6))
        c = Counter(e)
        H = -sum((v/len(e))*math.log2(v/len(e)) for v in c.values())
        laeufe = sum(1 for i in range(1,len(e)) if e[i]==e[i-1])
        print(f"  {nm}: {len(e)} B -> zlib {z} ({z/len(e):.3f}), H0={H:.3f} "
              f"({H/8:.3f}), distinkte Werte={len(c)}, Wiederholung Nachbar={laeufe/len(e)*100:.1f}%")
    # Nullgewichte?
    nullen = sum(1 for i in range(0,len(d),2) if d[i]==0 and d[i+1]==0)
    print(f"  exakte Nullen: {nullen/(len(d)//2)*100:.2f}%")
