import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nwc.gguf import lies_gguf
from nwc.nwc import komprimiere, dekomprimiere
P = "models/mmproj-Bonsai-27B-BF16.gguf"
f, ver, meta, tensoren, start = lies_gguf(P)
t = next(x for x in tensoren if x["name"]=="mm.2.weight")
f.seek(start+t["off"]); d = f.read(40*1024*1024)
paket = komprimiere(d, t["dims"])
assert dekomprimiere(paket) == d
open("gross.nwc","wb").write(paket); open("gross.raw","wb").write(d)
print(f"gross.raw {len(d)/1e6:.1f} MB -> {len(paket)/1e6:.1f} MB (Rate {len(paket)/len(d):.3f})")
