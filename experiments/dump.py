import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sys
from nwc.gguf import lies_gguf
from nwc.nwc import komprimiere, dekomprimiere
P = "models/mmproj-Bonsai-27B-BF16.gguf"
f, ver, meta, tensoren, start = lies_gguf(P)
t = next(x for x in tensoren if x["name"]=="mm.2.weight")
nb = 16*1024*1024
f.seek(start+t["off"]); d = f.read(nb)
paket = komprimiere(d, t["dims"])
assert dekomprimiere(paket) == d, "Python-Roundtrip kaputt"
open("test.nwc","wb").write(paket)
open("test.raw","wb").write(d)
print(f"test.raw {len(d)/1e6:.2f} MB -> test.nwc {len(paket)/1e6:.2f} MB (Rate {len(paket)/len(d):.3f})")
