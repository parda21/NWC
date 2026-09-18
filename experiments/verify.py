import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import math
from collections import Counter, defaultdict
from nwc.gguf import lies_gguf
P = "models/mmproj-Bonsai-27B-BF16.gguf"
f, ver, meta, tensoren, start = lies_gguf(P)

def H0(b):
    c=Counter(b); n=len(b); return -sum((v/n)*math.log2(v/n) for v in c.values())
def Hb(b,s):
    tab=defaultdict(Counter)
    for i in range(s,len(b)): tab[b[i-s]][b[i]]+=1
    n=len(b)-s; bits=0.0
    for _,c in tab.items():
        m=sum(c.values()); bits+=-sum(v*math.log2(v/m) for v in c.values())
    return bits/n

t = next(x for x in tensoren if x["name"]=="mm.2.weight")
print("feiner Sweep um 2304, mm.2.weight (6 MB ab Tensorstart):")
f.seek(start+t["off"]); high = f.read(6*1024*1024)[1::2]
h0 = H0(high)
for s in (2300,2302,2303,2304,2305,2306,2308,2312,4608,6912,2304*2):
    print(f"   stride {s:5d}: {Hb(high,s):.4f}  ({(h0-Hb(high,s))/h0*100:+6.2f} %)")

print("\nselber Tensor, aber 20 MB weiter hinten:")
f.seek(start+t["off"]+20*1024*1024); high2 = f.read(6*1024*1024)[1::2]
h02 = H0(high2)
for s in (1,2304,4608):
    print(f"   stride {s:5d}: {Hb(high2,s):.4f}  ({(h02-Hb(high2,s))/h02*100:+6.2f} %)")

print("\nandere Tensoren, stride = ne0/2 vs ne0:")
for nm in ("mm.0.weight","v.blk.1.ffn_up.weight","v.blk.5.attn_k.weight"):
    tt = next((x for x in tensoren if x["name"]==nm), None)
    if tt is None: continue
    f.seek(start+tt["off"]); hh = f.read(4*1024*1024)[1::2]
    if len(hh) < 4*tt["dims"][0]: continue
    a, b = tt["dims"][0]//2, tt["dims"][0]
    h=H0(hh)
    print(f"   {nm[:26]:26s} ne0={tt['dims'][0]:5d}  H0={h:.3f}  "
          f"ne0/2:{(h-Hb(hh,a))/h*100:+6.2f} %  ne0:{(h-Hb(hh,b))/h*100:+6.2f} %")
