import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import math
from collections import Counter, defaultdict
from nwc.gguf import lies_gguf

P = "models/mmproj-Bonsai-27B-BF16.gguf"
f, ver, meta, tensoren, start = lies_gguf(P)
t = next(x for x in tensoren if x["name"]=="mm.2.weight")
f.seek(start + t["off"]); d = f.read(8*1024*1024)
high = d[1::2]

def H0(b):
    c = Counter(b); n=len(b)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

def Hn(b, ordnung):
    """bedingte Entropie H(X | vorherige 'ordnung' Bytes), plus Tabellenkosten"""
    tab = defaultdict(Counter)
    for i in range(ordnung, len(b)):
        tab[bytes(b[i-ordnung:i])][b[i]] += 1
    n = len(b) - ordnung
    bits = 0.0
    for ctx, c in tab.items():
        m = sum(c.values())
        bits += -sum(v*math.log2(v/m) for v in c.values())
    kosten = sum(len(c) for c in tab.values()) * 2 * 8   # ~2 Byte pro Tabelleneintrag
    return bits/n, len(tab), kosten/8

print(f"high-Ebene, {len(high)/1e6:.1f} MB\n")
print(f"{'Modell':12s} {'Bits/Byte':>10s} {'Kontexte':>10s} {'Tabelle':>12s} {'Rate gesamt':>12s}")
h0 = H0(high)
print(f"{'Order-0':12s} {h0:10.3f} {1:10d} {512:10d} B {(h0/8/2 + 0.5):12.3f}")
for o in (1,2,3):
    h, nctx, kost = Hn(high, o)
    # Rate auf die gesamte bf16-Datei bezogen: high komprimiert + low roh
    rate = (h/8)/2 + 0.5 + kost/len(d)
    print(f"{'Order-'+str(o):12s} {h:10.3f} {nctx:10d} {kost/1e6:10.2f} MB {rate:12.3f}")
