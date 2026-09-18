import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import random
from collections import Counter
from nwc.rans import normiere, tabellen, kodiere, dekodiere
import math

random.seed(0)
faelle = {
    "gleichverteilt":  bytes(random.randrange(256) for _ in range(50000)),
    "schief (wie Exponent)": bytes(random.choice([60]*50+[61]*30+[62]*15+[59]*4+[63]) for _ in range(50000)),
    "ein Symbol":      bytes([7])*10000,
    "zwei Symbole":    bytes(random.choice([3,200]) for _ in range(10000)),
    "leer":            b"",
    "ein Byte":        b"\x42",
}
for name, d in faelle.items():
    z = [0]*256
    for b in d: z[b]+=1
    freq = normiere(z); kum, s2s = tabellen(freq)
    for ns in (1, 4, 16):
        p = kodiere(d, freq, kum, ns)
        zurueck = dekodiere(p, len(d), freq, kum, s2s, ns)
        ok = zurueck == d
        if not ok:
            print(f"  KAPUTT: {name} n_streams={ns}"); break
    else:
        n = len(d)
        H = 0.0
        if n:
            c = Counter(d); H = -sum((v/n)*math.log2(v/n) for v in c.values())
        p16 = kodiere(d, freq, kum, 16)
        ideal = H*n/8 + 16*4
        print(f"  {name:24s} n={n:6d}  rANS={len(p16):6d} B  Entropie+Zustaende={ideal:8.0f} B  "
              f"Overhead={100*(len(p16)-ideal)/max(ideal,1):+5.1f}%  roundtrip OK")
