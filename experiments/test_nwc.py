import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import zlib, time, struct
from nwc.gguf import lies_gguf
from nwc.nwc import komprimiere, dekomprimiere

P = "models/mmproj-Bonsai-27B-BF16.gguf"
f, ver, meta, tensoren, start = lies_gguf(P)

print(f"{'Tensor':26s} {'MB':>5s} {'stride':>6s} {'NWC1':>7s} {'zlib-6':>7s} {'xz-ish':>7s} {'roundtrip':>10s}")
print("-"*74)
GRENZE = 4*1024*1024
ges_n = ges_nwc = ges_z = 0
for name in ("mm.2.weight","mm.0.weight","v.blk.0.ffn_up.weight","v.blk.5.attn_k.weight","v.blk.10.ffn_down.weight"):
    t = next((x for x in tensoren if x["name"]==name), None)
    if t is None: continue
    nb = min(t["n"]*2, GRENZE); nb -= nb % 2
    f.seek(start+t["off"]); d = f.read(nb)
    t0=time.time(); paket = komprimiere(d, t["dims"]); tk=time.time()-t0
    t0=time.time(); zurueck = dekomprimiere(paket); td=time.time()-t0
    ok = zurueck == d
    s = struct.unpack("<I", paket[16:20])[0]
    import lzma
    xz = len(lzma.compress(d, preset=6))
    z  = len(zlib.compress(d, 6))
    print(f"{name[:26]:26s} {len(d)/1e6:5.1f} {s:6d} {len(paket)/len(d):7.3f} {z/len(d):7.3f} "
          f"{xz/len(d):7.3f} {'OK' if ok else 'KAPUTT':>10s}")
    ges_n += len(d); ges_nwc += len(paket); ges_z += z
print("-"*74)
print(f"{'GESAMT':26s} {ges_n/1e6:5.1f} {'':6s} {ges_nwc/ges_n:7.3f} {ges_z/ges_n:7.3f}")
print(f"\nkomprimieren {tk:.1f}s, dekomprimieren {td:.1f}s (reines Python)")
