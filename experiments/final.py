import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import zlib, lzma, time, hashlib, struct
from nwc.gguf import lies_gguf
from nwc.nwc import komprimiere, dekomprimiere

P = "models/mmproj-Bonsai-27B-BF16.gguf"
f, ver, meta, tensoren, start = lies_gguf(P)
bf = [t for t in tensoren if t["typ"]=="BF16"]

# 20 Tensoren quer durchs Modell, je bis 3 MB
import random; random.seed(3)
auswahl = sorted(random.sample(bf, min(20, len(bf))), key=lambda t: -t["n"])
ges_n = ges_nwc = ges_z = ges_xz = 0
t_dek = 0.0
fehler = 0
for t in auswahl:
    nb = min(t["n"]*2, 3*1024*1024); nb -= nb % 2
    if nb < 4096: continue
    f.seek(start+t["off"]); d = f.read(nb)
    paket = komprimiere(d, t["dims"])
    t0 = time.perf_counter(); zurueck = dekomprimiere(paket); t_dek += time.perf_counter()-t0
    if hashlib.sha256(zurueck).digest() != hashlib.sha256(d).digest():
        fehler += 1; print("  KAPUTT:", t["name"])
    ges_n += len(d); ges_nwc += len(paket)
    ges_z += len(zlib.compress(d,6)); ges_xz += len(lzma.compress(d, preset=6))

print(f"{len(auswahl)} Tensoren, {ges_n/1e6:.1f} MB echte BF16-Gewichte")
print(f"  bitexakter Roundtrip : {'ALLE OK' if fehler==0 else str(fehler)+' FEHLER'}")
print()
print(f"  {'unkomprimiert':22s} {ges_n/1e6:8.2f} MB   1.000")
print(f"  {'NWC1 (dieser Codec)':22s} {ges_nwc/1e6:8.2f} MB   {ges_nwc/ges_n:.3f}   = {ges_n/ges_nwc:.2f}x")
print(f"  {'zlib -6':22s} {ges_z/1e6:8.2f} MB   {ges_z/ges_n:.3f}   = {ges_n/ges_z:.2f}x")
print(f"  {'xz -6':22s} {ges_xz/1e6:8.2f} MB   {ges_xz/ges_n:.3f}   = {ges_n/ges_xz:.2f}x")
print()
print(f"  Dekodierdurchsatz (Python, 1 Kern): {ges_n/t_dek/1e6:.1f} MB/s")
