"""Alle BF16-Tensoren einer GGUF-Datei als rohen Byteblock schreiben (nur I/O)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sys
from experiments.reference.gguf import lies_gguf

P = sys.argv[1] if len(sys.argv) > 1 else \
    "models/mmproj-Bonsai-27B-BF16.gguf"
ziel = sys.argv[2] if len(sys.argv) > 2 else "big.raw"
grenze = int(sys.argv[3]) * 1024 * 1024 if len(sys.argv) > 3 else None

f, ver, meta, tensoren, start = lies_gguf(P)
bf = [t for t in tensoren if t["typ"] == "BF16"]
n = 0
with open(ziel, "wb") as out:
    for t in bf:
        f.seek(start + t["off"])
        d = f.read(t["n"] * 2)
        if grenze and n + len(d) > grenze:
            d = d[: (grenze - n) & ~1]
        out.write(d); n += len(d)
        if grenze and n >= grenze:
            break
print(f"{ziel}: {n/1e6:.1f} MB aus {len(bf)} BF16-Tensoren")
