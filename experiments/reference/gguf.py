import struct
from collections import Counter

GGML_TYP = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
            10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",
            24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",30:"BF16"}
TYP_GROESSE = {"F32":4,"F16":2,"BF16":2,"I8":1,"I16":2,"I32":4,"I64":8,"F64":8}

class Leser:
    def __init__(s, f): s.f = f
    def u32(s): return struct.unpack("<I", s.f.read(4))[0]
    def u64(s): return struct.unpack("<Q", s.f.read(8))[0]
    def i32(s): return struct.unpack("<i", s.f.read(4))[0]
    def txt(s):
        n = s.u64(); return s.f.read(n).decode("utf-8", "replace")
    def wert(s, t):
        if t == 8: return s.txt()
        if t == 9:
            et = s.u32(); n = s.u64()
            return [s.wert(et) for _ in range(n)]
        groesse = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t]
        fmt = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<B",10:"<Q",11:"<q",12:"<d"}[t]
        return struct.unpack(fmt, s.f.read(groesse))[0]

def lies_gguf(pfad):
    f = open(pfad, "rb"); r = Leser(f)
    assert f.read(4) == b"GGUF", "kein GGUF"
    version = r.u32(); n_tensoren = r.u64(); n_meta = r.u64()
    meta = {}
    for _ in range(n_meta):
        k = r.txt(); t = r.u32(); meta[k] = r.wert(t)
    tensoren = []
    for _ in range(n_tensoren):
        name = r.txt()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        typ = GGML_TYP.get(r.u32(), "?")
        off = r.u64()
        tensoren.append({"name": name, "dims": dims, "typ": typ, "off": off,
                         "n": 1 if not dims else __import__("math").prod(dims)})
    align = meta.get("general.alignment", 32)
    pos = f.tell()
    daten_start = (pos + align - 1) // align * align
    return f, version, meta, tensoren, daten_start

if __name__ == "__main__":
    P = "models/mmproj-Bonsai-27B-BF16.gguf"
    f, ver, meta, tensoren, start = lies_gguf(P)
    print(f"GGUF v{ver}, {len(tensoren)} Tensoren, Datenbereich ab Offset {start}")
    c = Counter(t["typ"] for t in tensoren)
    print("Typen:", dict(c))
    bf = [t for t in tensoren if t["typ"] == "BF16"]
    gesamt = sum(t["n"] for t in bf)
    print(f"BF16-Tensoren: {len(bf)}, {gesamt:,} Gewichte = {gesamt*2/1e6:.1f} MB")
    print("\ngroesste BF16-Tensoren:")
    for t in sorted(bf, key=lambda t: -t["n"])[:5]:
        print(f"  {t['name'][:48]:48s} {str(t['dims']):20s} {t['n']*2/1e6:8.2f} MB  @{start+t['off']}")
