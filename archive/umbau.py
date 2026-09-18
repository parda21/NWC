"""Einmaliger Umbau der Projektstruktur. Verschiebt, loescht Zwischenstaende, passt Importe an."""
import os, re, shutil, glob
R = os.path.dirname(os.path.abspath(__file__))
os.chdir(R)

ordner = ["nwc", "csrc", "experiments", "scripts", "tests", "build", "data", "archive"]
for o in ordner: os.makedirs(o, exist_ok=True)

ziel = {
    "nwc":         ["nwc_torch.py", "rans.py", "nwc.py", "nwc_gpu.py", "gguf.py"],
    "csrc":        ["nwcenc.c", "nwc_ops.cu", "nwcenc_fse.c", "nwcdec.c", "nwcdec_mt.c"],
    "experiments": ["bandbreite.cu", "nwcdec_cuda.cu", "nwcdec_cuda2.cu", "nwcdec_cuda3.cu", "nwcdec_cuda4.cu",
                    "nwcdec_cuda5.cu", "matvec.cu", "matvec2.cu", "matvec3.cu", "matvec4.cu", "matvec5.cu",
                    "matvec6.cu", "test_fse.c", "analyse.py", "kontext.py", "stride.py", "verify.py", "warum.py",
                    "final.py", "dump.py", "dump2.py", "mach_nostride.py", "mach_v2.py", "test_gpu_format.py",
                    "test_nwc.py", "test_rans.py"],
    "scripts":     ["bench_qwen.py", "profil3b.py", "extrahiere.py", "dl_qwen.py", "zeitprofil.py"],
    "tests":       ["test_nwc_torch.py", "test_k.py"],
    "data":        ["W.raw", "W4.nwc", "gpu.raw", "logits3b.pt"],
    "archive":      ["komp_original.py"],
    "build":       ["nwc_ops.dll", "nwcenc.exe", "nwcenc_fse.exe", "matvec6.exe", "nwcdec_cuda5_th128.exe"],
}
for o, dateien in ziel.items():
    for f in dateien:
        if os.path.exists(f): shutil.move(f, os.path.join(o, f))

# Zwischenstaende loeschen
loeschen = ["W.nwc", "W5.nwc", "big.raw", "big_32768.nwc", "big_8192.nwc", "gpu_32_131072.nwc", "gpu_32_32768.nwc",
            "gpu_32_8192.nwc", "gpu_64_131072.nwc", "gpu_nostride.nwc"] + glob.glob("v2_*.nwc") + glob.glob("v3_*.nwc") \
           + glob.glob("*.exe") + glob.glob("*.obj") + glob.glob("*.lib") + glob.glob("*.exp") + glob.glob("test.*")
for f in loeschen:
    if os.path.exists(f): os.remove(f)
shutil.rmtree("__pycache__", ignore_errors=True)

# Paket
open("nwc/__init__.py", "w").write('"""NWC -- verlustfreie BF16-Gewichtskompression mit fusioniertem GPU-Matvec."""\n')
def ersetze(pfad, paare):
    s = open(pfad, encoding="utf-8").read(); s0 = s
    for a, b in paare: s = re.sub(a, b, s)
    if s != s0: open(pfad, "w", encoding="utf-8").write(s)

ersetze("nwc/nwc.py",     [(r"^from rans import", "from .rans import")])
ersetze("nwc/nwc_gpu.py", [(r"^from rans import", "from .rans import"), (r"^from nwc import", "from .nwc import")])
ersetze("nwc/nwc_torch.py", [(r'os\.path\.join\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\), "nwc_ops\.dll"\)',
                              'os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "nwc_ops.dll")')])

kopf = ("import os, sys\n"
        "ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
        "sys.path.insert(0, ROOT)\n")
importe = [(r"^from nwc_torch import", "from nwc.nwc_torch import"),
           (r"^from gguf import", "from nwc.gguf import"),
           (r"^from rans import", "from nwc.rans import"),
           (r"^from nwc import (komprimiere|dekomprimiere|suche_stride)", r"from nwc.nwc import \1"),
           (r"^import nwc, nwc_gpu$", "from nwc import nwc as nwc, nwc_gpu"),
           (r"^import nwc_gpu$", "from nwc import nwc_gpu"),
           (r"^from nwc_gpu import", "from nwc.nwc_gpu import"),
           (r"^from nwc import \*", "from nwc.nwc import *"),
           (r'open\("(W\.raw|gpu\.raw|big\.raw)", "rb"\)', r'open(os.path.join(ROOT, "data", "\1"), "rb")'),
           (r'"logits3b\.pt"', r'os.path.join(ROOT, "data", "logits3b.pt")'),
           (r'local_dir="models', r'local_dir=ROOT + "/models')]
for pfad in glob.glob("scripts/*.py") + glob.glob("tests/*.py") + glob.glob("experiments/*.py"):
    s = open(pfad, encoding="utf-8").read()
    if "ROOT = " not in s:
        # Kopf nach dem Docstring einfuegen
        m = re.match(r'(\s*""".*?"""\s*\n)', s, re.S)
        s = (m.group(1) + kopf + s[m.end():]) if m else (kopf + s)
        open(pfad, "w", encoding="utf-8").write(s)
    ersetze(pfad, [(p, q) for p, q in importe])
    # regex mit ^ braucht MULTILINE -> nochmal explizit
    s = open(pfad, encoding="utf-8").read()
    for p, q in importe:
        s = re.sub(p, q, s, flags=re.M)
    open(pfad, "w", encoding="utf-8").write(s)

open(".gitignore", "w").write("build/\ndata/\nmodels/\n__pycache__/\n*.pyc\n*.obj\n*.lib\n*.exp\n")
print("Umbau fertig.")
for o in ordner:
    print(f"  {o + '/':14s}", ", ".join(sorted(os.listdir(o)))[:150])
