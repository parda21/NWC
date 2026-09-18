"""Build the CUDA library from csrc/nwc_ops.cu into nwc/lib/ (for installations from source).
usage: python -m nwc.build [--arch sm_86,sm_89] [--nvcc PATH]
Without --arch the result is a fatbin for sm_80, sm_86, sm_89, sm_90 plus PTX for compute_90 (JIT-compiled on newer
GPUs). Windows needs the MSVC environment (vcvars64.bat) in the calling terminal or Visual Studio in its default path."""
import os, shutil, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(os.path.dirname(HERE), "csrc", "nwc_ops.cu")
TARGET_DIR = os.path.join(HERE, "lib")
ARCHS = ["sm_80", "sm_86", "sm_89", "sm_90"]


def gencode(archs):
    args = []
    for a in archs:
        n = a.split("_")[1]
        args += ["-gencode", f"arch=compute_{n},code=sm_{n}"]
    highest = max(int(a.split("_")[1]) for a in archs)
    args += ["-gencode", f"arch=compute_{highest},code=compute_{highest}"]     # PTX for newer GPUs
    return args


def find_nvcc(path=None):
    if path: return path
    n = shutil.which("nvcc")
    if n: return n
    for var in ("CUDA_PATH", "CUDA_HOME"):
        p = os.environ.get(var)
        if p and os.path.exists(os.path.join(p, "bin", "nvcc" + (".exe" if os.name == "nt" else ""))):
            return os.path.join(p, "bin", "nvcc")
    raise SystemExit("nvcc not found (PATH, CUDA_PATH or --nvcc)")


def build(archs=None, nvcc=None, target_dir=TARGET_DIR, source=SOURCE):
    if not os.path.exists(source): raise SystemExit(f"source missing: {source} (install from the git repository)")
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, "nwc_ops.dll" if os.name == "nt" else "nwc_ops.so")
    cmd = [find_nvcc(nvcc), "-O3", "--shared", "-o", target, source] + gencode(archs or ARCHS)
    if os.name != "nt": cmd += ["-Xcompiler", "-fPIC"]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    for ext in ("exp", "lib"):                                                  # MSVC by-products
        p = os.path.join(target_dir, "nwc_ops." + ext)
        if os.path.exists(p): os.remove(p)
    print("built:", target)
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="", help="comma list, e.g. sm_86,sm_89 (default: sm_80..sm_90 + PTX)")
    ap.add_argument("--nvcc", default=None)
    a = ap.parse_args()
    build([x.strip() for x in a.arch.split(",") if x.strip()] or None, a.nvcc)
