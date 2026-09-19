"""Environment check: is this machine ready to run NWC, and if not, what to do about it.
usage: python -m nwc.doctor
Exit code 0 when everything needed for the token path works (a compressed matrix is round-tripped on the GPU)."""
import os, sys, platform, subprocess

OK, WARN, FAIL = "ok  ", "warn", "FAIL"


def _nvidia_smi():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return [tuple(x.strip() for x in line.split(",")) for line in out if line.strip()]
    except Exception:
        return []


def main():
    problems, warnings, rows = [], [], []
    def row(status, what, detail=""):
        rows.append((status, what, detail))
        if status == FAIL: problems.append(what)
        if status == WARN: warnings.append(what)

    row(OK, "python", f"{platform.python_version()} on {platform.system()} {platform.machine()}")
    try:
        import torch
        row(OK, "torch", f"{torch.__version__}, CUDA build {torch.version.cuda}" if torch.version.cuda
            else f"{torch.__version__} (CPU-only build)")
        if not torch.version.cuda:
            row(FAIL, "torch has no CUDA support", "install a CUDA build: https://pytorch.org/get-started/locally/")
    except ImportError:
        torch = None
        row(FAIL, "torch not installed", "pip install torch  (CUDA build, see https://pytorch.org/get-started/locally/)")

    smi = _nvidia_smi()
    if smi:
        for name, drv, total, used in smi: row(OK, "gpu (driver)", f"{name}, driver {drv}, {used} of {total} in use")
    else:
        row(WARN, "nvidia-smi not found", "cannot report the driver version; the checks below tell whether CUDA works")

    cap = None
    if torch is not None and torch.version.cuda:
        if not torch.cuda.is_available():
            row(FAIL, "torch.cuda.is_available() is False",
                "no usable NVIDIA GPU or the driver is older than the CUDA build of torch; update the driver")
        else:
            i = torch.cuda.current_device()
            cap = torch.cuda.get_device_capability(i)
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            row(OK, "cuda device", f"{props.name}, sm_{cap[0]}{cap[1]}, {props.multi_processor_count} SMs, "
                                   f"{free/1e9:.1f} of {total/1e9:.1f} GB free")
            if cap < (7, 5):
                row(FAIL, f"compute capability {cap[0]}.{cap[1]} too old", "NWC needs sm_75 (Turing) or newer")
            elif cap < (8, 0):
                row(WARN, "Turing GPU", "the kernel builds for sm_75 but has not been measured there; please report results")
            fits = [("Qwen3-0.6B", 1.5), ("Qwen3-4B", 6.0), ("Qwen2.5-7B / Llama-3.1-8B", 10.5)]
            row(OK, "fits in free VRAM", ", ".join(m for m, gb in fits if free / 1e9 > gb) or "none of the reference models")

    lib_path = None
    try:
        from . import nwc_torch
        lib_path = nwc_torch._LIB_PATH
        row(OK, "kernel library", f"{lib_path} (format v{nwc_torch.VERSION})")
    except ImportError as e:
        row(FAIL, "kernel library not found", str(e).split(";")[0] + " -> pip install --force-reinstall neural-weight-compression, "
            "or python -m nwc.build (needs nvcc)")
    except OSError as e:
        row(FAIL, "kernel library does not load", f"{e}; on Windows the CUDA runtime DLLs must be in PATH or CUDA_PATH")

    if lib_path and cap is not None and cap >= (7, 5):
        try:
            import torch
            torch.manual_seed(0)
            w = (torch.randn(1000, 2048) * 0.02).to(torch.bfloat16)
            c = nwc_torch.NWCWeight(w, device="cuda")
            back = c.dequant().cpu()
            x = torch.randn(2048).to(torch.bfloat16)
            y = c.linear_bf16(x.cuda().float(), None).cpu().float()
            ref = (w.float() @ x.float())
            if not torch.equal(back, w): row(FAIL, "dequantization is not bit-exact", "please open an issue with this output")
            elif (y - ref).abs().max() > 0.05 * (ref.abs().max() + 1e-3): row(FAIL, "matvec result off", "please open an issue with this output")
            else: row(OK, "gpu round trip", f"1000 x 2048 matrix: dequant bit-exact, matvec matches ({c.bytes/c.bytes_bf16:.2f} of BF16 size)")
        except RuntimeError as e:
            msg = str(e)
            if "no kernel image" in msg or "209" in msg or "invalid device function" in msg:
                row(FAIL, "no kernel for this GPU in the library",
                    f"rebuild for it: python -m nwc.build --arch sm_{cap[0]}{cap[1]}  (needs nvcc)")
            else:
                row(FAIL, "gpu round trip failed", msg)

    for mod, why in (("transformers", "model helpers and the demo"), ("accelerate", "loading compressed checkpoints"),
                     ("huggingface_hub", "downloading checkpoints")):
        try:
            __import__(mod); row(OK, mod, "installed")
        except ImportError:
            row(WARN, f"{mod} not installed", f"needed for {why}: pip install {mod}")

    width = max(len(r[1]) for r in rows)
    for status, what, detail in rows:
        print(f"[{status}] {what.ljust(width)}  {detail}")
    print()
    if problems:
        print(f"{len(problems)} problem(s); fix the FAIL lines above, then run python -m nwc.doctor again.")
        return 1
    print("Ready. Try:  python -m nwc.demo Parda21/Qwen3-0.6B-NWC --load --graph")
    if warnings: print(f"({len(warnings)} warning(s) above are optional.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
