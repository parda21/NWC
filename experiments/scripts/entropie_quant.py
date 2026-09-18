"""Wie viel Luft haette ein Entropiekodierer auf quantisierten Gewichten? Nimmt echte BF16-Tensoren (Qwen3-4B),
quantisiert sie zu FP8 (E4M3), INT8 und INT4 (gruppenweise absmax, Gruppe 128) und misst die empirische Entropie
(Bit je Wert) gegen die Speicherbreite. Nur I/O + numpy, keine GPU.
nutzung: python scripts/entropie_quant.py [--quelle models/Qwen3-4B] [--tensoren 24]"""
import os, sys, glob, argparse, json
import numpy as np, torch
from safetensors import safe_open
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--quelle", default=f"{ROOT}/models/Qwen3-4B")
ap.add_argument("--tensoren", type=int, default=24)
a = ap.parse_args()

def entropie(sym, bits):
    z = np.bincount(sym.astype(np.int64).ravel(), minlength=1 << bits).astype(np.float64)
    p = z[z > 0] / z.sum()
    return float(-(p * np.log2(p)).sum())

def fp8_e4m3(w):
    q = w.to(torch.float8_e4m3fn)                      # PyTorch-Konvertierung (saturierend)
    return q.view(torch.uint8).numpy()

def intq(w, bits, g=128):
    x = w.float().reshape(-1, g)
    s = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / (2 ** (bits - 1) - 1)
    q = torch.round(x / s).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1).to(torch.int64) + 2 ** (bits - 1)
    return q.numpy()

dateien = sorted(glob.glob(os.path.join(a.quelle, "*.safetensors")))
n_gesehen, summe = 0, {}
print(f"{'Tensor':45s} {'Werte':>10s} | {'BF16-High':>9s} {'FP8':>6s} {'INT8':>6s} {'INT4':>6s}  (Bit je Wert, empirische Entropie)")
for f in dateien:
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if n_gesehen >= a.tensoren: break
            if not (k.endswith("proj.weight") or k.endswith("embed_tokens.weight")): continue
            w = sf.get_tensor(k)
            if w.dtype != torch.bfloat16 or w.dim() != 2 or w.numel() % 128: continue
            w = w[: min(w.shape[0], 4096)].contiguous()   # Ausschnitt reicht fuer die Statistik
            hi = (w.view(torch.int16).numpy().view(np.uint16) >> 8).astype(np.uint8)
            e = {"bf16hi": entropie(hi, 8), "fp8": entropie(fp8_e4m3(w), 8), "int8": entropie(intq(w, 8), 8), "int4": entropie(intq(w, 4), 4)}
            for kk, v in e.items(): summe[kk] = summe.get(kk, 0) + v * w.numel()
            n_gesehen += 1; summe["n"] = summe.get("n", 0) + w.numel()
            print(f"{k[-45:]:45s} {w.numel():10d} | {e['bf16hi']:9.2f} {e['fp8']:6.2f} {e['int8']:6.2f} {e['int4']:6.2f}")
    if n_gesehen >= a.tensoren: break
n = summe["n"]
print("\nGewichtetes Mittel (Bit je Wert) und daraus die verlustfreie Rate gegen die Speicherbreite:")
print(f"  BF16: Exponentbyte {summe['bf16hi']/n:.2f} + Mantisse 8 -> Rate {(summe['bf16hi']/n + 8)/16:.3f}")
print(f"  FP8 (E4M3): {summe['fp8']/n:.2f} von 8 Bit -> Rate {summe['fp8']/n/8:.3f}")
print(f"  INT8 (g128): {summe['int8']/n:.2f} von 8 Bit -> Rate {summe['int8']/n/8:.3f}  (+ Skalen)")
print(f"  INT4 (g128): {summe['int4']/n:.2f} von 4 Bit -> Rate {summe['int4']/n/4:.3f}  (+ Skalen)")
print("\nDekoder-Bedarf bei 452 GB/s: Symbole je Sekunde, die der Kernel liefern muesste, um die Bandbreite zu saettigen")
for name, bits_raw, bits_c in (("BF16 (1 Symbol je 2 B)", 16, summe['bf16hi']/n + 8), ("FP8", 8, summe['fp8']/n), ("INT8", 8, summe['int8']/n), ("INT4", 4, summe['int4']/n)):
    print(f"  {name:24s}: {452e9 / (bits_c / 8) / 1e9:7.0f} G Symbole/s   (heute gemessen: ~330 G/s)")
