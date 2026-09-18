"""Kernel-Profil je Token: nativ vs NWC auf Qwen2.5-3B. nutzung: python profil3b.py [--nativ]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity
nativ = "--nativ" in sys.argv
P = "models/Qwen2.5-3B"
for i, a in enumerate(sys.argv):
    if a == "--modell": P = sys.argv[i + 1]
layers = None
for i, a in enumerate(sys.argv):
    if a == "--layers": layers = int(sys.argv[i + 1])
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True).eval()
if layers:                                              # Modell auf die ersten `layers` Schichten kuerzen
    model.model.layers = model.model.layers[:layers]
    model.config.num_hidden_layers = layers
    print(f"gekuerzt auf {layers} Schichten")
if nativ: model.cuda()
else:
    from nwc.nwc_torch import konvertiere, fusiere
    if "--fusion" in sys.argv: fusiere(model)
    konvertiere(model)
ids = tok("Die Kompression von Modellgewichten", return_tensors="pt").input_ids.cuda()
N = 32
with torch.no_grad():
    model.generate(ids, max_new_tokens=8, do_sample=False)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        model.generate(ids, max_new_tokens=N, do_sample=False)
ev = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)
ges = sum(e.self_device_time_total for e in ev)
print(f"\n{'nativ' if nativ else 'NWC'}: GPU-Zeit je Token {ges/N/1000:.2f} ms\n")
print(f"{'Kernel':70s} {'ms/Token':>9s} {'Anteil':>7s} {'Aufrufe/Token':>14s}")
for e in ev[:14]:
    print(f"{e.key[:70]:70s} {e.self_device_time_total/N/1000:9.3f} {100*e.self_device_time_total/ges:6.1f}% {e.count/N:14.1f}")
