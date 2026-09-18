"""Warp-Zeitprofil des bf16-Kernels (DLL mit -DTIMING). Liest je Warp: Eintritt, nach Init, Chunk-Anfaenge, Ende.
nutzung: NWC_DLL=build/nwc_ops_timing.dll python scripts/warpzeiten.py [block] [M] [K]"""
import os, sys, ctypes
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch, numpy as np
from nwc.nwc_torch import NWCWeight, _dll

block = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
M = int(sys.argv[2]) if len(sys.argv) > 2 else 151936
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2560
TIM_MAX, TIM_N = 48000, 20
n = M * K
raw = open(os.path.join(ROOT, "data", "W.raw"), "rb").read()
w = torch.frombuffer(bytearray(raw[:n * 2]), dtype=torch.int16).view(M, K).view(torch.bfloat16)
nw = NWCWeight(w, block=block)
xb = (torch.rand(K) - 0.5).cuda().to(torch.bfloat16); bias = torch.zeros(M, dtype=torch.bfloat16).cuda()
for _ in range(5): nw.linear_bf16(xb, bias)
torch.cuda.synchronize()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record(); nw.linear_bf16(xb, bias); e.record(); torch.cuda.synchronize()
print(f"block {block}, M {M}, K {K}: Kernel {s.elapsed_time(e):.3f} ms (inkl. Finalize)")

buf = np.zeros(TIM_MAX * TIM_N, dtype=np.uint64)
_dll.nwc_tim_lesen.argtypes = [ctypes.c_void_p]
assert _dll.nwc_tim_lesen(buf.ctypes.data) == 0
nb = min(n // block, TIM_MAX); nch = block // 512
t = buf.reshape(TIM_MAX, TIM_N)[:nb].astype(np.int64)
sm = (t[:, 0] >> 56) & 0xFF
MASK = 0x00FFFFFFFFFFFFFF
t0, tend = t[:, 0] & MASK, t[:, 19] & MASK          # globaltimer (ns, ~1 us Aufloesung)
c_in, c1, c_end = t[:, 18], t[:, 1], t[:, 2 + nch] if nch < 16 else t[:, 18]   # clock64 (Takte)
ch = t[:, 2:2 + nch]
ok = (tend > 0) & (t[:, 1] > 0)
t0, tend, ch, sm, c_in, c1 = t0[ok], tend[ok], ch[ok], sm[ok], c_in[ok], c1[ok]
T0 = t0.min(); dauer = (tend.max() - T0) / 1e3
print(f"Warps: {ok.sum()}, Kernel-Spanne aus Stempeln: {dauer:.1f} us")
print(f"Init (Eintritt -> Init-Loads ausgegeben): median {np.median(c1 - c_in):.0f} Takte")
lebt = ch[:, 0] * 0  # Lebensdauer in Takten: Eintritt -> letzter Chunk-Anfang + letzte Chunkdauer unbekannt -> nutze globaltimer
print(f"Warp-Lebensdauer (globaltimer): median {np.median(tend - t0)/1e3:.2f} us, p10 {np.percentile(tend - t0, 10)/1e3:.2f}, p90 {np.percentile(tend - t0, 90)/1e3:.2f}")
ct = np.diff(ch, axis=1)                              # Dauer je Chunk (Takte), letzter Chunk fehlt
print("Chunk-Dauer (median Takte) je Chunk-Index:", " ".join(f"{np.median(ct[:, c]):.0f}" for c in range(nch - 1)))
print(f"  erster Chunk (inkl. Warten auf Init-Daten): {np.median(ct[:, 0]):.0f} Takte, uebrige: {np.median(ct[:, 1:]) if nch > 2 else 0:.0f} Takte; Init->Chunk0: {np.median(ch[:, 0] - c1):.0f}")
# Effektive Zahl gleichzeitig aktiver Warps je SM (Zeitmittel ueber die Kernel-Spanne, 0-95 % um Anlauf/Tail zu meiden)
a, b_ = T0 + 0.05 * (tend.max() - T0), T0 + 0.95 * (tend.max() - T0)
akt = []
for s_ in np.unique(sm):
    m = sm == s_
    st, en = np.clip(t0[m], a, b_), np.clip(tend[m], a, b_)
    akt.append((en - st).sum() / (b_ - a))
print(f"aktive Warps je SM (Zeitmittel, 5-95 %): mittel {np.mean(akt):.1f}, min {np.min(akt):.1f}, max {np.max(akt):.1f} ({len(akt)} SMs)")
# Durchsatz: Symbole je us je SM
print(f"Symbole je Warp-us: {block / max(np.median(tend - t0), 1) * 1e3:.0f}")
