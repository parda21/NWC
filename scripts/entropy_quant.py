"""How much would lossless coding save on top of quantized weights? Quantizes every linear matrix of a BF16 model
to int8 (per output channel, symmetric), fp8 e4m3 (per output channel) and int4 (groups of 128), and reports the
zero-order entropy of the quantized symbols, i.e. the size an ideal memoryless coder (rANS) would reach, plus the
cost of the v9-style rank prefix code. BF16 is included for reference (exponent byte coded, mantissa raw).
usage: python scripts/entropy_quant.py [--model models/Qwen3-4B] [--device cuda] [--min 1000000]"""
import os, sys, glob, math, argparse
import torch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=f"{ROOT}/models/Qwen3-4B")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--min", type=int, default=1_000_000, help="only matrices with at least this many weights")
a = ap.parse_args()


def entropy_bits(hist):
    p = hist[hist > 0].double(); p = p / p.sum()
    return float(-(p * p.log2()).sum())


def prefix_code_bits(hist, maxr=15, escape_bits=None):
    """v9-style code: symbols ranked by frequency, rank r < maxr costs r+1 bits, others maxr+1 + escape_bits."""
    n = hist.sum().item(); esc = escape_bits if escape_bits is not None else math.ceil(math.log2(hist.numel()))
    ranked = hist.sort(descending=True).values.double()
    cost = sum((r + 1) * ranked[r].item() for r in range(min(maxr, ranked.numel())))
    cost += (maxr + 1 + esc) * ranked[maxr:].sum().item()
    return cost / n


def hist_u(x, bins): return torch.bincount(x.reshape(-1).to(torch.int64), minlength=bins)


totals = {}   # scheme -> [weights, bits_entropy, bits_prefix, bits_stored]
def add(scheme, n, h_bits, p_bits, stored):
    t = totals.setdefault(scheme, [0, 0.0, 0.0, 0.0])
    t[0] += n; t[1] += h_bits * n; t[2] += p_bits * n; t[3] += stored * n


files = sorted(glob.glob(os.path.join(a.model, "*.safetensors")))
if not files: sys.exit(f"no safetensors in {a.model}")
n_mats = 0
for f in files:
    with safe_open(f, "pt") as s:
        for k in s.keys():
            w = s.get_tensor(k)
            if w.dim() != 2 or w.numel() < a.min or w.dtype != torch.bfloat16: continue
            n_mats += 1
            w = w.to(a.device)
            M, K = w.shape; n = w.numel()
            # BF16 as NWC codes it: high byte (sign + 7 exponent bits) coded, low byte raw
            hi = (w.view(torch.int16).to(torch.int32) & 0xFFFF) >> 8
            h_hi = hist_u(hi, 256)
            # sign bit raw + 7-bit exponent coded (that is the v9 code); the ideal coder codes the byte as a whole
            add("BF16 (NWC today)", n, entropy_bits(h_hi) + 8, prefix_code_bits(hist_u(hi & 0x7F, 128), escape_bits=8) + 1 + 8, 16)
            wf = w.float()
            # int8, symmetric per output channel
            scale = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
            q8 = torch.round(wf / scale).clamp(-127, 127).to(torch.int32) + 128
            h8 = hist_u(q8, 256)
            add("int8 per-channel", n, entropy_bits(h8), prefix_code_bits(h8), 8 + 16 / K)
            # fp8 e4m3, per output channel scaled to the format's range
            s8 = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
            f8 = (wf / s8).to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
            hf = hist_u(f8, 256)
            exp_part = f8 >> 3                       # sign + 4 exponent bits (5 bits), mantissa 3 bits raw
            add("fp8 e4m3 per-channel", n, entropy_bits(hf), prefix_code_bits(hist_u(exp_part, 32)) + 3, 8 + 16 / K)
            # splits the v9 decoder can do: a small coded alphabet (rank prefix code, <= 16 symbols so no escape) + raw bits
            # fp8: 4-bit exponent coded, sign + 3-bit mantissa raw (a 4-bit raw plane)
            e4 = (f8 >> 3) & 0xF
            he = hist_u(e4, 16)
            add("  fp8: exp4 coded + 4 raw", n, entropy_bits(he) + 4, prefix_code_bits(he, maxr=16) + 4, 8 + 16 / K)
            # fp8 as the v9 decoder could run it: symbol = exponent >> 1 (8 symbols + escape for e = 0 / 15), sign coded raw
            # behind the rank code, low exponent bit + 3-bit mantissa as a raw nibble plane
            e0 = e4 == 0; e15 = e4 == 15
            hv = hist_u((e4 >> 1).masked_fill(e0 | e15, 8), 9)
            esc_frac = (e0 | e15).float().mean().item()
            add("  fp8: exp>>1 coded + sign + 4 raw", n, entropy_bits(hv) + 5, prefix_code_bits(hv, maxr=8, escape_bits=8) + 5, 8 + 16 / K)
            add("  fp8: escapes (fraction, x1000)", n, esc_frac * 1000, esc_frac * 1000, 1)
            # int8 sign-magnitude: top k bits of |q| coded, the rest raw, sign raw
            mag = (q8 - 128).abs()
            for k in (7, 6, 5, 4):
                top = mag >> (7 - k)
                ht = hist_u(top, 1 << k)
                maxr = 16 if k <= 4 else 15
                add(f"  int8: top{k} coded + {8-k} raw", n, entropy_bits(ht) + (8 - k), prefix_code_bits(ht, maxr=maxr, escape_bits=k) + (8 - k), 8 + 16 / K)
            # int4, groups of 128 along K, symmetric absmax
            G = 128
            if K % G == 0:
                wg = wf.view(M, K // G, G)
                s4 = wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / 7.0
                q4 = torch.round(wg / s4).clamp(-8, 7).to(torch.int32) + 8
                h4 = hist_u(q4, 16)
                add("int4 group-128", n, entropy_bits(h4), prefix_code_bits(h4), 4 + 16 / G)
            del w, wf

print(f"{a.model}: {n_mats} linear matrices, {totals['BF16 (NWC today)'][0]/1e9:.2f} G weights\n")
print(f"{'scheme':<24}{'stored bits/w':>14}{'entropy bits/w':>16}{'ideal coder':>13}{'v9 prefix code':>16}{'GB at 4B':>10}")
for scheme, (n, hb, pb, sb) in totals.items():
    hb, pb, sb = hb / n, pb / n, sb / n
    print(f"{scheme:<24}{sb:>14.2f}{hb:>16.2f}{hb/sb:>13.3f}{pb/sb:>16.3f}{hb*n/8/1e9:>10.2f}")
print("\nstored: the quantized format incl. scales; entropy: zero-order, per matrix, weighted; ideal coder = entropy/stored;\n"
      "v9 prefix code = what the current rank code would reach (escape beyond rank 15). GB at 4B: entropy-coded size of these matrices.")
