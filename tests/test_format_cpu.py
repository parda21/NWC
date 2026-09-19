"""Format v9 reference decoder in plain Python: encodes matrices with the library's host encoder and decodes the
resulting streams without the GPU, checking bit-exactness. Documents the format independently of the CUDA kernel and
runs on machines without a GPU (CI). usage: python tests/test_format_cpu.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from nwc.nwc_torch import NWCWeight, VERSION, FIXED_BLOCK, HDR_BLOCK, LANES

ROWS, COLS, PEEK, MAXR, FLAG = 8, 512, 12, 15, 1 << 30
assert VERSION == 9 and FIXED_BLOCK == ROWS * COLS and HDR_BLOCK == LANES


def exp_of_rank(lut):
    """The 15 unary ranks -> 7-bit exponent, read back from the lookup table (ranks 0..10 from the patterns
    'r zeros, one, sign 0', ranks 11..14 from entries 0 and 1, which never hold a complete code)."""
    e = [(int(lut[1 << (PEEK - 1 - r)]) >> 8) & 0x7F for r in range(11)]
    e += [(int(lut[0]) >> 8) & 0x7F, (int(lut[0]) >> 16) & 0x7F, (int(lut[1]) >> 8) & 0x7F, (int(lut[1]) >> 16) & 0x7F]
    return e


class Bits:
    """MSB-first reader over little-endian 32-bit words."""
    def __init__(self, data, byte_off, n_words):
        self.words = [int.from_bytes(data[byte_off + 4 * i: byte_off + 4 * i + 4], "little") for i in range(n_words)]
        self.pos = 0
    def bit(self):
        w, b = divmod(self.pos, 32); self.pos += 1
        return (self.words[w] >> (31 - b)) & 1
    def bits(self, n):
        v = 0
        for _ in range(n): v = (v << 1) | self.bit()
        return v


def decode(c: NWCWeight) -> torch.Tensor:
    M, K, K16 = c.out_features, c.in_features, c.K16
    CB = (K + COLS - 1) // COLS
    data, bases, hdr, low = c.data.cpu().numpy().tobytes(), c.bases.cpu().tolist(), c.hdr.cpu().tolist(), c.low.cpu()
    eor = exp_of_rank(c.lut.cpu().tolist())
    high = torch.zeros(M, K, dtype=torch.int32)
    for b in range(len(bases)):
        rb, cb = divmod(b, CB)
        pos = bases[b]
        for L in range(LANES):
            n_words = hdr[b * HDR_BLOCK + L]
            r = Bits(data, pos, n_words); pos += 4 * n_words
            for i in range(ROWS * COLS // LANES):
                row, col = rb * ROWS + (i >> 4), cb * COLS + L * 16 + (i & 15)
                zeros = 0
                while r.bit() == 0: zeros += 1
                if zeros < MAXR: sign = r.bit(); e = (sign << 7) | eor[zeros]   # rank code + sign
                else: sign = None; e = r.bits(8)                                  # escape: raw byte
                if row < M and col < K: high[row, col] = e
                else: assert zeros == 0 and sign == 0, "filler weights are coded as rank 0, sign 0"
    lo = low.view(M, K16)[:, :K].to(torch.int32)
    return ((high << 8) | lo).to(torch.int16).view(torch.bfloat16)


def check(M, K, scale=0.02, seed=0):
    torch.manual_seed(seed)
    w = (torch.randn(M, K) * scale).to(torch.bfloat16)
    w.view(-1)[:: max(1, M * K // 50)] = torch.tensor(1e30, dtype=torch.bfloat16)   # rare exponents -> escapes
    c = NWCWeight(w, device="cpu")
    back = decode(c)
    ok = torch.equal(back, w)
    print(f"{M:>5} x {K:<5}  {c.bytes/1e3:8.1f} KB, ratio {c.bytes/c.bytes_bf16:.3f}, blocks {c.bases.numel():4d}  ->  "
          f"{'bit-exact' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    results = [check(8, 512), check(13, 40), check(3, 4097), check(100, 1000, scale=1.0), check(64, 2048, seed=1)]
    print("OK" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)
