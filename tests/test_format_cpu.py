"""Format v9 reference decoder in plain Python: encodes matrices with the library's host encoder and decodes the
resulting streams without the GPU, checking bit-exactness. Documents the format independently of the CUDA kernel and
runs on machines without a GPU (CI). usage: python tests/test_format_cpu.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from nwc.nwc_torch import NWCWeight, VERSION, FIXED_BLOCK, HDR_BLOCK, LANES

ROWS, COLS, PEEK, MAXR, FLAG = 8, 512, 12, 15, 1 << 30
assert VERSION in (9, 10, 11) and FIXED_BLOCK == ROWS * COLS and HDR_BLOCK == LANES


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
    """Both layouts. Layout 1: 16-row blocks, lane L = (r = L >> 2, q = L & 3) holds, per 32-column super-chunk, weight
    i (0..15) at row r + 8 * ((i >> 1) & 1), column 8q + 4 * ((i >> 3) & 1) + 2 * ((i >> 2) & 1) + (i & 1) (two
    mma.m16n8k16 A fragments); the raw plane is [row group][super-chunk slot][lane][16 bytes]; 16-column chunks with
    columns >= K16 are not coded (an odd count leaves the last super-chunk half full)."""
    M, K, K16 = c.out_features, c.in_features, c.K16
    CB = (K + COLS - 1) // COLS
    rows = 16 if c.layout else ROWS
    nl = (K16 - (CB - 1) * COLS) // 16; nsl = (nl + 1) // 2; spr = 16 * (CB - 1) + nsl
    data, bases, hdr, low = bytes(c.data.cpu().tolist()), c.bases.cpu().tolist(), c.hdr.cpu().tolist(), c.low.cpu()
    eor = exp_of_rank(c.lut.cpu().tolist())
    high = torch.zeros(M, K, dtype=torch.int32)
    lo = torch.zeros(M, K, dtype=torch.int32)
    for b in range(len(bases)):
        rb, cb = divmod(b, CB)
        pos = bases[b]
        nchunk = nl if cb == CB - 1 else 32
        for L in range(LANES):
            n_words = hdr[b * HDR_BLOCK + L]
            r = Bits(data, pos, n_words); pos += 4 * n_words
            nw = ((nchunk + 1) // 2) * 16 if c.layout else ROWS * COLS // LANES
            for i in range(nw):
                if c.layout:
                    j = i & 15
                    row = rb * 16 + (L >> 2) + 8 * ((j >> 1) & 1)
                    col = cb * COLS + (i >> 4) * 32 + 8 * (L & 3) + 4 * ((j >> 3) & 1) + 2 * ((j >> 2) & 1) + (j & 1)
                else:
                    row, col = rb * ROWS + (i >> 4), cb * COLS + L * 16 + (i & 15)
                zeros = 0
                while r.bit() == 0: zeros += 1
                if zeros < MAXR: sign = r.bit(); e = (sign << 7) | eor[zeros]   # rank code + sign
                else: sign = None; e = r.bits(8)                                  # escape: raw byte
                if row < M and col < K:
                    high[row, col] = e
                    if c.layout: lo[row, col] = int(low[((rb * spr + cb * 16 + (i >> 4)) * LANES + L) * 16 + j])
                else: assert zeros == 0 and sign == 0, "filler weights are coded as rank 0, sign 0"
    if not c.layout: lo = low[:M * K16].view(M, K16)[:, :K].to(torch.int32)
    return ((high << 8) | lo).to(torch.int16).view(torch.bfloat16)


def decode2(c: NWCWeight) -> torch.Tensor:
    """Layout 2 (BF16): blocks and lane order of layout 1; per weight a nibble [sign | rank] in the nibble plane
    [slot][lane][8 bytes] (nibble j of a super-chunk in byte j >> 1, low nibble for even j), the low byte in the raw
    plane [slot][lane][16 bytes] behind it; rank 7 marks an exception whose 16-bit value sits in `data` (per block
    at bases[b], the lanes' entries of 3 bytes (index, lo, hi) one after the other, hdr = entries per lane)."""
    M, K, K16 = c.out_features, c.in_features, c.K16
    CB = (K + COLS - 1) // COLS
    nl = (K16 - (CB - 1) * COLS) // 16; nsl = (nl + 1) // 2; spr = 16 * (CB - 1) + nsl
    Mp = (M + 15) // 16 * 16
    data, bases, hdr, low = bytes(c.data.cpu().tolist()), c.bases.cpu().tolist(), c.hdr.cpu().tolist(), c.low.cpu().tolist()
    lut = c.lut.cpu().tolist()
    hb = [(int(lut[r >> 2]) >> (8 * (r & 3))) & 0xFF for r in range(8)]          # high byte (no sign) of rank 0..6, rank 7 = 0
    off_r = (Mp // 16) * spr * LANES * 8
    w = torch.zeros(M, K, dtype=torch.int32)
    for b in range(len(bases)):
        rb, cb = divmod(b, CB)
        nchunk = nl if cb == CB - 1 else 32
        pos = bases[b]
        for L in range(LANES):
            n_exc = hdr[b * HDR_BLOCK + L]
            exc = {data[pos + 3 * k]: data[pos + 3 * k + 1] | (data[pos + 3 * k + 2] << 8) for k in range(n_exc)}
            assert len(exc) == n_exc
            pos += 3 * n_exc
            for i in range(((nchunk + 1) // 2) * 16):
                sc, j = i >> 4, i & 15
                row = rb * 16 + (L >> 2) + 8 * ((j >> 1) & 1)
                col = cb * COLS + sc * 32 + 8 * (L & 3) + 4 * ((j >> 3) & 1) + 2 * ((j >> 2) & 1) + (j & 1)
                slot = (rb * spr + cb * 16 + sc) * LANES + L
                nib = (low[slot * 8 + (j >> 1)] >> (4 * (j & 1))) & 15
                raw = low[off_r + slot * 16 + j]
                if row >= M or col >= K: assert nib == 0 and raw == 0, "fillers are zero"; continue
                s, rank = nib >> 3, nib & 7
                if rank < 7: w[row, col] = ((s << 7) | hb[rank]) << 8 | raw
                else: assert raw == 0; w[row, col] = exc.pop(i)
            assert not exc, "every exception entry belongs to a coded weight"
    return w.to(torch.int16).view(torch.bfloat16)


def check(M, K, scale=0.02, seed=0, layout=0):
    torch.manual_seed(seed)
    w = (torch.randn(M, K) * scale).to(torch.bfloat16)
    w.view(-1)[:: max(1, M * K // 50)] = torch.tensor(1e30, dtype=torch.bfloat16)   # rare exponents -> escapes
    c = NWCWeight(w, device="cpu", layout=layout)
    back = decode2(c) if layout == 2 else decode(c)
    ok = torch.equal(back, w)
    print(f"layout {layout}: {M:>5} x {K:<5}  {c.bytes/1e3:8.1f} KB, ratio {c.bytes/c.bytes_bf16:.3f}, blocks {c.bases.numel():4d}  ->  "
          f"{'bit-exact' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    from nwc.nwc_torch import LAYOUT1, LAYOUT2
    results = []
    for layout in [0] + ([1] if LAYOUT1 else []) + ([2] if LAYOUT2 else []):
        results += [check(8, 512, layout=layout), check(13, 40, layout=layout), check(3, 4097, layout=layout),
                    check(100, 1000, scale=1.0, layout=layout), check(64, 2048, seed=1, layout=layout), check(17, 520, layout=layout)]
    print("OK" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)
