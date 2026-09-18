"""NWC1 -- verlustfreier BF16-Gewichtskompressor mit blockparallelem rANS.

Aufbau:
  1. Byte-Ebenen trennen  : high = Vorz.+Exponent (~2,6 Bit), low = Mantisse (~8 Bit)
  2. Stride-Praediktor    : residual[i] = high[i] XOR high[i-s], s pro Tensor gesucht
  3. Entropiekodierung    : verschraenkter rANS auf residual, unabhaengige Bloecke
  4. Mantisse             : roh durchgereicht (inkompressibel)

Dekodierung ist zweistufig und in beiden Stufen parallel:
  Stufe A  rANS je Block unabhaengig   -> Blockanzahl-fache Parallelitaet
  Stufe B  XOR-Kette je Restklasse     -> s-fache Parallelitaet, coalesced
"""
import struct, math
from collections import Counter
from rans import normiere, tabellen, kodiere, dekodiere, SKALA_BITS, M

MAGIC = b"NWC1"
BLOCK = 1 << 20          # Symbole je rANS-Block
N_STREAMS = 16


def _h0(zaehler, n):
    return -sum((v / n) * math.log2(v / n) for v in zaehler if v)


def suche_stride(high, dims=None, probe=1 << 20):
    """Bester Stride nach Order-0-Entropie des XOR-Residuums. 0 = kein Praediktor."""
    s_probe = high[:probe]
    n = len(s_probe)
    if n < 64:
        return 0
    kand = set(range(1, 17)) | {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}
    if dims:
        for d in dims[:2]:
            for m in (d // 4, d // 2, d, 2 * d):
                if m > 0:
                    kand |= {m - 1, m, m + 1}
    kand = sorted(x for x in kand if 0 < x < n // 4)

    bester, beste_h = 0, _h0(Counter(s_probe).values(), n)
    for s in kand:
        z = [0] * 256
        for i in range(s, n):
            z[s_probe[i] ^ s_probe[i - s]] += 1
        h = _h0(z, n - s)
        if h < beste_h - 1e-4:
            bester, beste_h = s, h
    return bester


def _delta(high, s):
    if s == 0:
        return high
    r = bytearray(high)
    for i in range(len(high) - 1, s - 1, -1):
        r[i] = high[i] ^ high[i - s]
    return bytes(r)


def _undelta(res, s):
    if s == 0:
        return res
    h = bytearray(res)
    for i in range(s, len(h)):
        h[i] ^= h[i - s]
    return bytes(h)


def komprimiere(bf16, dims=None):
    assert len(bf16) % 2 == 0, "bf16 braucht gerade Byteanzahl"
    low, high = bf16[0::2], bf16[1::2]

    s = suche_stride(high, dims)
    res = _delta(high, s)

    z = [0] * 256
    for b in res:
        z[b] += 1
    freq = normiere(z)
    kum, _ = tabellen(freq)

    bloecke, offsets = [], []
    pos = 0
    for start in range(0, len(res), BLOCK):
        stueck = res[start:start + BLOCK]
        p = kodiere(stueck, freq, kum, N_STREAMS)
        offsets.append(pos); pos += len(p); bloecke.append(p)

    kopf = bytearray(MAGIC)
    kopf += struct.pack("<BBBB", 1, N_STREAMS, SKALA_BITS, 0)
    kopf += struct.pack("<QIII", len(high), s, BLOCK, len(bloecke))
    kopf += struct.pack(f"<{256}H", *freq)
    kopf += struct.pack(f"<{len(offsets)}I", *offsets)
    return bytes(kopf) + b"".join(bloecke) + low


def dekomprimiere(paket):
    assert paket[:4] == MAGIC
    _, n_streams, skala, _ = struct.unpack("<BBBB", paket[4:8])
    n_high, s, block, n_bloecke = struct.unpack("<QIII", paket[8:28])
    p = 28
    freq = list(struct.unpack_from(f"<{256}H", paket, p)); p += 512
    offsets = list(struct.unpack_from(f"<{n_bloecke}I", paket, p)); p += 4 * n_bloecke
    kum, slot2sym = tabellen(freq)

    daten_start = p
    low_start = len(paket) - n_high          # Mantisse liegt am Ende, genau n_high Bytes
    res = bytearray()
    for i in range(n_bloecke):
        anfang = daten_start + offsets[i]
        ende = daten_start + offsets[i + 1] if i + 1 < n_bloecke else low_start
        n_sym = min(block, n_high - i * block)
        res += dekodiere(paket[anfang:ende], n_sym, freq, kum, slot2sym, n_streams)

    high = _undelta(bytes(res), s)
    low = paket[low_start:]
    aus = bytearray(2 * n_high)
    aus[0::2] = low
    aus[1::2] = high
    return bytes(aus)
