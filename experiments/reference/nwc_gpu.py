"""NWC2 -- GPU-taugliche Variante von NWC1.

Unterschied: statt EINES verschraenkten Bytestroms je Block gibt es LANES
voellig unabhaengige Teilstroeme. Lane L dekodiert die Symbole
L, L+LANES, L+2*LANES, ... ihres Blocks.

  -> Lesen  : jede Lane hat ihren eigenen Bytebereich, keine Synchronisation
  -> Schreiben: benachbarte Lanes schreiben benachbarte Adressen (coalesced)

Kosten: je Teilstrom 4 Byte Endzustand + 2 Byte Offset.
"""
import struct
from collections import Counter
from rans import normiere, tabellen, kodiere, dekodiere
from nwc import suche_stride, _delta, _undelta

MAGIC = b"NWC2"


def komprimiere(bf16, dims=None, lanes=32, block=32768, stride_fest=None, renorm=16):
    assert len(bf16) % 2 == 0
    low, high = bf16[0::2], bf16[1::2]
    s = suche_stride(high, dims) if stride_fest is None else stride_fest
    res = _delta(high, s)

    z = [0] * 256
    for b in res: z[b] += 1
    freq = normiere(z); kum, _ = tabellen(freq)

    blockbasen, subrel, daten = [], [], bytearray()
    for start in range(0, len(res), block):
        stueck = res[start:start + block]
        blockbasen.append(len(daten))
        rel = []
        for lane in range(lanes):
            rel.append(len(daten) - blockbasen[-1])
            daten += kodiere(stueck[lane::lanes], freq, kum, 1, renorm)
        assert max(rel) < 65536, "Block zu gross fuer 16-bit-Offsets"
        subrel.append(rel)
    n_bloecke = len(blockbasen)

    kopf = bytearray(MAGIC)
    kopf += struct.pack("<BBBB", 2, lanes, 12, renorm)
    kopf += struct.pack("<QIII", len(high), s, block, n_bloecke)
    kopf += struct.pack(f"<{256}H", *freq)
    kopf += struct.pack(f"<{n_bloecke}I", *blockbasen)
    for rel in subrel:
        kopf += struct.pack(f"<{lanes}H", *rel)
    return bytes(kopf) + bytes(daten) + low


def lies_kopf(paket):
    _, lanes, skala, renorm = struct.unpack("<BBBB", paket[4:8])
    renorm = renorm or 8                       # 0 = altes Format mit Byte-Renormalisierung
    n_high, s, block, n_bloecke = struct.unpack("<QIII", paket[8:28])
    p = 28
    freq = list(struct.unpack_from(f"<{256}H", paket, p)); p += 512
    basen = list(struct.unpack_from(f"<{n_bloecke}I", paket, p)); p += 4 * n_bloecke
    subrel = []
    for _ in range(n_bloecke):
        subrel.append(list(struct.unpack_from(f"<{lanes}H", paket, p))); p += 2 * lanes
    return dict(lanes=lanes, n_high=n_high, stride=s, block=block, n_bloecke=n_bloecke,
                freq=freq, basen=basen, subrel=subrel, daten_start=p, renorm=renorm)


def dekomprimiere(paket):
    assert paket[:4] == MAGIC
    h = lies_kopf(paket)
    kum, slot2sym = tabellen(h["freq"])
    n_high, lanes, block = h["n_high"], h["lanes"], h["block"]
    ds, low_start = h["daten_start"], len(paket) - h["n_high"]

    res = bytearray(n_high)
    for b in range(h["n_bloecke"]):
        basis = b * block
        n_sym = min(block, n_high - basis)
        for lane in range(lanes):
            n_lane = (n_sym - lane + lanes - 1) // lanes
            if n_lane <= 0: continue
            a = ds + h["basen"][b] + h["subrel"][b][lane]
            sub = dekodiere(paket[a:], n_lane, h["freq"], kum, slot2sym, 1, h["renorm"])
            res[basis + lane : basis + n_sym : lanes] = sub

    high = _undelta(bytes(res), h["stride"])
    aus = bytearray(2 * n_high)
    aus[0::2] = paket[low_start:]
    aus[1::2] = high
    return bytes(aus)
