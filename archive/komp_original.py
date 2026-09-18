from collections import Counter

MAX_EINTRAEGE = 255
MIN_LAENGE = 3
MAX_LAENGE = 16


def finde_kandidaten(data: bytes) -> Counter:
    zaehler = Counter()
    n = len(data)
    for laenge in range(MIN_LAENGE, MAX_LAENGE + 1):
        if laenge > n:
            break
        for i in range(n - laenge + 1):
            zaehler[data[i:i + laenge]] += 1
    return zaehler


def waehle_woerterbuch(data: bytes) -> list:
    kandidaten = finde_kandidaten(data)
    bewertet = []
    for muster, anzahl in kandidaten.items():
        if anzahl < 2:
            continue
        gewinn = (len(muster) - 2) * anzahl
        kosten = 1 + len(muster)
        netto = gewinn - kosten
        if netto > 0:
            bewertet.append((netto, muster))
    bewertet.sort(key=lambda x: (-x[0], -len(x[1])))
    woerterbuch = []
    for _, muster in bewertet:
        if len(woerterbuch) >= MAX_EINTRAEGE:
            break
        if any(muster in vorhanden for vorhanden in woerterbuch):
            continue
        woerterbuch.append(muster)
    return woerterbuch


def waehle_escape(data: bytes) -> int:
    zaehler = Counter(data)
    for kandidat in range(256):
        if zaehler[kandidat] == 0:
            return kandidat
    return min(range(256), key=lambda b: zaehler[b])


def baue_handbuch(escape, woerterbuch):
    out = bytearray([escape, len(woerterbuch)])
    for muster in woerterbuch:
        out.append(len(muster))
        out.extend(muster)
    return bytes(out)


def lies_handbuch(handbuch):
    if not handbuch:
        return 0, []
    escape = handbuch[0]
    anzahl = handbuch[1]
    woerterbuch = []
    pos = 2
    for _ in range(anzahl):
        laenge = handbuch[pos]
        pos += 1
        woerterbuch.append(handbuch[pos:pos + laenge])
        pos += laenge
    return escape, woerterbuch


def compress(data):
    if not data:
        return b"", b""
    woerterbuch = waehle_woerterbuch(data)
    escape = waehle_escape(data)
    handbuch = baue_handbuch(escape, woerterbuch)
    reihenfolge = sorted(range(len(woerterbuch)), key=lambda i: -len(woerterbuch[i]))
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        treffer = None
        for idx in reihenfolge:
            muster = woerterbuch[idx]
            if data.startswith(muster, i):
                treffer = idx
                break
        if treffer is not None:
            out.append(escape)
            out.append(treffer)
            i += len(woerterbuch[treffer])
        else:
            byte = data[i]
            if byte == escape:
                out.append(escape)
                out.append(255)
            else:
                out.append(byte)
            i += 1
    return bytes(out), handbuch


def decompress(payload, handbuch):
    if not handbuch:
        return payload
    escape, woerterbuch = lies_handbuch(handbuch)
    out = bytearray()
    i = 0
    n = len(payload)
    while i < n:
        byte = payload[i]
        if byte == escape and i + 1 < n:
            idx = payload[i + 1]
            if idx == 255:
                out.append(escape)
            else:
                out.extend(woerterbuch[idx])
            i += 2
        else:
            out.append(byte)
            i += 1
    return bytes(out)
