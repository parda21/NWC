"""Verschraenkter rANS (range Asymmetric Numeral Systems).

N unabhaengige Zustaende, Symbol i nutzt Zustand i % N. Dekodierung laeuft
damit N-fach parallel -- Voraussetzung fuer SIMD und CUDA.
"""

SKALA_BITS = 12
M = 1 << SKALA_BITS
L = 1 << 23                    # untere Schranke, Byte-Renormalisierung


def normiere(zaehler, M=M):
    """Haeufigkeiten auf Summe genau M skalieren, benutzte Symbole >= 1."""
    gesamt = sum(zaehler)
    if gesamt == 0:
        return [0] * 256
    freq = [0] * 256
    for i, c in enumerate(zaehler):
        if c:
            freq[i] = max(1, (c * M) // gesamt)
    diff = M - sum(freq)
    reihenfolge = sorted((i for i in range(256) if freq[i]), key=lambda i: -freq[i])
    k = 0
    while diff != 0:
        i = reihenfolge[k % len(reihenfolge)]
        if diff > 0:
            freq[i] += 1; diff -= 1
        elif freq[i] > 1:
            freq[i] -= 1; diff += 1
        k += 1
    assert sum(freq) == M
    return freq


def tabellen(freq):
    kum = [0] * 257
    for i in range(256):
        kum[i + 1] = kum[i] + freq[i]
    slot2sym = bytearray(M)
    for s in range(256):
        for j in range(kum[s], kum[s + 1]):
            slot2sym[j] = s
    return kum, bytes(slot2sym)


def kodiere(daten, freq, kum, n_streams=16, renorm=8):
    """renorm=8: Byte-Renormalisierung, L=2^23 (NWC1).
       renorm=16: 16-Bit-Renormalisierung, L=2^16 -- hoechstens EIN Nachladen
                  pro Symbol, kein Schleifenkopf auf der GPU (NWC2)."""
    L_ = L if renorm == 8 else (1 << 16)
    zustand = [L_] * n_streams
    puffer = bytearray()                       # rueckwaerts aufgebaut
    for i in range(len(daten) - 1, -1, -1):
        s = daten[i]
        f = freq[s]
        k = i % n_streams
        x = zustand[k]
        x_max = ((L_ >> SKALA_BITS) << renorm) * f
        while x >= x_max:
            if renorm == 8:
                puffer.append(x & 0xFF)
            else:
                puffer.append(x & 0xFF); puffer.append((x >> 8) & 0xFF)
            x >>= renorm
        zustand[k] = ((x // f) << SKALA_BITS) + (x % f) + kum[s]
    for k in range(n_streams):                 # Zustaende zuletzt -> Dekoder liest sie zuerst
        x = zustand[k]
        puffer += bytes([x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF, (x >> 24) & 0xFF])
    puffer.reverse()
    return bytes(puffer)


def dekodiere(nutzlast, n_symbole, freq, kum, slot2sym, n_streams=16, renorm=8):
    L_ = L if renorm == 8 else (1 << 16)
    p = 0
    zustand = [0] * n_streams
    for k in range(n_streams - 1, -1, -1):     # Spiegelbild der Kodierreihenfolge
        b = nutzlast
        zustand[k] = (b[p] << 24) | (b[p+1] << 16) | (b[p+2] << 8) | b[p+3]
        p += 4
    aus = bytearray(n_symbole)
    maske = M - 1
    for i in range(n_symbole):
        k = i % n_streams
        x = zustand[k]
        slot = x & maske
        s = slot2sym[slot]
        aus[i] = s
        x = freq[s] * (x >> SKALA_BITS) + slot - kum[s]
        if renorm == 8:
            while x < L_:
                x = (x << 8) | nutzlast[p]; p += 1
        elif x < L_:                            # 16 Bit: genau ein Nachladen reicht
            x = (x << 16) | (nutzlast[p] << 8) | nutzlast[p + 1]; p += 2
        zustand[k] = x
    return bytes(aus)
