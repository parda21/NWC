"""NWC for PyTorch: NWCLinear keeps BF16 weights losslessly compressed in VRAM.

  batch 1 (token generation) -> fused matvec, decoded in registers
  otherwise (prefill)        -> dequantization into a temporary BF16 buffer + F.linear
"""
import ctypes, os
import torch, torch.nn as nn, torch.nn.functional as F

LANES = 32
NPAIR = 1024                      # size of the freq/psym buffers (v8 pair alphabet; v9 uses psym[0..15])
TARGET_THREADS = 160_000          # ~2.3 waves on an RTX 4070 (block choice for formats < 9)


def _find_library():
    """Search order: NWC_DLL (A/B tests), nwc/lib/ (pip wheel or `python -m nwc.build`), build/ (repository build)."""
    name = "nwc_ops.dll" if os.name == "nt" else "nwc_ops.so"
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.environ.get("NWC_DLL"), os.path.join(here, "lib", name), os.path.join(os.path.dirname(here), "build", name)]
    for c in candidates:
        if c and os.path.exists(c): return c
    raise ImportError("NWC library not found: install the wheel for this platform or run `python -m nwc.build` "
                      "(needs nvcc); looked for " + ", ".join(c for c in candidates if c))


_LIB_PATH = _find_library()
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for _p in os.environ.get("CUDA_PATH", ""), os.path.join(os.environ.get("CUDA_PATH", ""), "bin"):
        if _p and os.path.isdir(_p): os.add_dll_directory(_p)
_lib = ctypes.CDLL(_LIB_PATH)
P = ctypes.c_void_p
U64, U32 = ctypes.c_uint64, ctypes.c_uint32
try:                              # v8+: the library reports states per lane, table size, header bytes per block
    _lib.nwc_info.restype = ctypes.c_int; _lib.nwc_info.argtypes = [ctypes.c_int]
    VERSION, NSTATE, TAB_WORDS, HDR_BLOCK = (_lib.nwc_info(i) for i in range(4))
except AttributeError:            # v7 library: 2 states, main table u32[4096] + pair table u16[1024], 64 length bytes
    VERSION, NSTATE, TAB_WORDS, HDR_BLOCK = 7, 2, 4096 + NPAIR // 2, 64
X_FP32 = VERSION >= 8             # v8+: the token path takes x as fp32
V9 = VERSION >= 9                 # v9: prefix code, fixed block of 8 rows x 512 columns (any M x K), encoder/dequant need K
FIXED_BLOCK = _lib.nwc_info(4) if V9 else 0
if V9:
    _lib.nwc_layout.restype = ctypes.c_int; _lib.nwc_layout.argtypes = [U64, U32, P]
BLOCK_MAX = int(os.environ.get("NWC_BLOCK_MAX", 4096))   # formats < 9: 2048 and 4096 equally fast, 8192 slower (L1 working set)
_lib.nwc_encode.restype = ctypes.c_int64
_lib.nwc_encode.argtypes = [P, U64, U32, P, P, P, P, P, U64, P]
_lib.nwc_build_tab.argtypes = [P, P, P]
_lib.nwc_gather.restype = ctypes.c_int
_lib.nwc_gather.argtypes = [P, P, P, P, P, P, P, U32, P, U64, U32, U32]
_lib.nwc_matvec.restype = ctypes.c_int
_lib.nwc_matvec.argtypes = [P, P, P, P, P, P, P, P, U64, U32, U32]
_lib.nwc_dequant.restype = ctypes.c_int
_lib.nwc_dequant.argtypes = [P, P, P, P, P, P, P, U64, U32] + ([U32] if V9 else [])
_lib.nwc_linear_bf16.restype = ctypes.c_int
_lib.nwc_linear_bf16.argtypes = [P, P, P, P, P, P, P, P, P, P, U64, U32, U32]
_lib.nwc_setup.restype = ctypes.c_int
_lib.nwc_setup()


def _ptr(t): return ctypes.c_void_p(t.data_ptr())
def _stream(): return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def fits(out_features, in_features):
    """Whether a matrix of this shape can be compressed with the loaded library."""
    n = out_features * in_features
    if V9: return in_features >= 1 and out_features >= 1          # v9 pads to 8 rows x 512 columns
    return n % 512 == 0 and in_features % 16 == 0 and in_features >= 512


def choose_block(n):
    """Largest block (multiple of 512, <= BLOCK_MAX) that divides n and yields >= TARGET_THREADS lane streams."""
    if V9: return FIXED_BLOCK
    for block in (8192, 4096, 2048, 1024, 512):
        if block <= BLOCK_MAX and n % block == 0 and n // block * LANES >= TARGET_THREADS:
            return block
    for block in (512, 1024, 2048, 4096, 8192):            # small tensors: smallest block that fits
        if n % block == 0:
            return block
    raise ValueError(n)


class NWCWeight:
    """Compressed representation of an [out, in] BF16 matrix on the GPU."""
    def __init__(self, w: torch.Tensor, device="cuda", block=None):
        assert w.dtype == torch.bfloat16 and w.dim() == 2
        out_f, in_f = w.shape
        assert fits(out_f, in_f), f"shape {tuple(w.shape)} not supported"
        n = out_f * in_f
        block = block or choose_block(n)
        if V9:                                                     # blocks, mantissa bytes (stride K16), scratch, K16 from the library
            lay = torch.empty(4, dtype=torch.int64)
            if _lib.nwc_layout(out_f, in_f, _ptr(lay)): raise RuntimeError("nwc_layout")
            nb, n_low, n_scratch, self.K16 = lay.tolist()
        else:
            nb, n_low, n_scratch, self.K16 = n // block, n, out_f * (in_f // block + 2), in_f
        w16 = w.detach().contiguous().cpu().view(torch.int16)
        freq = torch.empty(NPAIR, dtype=torch.int16)               # v8: pair frequencies; v9: unused
        psym = torch.empty(NPAIR, dtype=torch.int16)               # v8: index -> two exponent bytes; v9: exp_of_rank[16]
        bases = torch.empty(nb, dtype=torch.int32)
        hdr = torch.empty(nb * HDR_BLOCK, dtype=torch.uint8)      # header per block (v9: stream length in words per lane)
        low = torch.empty(n_low, dtype=torch.uint8)
        for cap in (n + n // 2 + nb * 64 + 4096, 3 * n + nb * 64 + 4096):   # 12 bits per weight is practically always enough
            data = torch.empty(cap, dtype=torch.uint8)
            dl = _lib.nwc_encode(_ptr(w16), n, in_f if V9 else block, _ptr(freq), _ptr(psym), _ptr(bases), _ptr(hdr), _ptr(data), cap, _ptr(low))
            if dl >= 0: break
        if dl < 0: raise RuntimeError("nwc_encode failed")
        lut = torch.empty(TAB_WORDS, dtype=torch.int32)
        _lib.nwc_build_tab(_ptr(freq), _ptr(psym), _ptr(lut))
        self.out_features, self.in_features, self.n, self.block = out_f, in_f, n, block
        self.data = data[:dl + 64].clone().to(device)
        self.bases = bases.to(device)
        self.hdr = hdr.to(device)
        self.lut = lut.to(device)
        self.low = low.to(device)
        self.bytes = dl + self.bases.numel() * 4 + self.hdr.numel() + self.lut.numel() * 4 + n_low
        self.bytes_bf16 = 2 * n
        self.scratch = torch.empty(n_scratch, dtype=torch.float32, device=device)

    @classmethod
    def from_tensors(cls, out_f, in_f, block, data, bases, hdr, lut, low, device="cuda"):
        """Compressed representation from stored tensors (nwc.checkpoint), without the BF16 originals."""
        self = cls.__new__(cls)
        n = out_f * in_f
        if V9:
            lay = torch.empty(4, dtype=torch.int64)
            if _lib.nwc_layout(out_f, in_f, _ptr(lay)): raise RuntimeError("nwc_layout")
            nb, n_low, n_scratch, self.K16 = lay.tolist()
        else:
            nb, n_low, n_scratch, self.K16 = n // block, n, out_f * (in_f // block + 2), in_f
        if bases.numel() != nb or low.numel() != n_low: raise ValueError("checkpoint does not match this library version")
        self.out_features, self.in_features, self.n, self.block = out_f, in_f, n, block
        self.data, self.bases, self.hdr, self.lut, self.low = (t.to(device) for t in (data, bases, hdr, lut, low))
        self.bytes = self.data.numel() + self.bases.numel() * 4 + self.hdr.numel() + self.lut.numel() * 4 + n_low
        self.bytes_bf16 = 2 * n
        self.scratch = torch.empty(n_scratch, dtype=torch.float32, device=device)
        return self

    def tensors(self):
        """The five tensors that fully describe the compressed state (for nwc.checkpoint)."""
        return {"data": self.data, "bases": self.bases, "hdr": self.hdr, "lut": self.lut, "low": self.low}

    def _x16(self, x_f32):
        """v9: extend x to K16 (multiple of 16) with zeros if K is not one."""
        return x_f32 if self.K16 == self.in_features else F.pad(x_f32, (0, self.K16 - self.in_features))

    def matvec(self, x_f32: torch.Tensor) -> torch.Tensor:
        """fp32 test path: y = W x with fp32 output."""
        y = torch.empty(self.out_features, dtype=torch.float32, device=x_f32.device)
        xin = self._x16(x_f32.contiguous())
        err = _lib.nwc_matvec(_stream(), _ptr(self.data), _ptr(self.bases), _ptr(self.hdr), _ptr(self.lut),
                              _ptr(self.low), _ptr(xin), _ptr(y), self.n, self.block, self.in_features)
        if err: raise RuntimeError(f"nwc_matvec CUDA error {err}")
        return y

    def linear_bf16(self, x: torch.Tensor, bias) -> torch.Tensor:
        """Token path: y = W x + bias, x BF16 or fp32, y BF16."""
        y = torch.empty(self.out_features, dtype=torch.bfloat16, device=x.device)
        xin = (x if x.dtype == torch.float32 else x.float()) if X_FP32 else x   # v8+: x fp32
        if V9: xin = self._x16(xin)
        err = _lib.nwc_linear_bf16(_stream(), _ptr(self.data), _ptr(self.bases), _ptr(self.hdr), _ptr(self.lut),
                                   _ptr(self.low), _ptr(xin), _ptr(bias) if bias is not None else None,
                                   _ptr(self.scratch), _ptr(y), self.n, self.block, self.in_features)
        if err: raise RuntimeError(f"nwc_linear_bf16 CUDA error {err}")
        return y

    def gather(self, ids: torch.Tensor) -> torch.Tensor:
        """Rows ids -> BF16[..., in_features] (embedding lookup straight from the compressed state)."""
        flat = ids.reshape(-1).to(device=self.data.device, dtype=torch.int64).contiguous()
        out = torch.empty(flat.numel(), self.K16, dtype=torch.bfloat16, device=self.data.device)
        err = _lib.nwc_gather(_stream(), _ptr(self.data), _ptr(self.bases), _ptr(self.hdr), _ptr(self.lut),
                              _ptr(self.low), _ptr(flat), flat.numel(), _ptr(out), self.n, self.block, self.in_features)
        if err: raise RuntimeError(f"nwc_gather CUDA error {err}")
        if self.K16 != self.in_features: out = out[:, :self.in_features].contiguous()
        return out.view(*ids.shape, self.in_features)

    def dequant(self) -> torch.Tensor:
        """The full BF16 matrix (row-major); a view with row stride K16 if K is not a multiple of 16."""
        w = torch.empty((self.out_features, self.K16), dtype=torch.bfloat16, device=self.data.device)
        err = _lib.nwc_dequant(_stream(), _ptr(self.data), _ptr(self.bases), _ptr(self.hdr), _ptr(self.lut),
                               _ptr(self.low), _ptr(w), self.n, self.block, *((self.in_features,) if V9 else ()))
        if err: raise RuntimeError(f"nwc_dequant CUDA error {err}")
        return w if self.K16 == self.in_features else w[:, :self.in_features]


class NWCLinear(nn.Module):
    """Drop-in replacement for nn.Linear on a compressed matrix."""
    def __init__(self, linear: nn.Linear = None, device="cuda", block=None, w: "NWCWeight" = None, bias=None):
        super().__init__()
        self.w = w if w is not None else NWCWeight(linear.weight.data, device, block)
        self.in_features, self.out_features = self.w.in_features, self.w.out_features
        if linear is not None and linear.bias is not None: bias = linear.bias.data
        self.bias = None if bias is None else nn.Parameter(bias.to(device=device, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, x):
        if x.numel() == self.in_features and x.dtype == torch.bfloat16:   # one token -> fused kernel
            xf = x if x.is_contiguous() else x.contiguous()
            y = self.w.linear_bf16(xf.view(-1), None if self.bias is None else self.bias.data)
            return y.view(*x.shape[:-1], self.out_features)
        return F.linear(x, self.w.dequant(), self.bias)         # prefill -> temporary dequantization

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, block={self.w.block}, "
                f"{self.w.bytes/1e6:.1f} MB (BF16: {self.w.bytes_bf16/1e6:.1f} MB)")


class NWCEmbedding(nn.Module):
    """nn.Embedding replacement on a compressed matrix (shares the weight with the tied lm_head)."""
    def __init__(self, emb: nn.Embedding = None, w: NWCWeight = None, padding_idx=None):
        super().__init__()
        self.w = w
        self.num_embeddings, self.embedding_dim = w.out_features, w.in_features
        self.padding_idx = emb.padding_idx if emb is not None else padding_idx

    def forward(self, ids):
        return self.w.gather(ids)

    def extra_repr(self):
        return f"{self.num_embeddings}, {self.embedding_dim}, {self.w.bytes/1e6:.1f} MB (BF16: {self.w.bytes_bf16/1e6:.1f} MB)"


class _FusedHead(nn.Module):
    """One output slice of a fused projection: head 0 computes the whole fused matvec, the others pick up their slice."""
    def __init__(self, shared, index, start, end):
        super().__init__()
        self.g, self.index, self.start, self.end = shared, index, start, end
        self.in_features, self.out_features = shared["lin"].in_features, end - start
    def forward(self, x):
        g = self.g
        if self.index == 0 or g["x_id"] != id(x) or g["y"] is None:
            g["y"] = g["lin"](x); g["x_id"] = id(x)
        return g["y"][..., self.start:self.end]


def fuse(model: nn.Module, groups=(("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")), device="cuda"):
    """Concatenate sibling nn.Linear layers with the same input into one NWC matrix (fewer, larger kernel launches).
    Call before `convert`; the fused layers are NWC afterwards. Returns the number of fused groups."""
    n_fused = 0
    for mod in list(model.modules()):
        for names in groups:
            lins = [getattr(mod, nm, None) for nm in names]
            if not all(isinstance(l, nn.Linear) for l in lins): continue
            if len({l.in_features for l in lins}) != 1 or any(l.weight.dtype != torch.bfloat16 for l in lins): continue
            W = torch.cat([l.weight.data for l in lins], 0)
            if not fits(W.shape[0], W.shape[1]): continue
            b = None
            if all(l.bias is not None for l in lins): b = torch.cat([l.bias.data for l in lins], 0)
            elif any(l.bias is not None for l in lins):
                b = torch.cat([l.bias.data if l.bias is not None else torch.zeros(l.out_features, dtype=W.dtype) for l in lins], 0)
            whole = nn.Linear(W.shape[1], W.shape[0], bias=b is not None, dtype=torch.bfloat16)
            whole.weight.data = W
            if b is not None: whole.bias.data = b
            lin = NWCLinear(whole, device)
            shared = {"lin": lin, "y": None, "x_id": None, "names": list(names), "sizes": [l.out_features for l in lins]}
            a = 0
            for i, (nm, l) in enumerate(zip(names, lins)):
                setattr(mod, nm, _FusedHead(shared, i, a, a + l.out_features)); a += l.out_features
                l.weight.data = torch.empty(0)
            n_fused += 1
    print(f"NWC: {n_fused} projection groups fused")
    return n_fused


def convert(model: nn.Module, device="cuda", verbose=True, tied=True):
    """Replace every supported nn.Linear (and a tied embedding) by NWC modules; move the rest to `device`.
    Returns (bytes_nwc, bytes_bf16) of the compressed matrices."""
    total_nwc = total_bf16 = 0
    replaced, skipped, blocks = 0, [], {}
    shared_ptrs = {m.weight.data_ptr() for m in model.modules() if isinstance(m, nn.Embedding)}   # tied lm_head
    # tied embedding + lm_head: one compressed matrix for both (lookup via gather, output via matvec)
    if tied:
        embs = {m.weight.data_ptr(): m for m in model.modules() if isinstance(m, nn.Embedding)
                and m.weight.dtype == torch.bfloat16 and fits(m.num_embeddings, m.embedding_dim)}
        lins = [(name, mod, child, cm) for name, mod in list(model.named_modules()) for child, cm in list(mod.named_children())
                if isinstance(cm, nn.Linear) and cm.weight.data_ptr() in embs]
        for name, mod, child, cm in lins:
            emb = embs[cm.weight.data_ptr()]
            w = NWCWeight(emb.weight.data, device)
            setattr(mod, child, NWCLinear(cm, device, w=w))
            for ename, emod in list(model.named_modules()):
                for echild, ecm in list(emod.named_children()):
                    if ecm is emb: setattr(emod, echild, NWCEmbedding(emb, w))
            total_nwc += w.bytes; total_bf16 += w.bytes_bf16; replaced += 1
            blocks[w.block] = blocks.get(w.block, 0) + 1
            shared_ptrs.discard(cm.weight.data_ptr())
            emb.weight.data = torch.empty(0); cm.weight.data = torch.empty(0)
    for m in model.modules():                                     # count layers fused earlier
        if isinstance(m, _FusedHead) and m.index == 0:
            w = m.g["lin"].w; total_nwc += w.bytes; total_bf16 += w.bytes_bf16; replaced += 1
            blocks[w.block] = blocks.get(w.block, 0) + 1
    for name, mod in list(model.named_modules()):
        for child, cm in list(mod.named_children()):
            if isinstance(cm, nn.Linear) and cm.weight.dtype == torch.bfloat16 \
               and fits(cm.out_features, cm.in_features) and cm.weight.data_ptr() not in shared_ptrs:
                new = NWCLinear(cm, device)
                setattr(mod, child, new)
                total_nwc += new.w.bytes; total_bf16 += new.w.bytes_bf16; replaced += 1
                blocks[new.w.block] = blocks.get(new.w.block, 0) + 1
                cm.weight.data = torch.empty(0)
            elif isinstance(cm, nn.Linear):
                reason = "tied" if cm.weight.data_ptr() in shared_ptrs else "shape"
                skipped.append(f"{name}.{child} {tuple(cm.weight.shape)} [{reason}]")
    model.to(device)
    if verbose:
        print(f"NWC: {replaced} linear layers replaced, {total_bf16/1e9:.2f} GB BF16 -> {total_nwc/1e9:.2f} GB "
              f"(ratio {total_nwc/max(total_bf16,1):.4f}); block sizes: {dict(sorted(blocks.items()))}")
        if skipped: print("  skipped:", ", ".join(skipped[:6]), "..." if len(skipped) > 6 else "")
    return total_nwc, total_bf16
