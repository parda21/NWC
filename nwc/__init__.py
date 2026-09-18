"""NWC -- Neural Weight Compression: lossless BF16 weight compression with a fused GPU matvec.

    from nwc import convert, fuse, save_pretrained, load_pretrained
    fuse(model); convert(model)              # nn.Linear -> NWCLinear, model to the GPU
    save_pretrained(model, "Qwen3-4B-NWC")   # write a compressed checkpoint
    model = load_pretrained("Qwen3-4B-NWC")  # load without the BF16 originals
"""
__version__ = "0.9.0"

from .nwc_torch import NWCWeight, NWCLinear, NWCEmbedding, convert, fuse, fits, choose_block   # noqa: E402
from .checkpoint import save_pretrained, load_pretrained                                       # noqa: E402

__all__ = ["NWCWeight", "NWCLinear", "NWCEmbedding", "convert", "fuse", "fits", "choose_block",
           "save_pretrained", "load_pretrained", "__version__"]
