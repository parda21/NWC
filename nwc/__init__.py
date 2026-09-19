"""NWC -- Neural Weight Compression: lossless BF16 weight compression with a fused GPU matvec.

    from nwc import convert, fuse, save_pretrained, load_pretrained
    fuse(model); convert(model)              # nn.Linear -> NWCLinear, model to the GPU (lossless BF16)
    fuse(model, elem="fp8"); convert(model, elem="fp8")   # or weight-only fp8, the fp8 values stored lossless
    save_pretrained(model, "Qwen3-4B-NWC")   # write a compressed checkpoint
    model = load_pretrained("Qwen3-4B-NWC")  # load without the BF16 originals (or a HF repo id)
    export_bf16("Qwen3-4B-NWC", "Qwen3-4B")  # back to a plain BF16 checkpoint, bit-identical
"""
__version__ = "0.10.0"

from .nwc_torch import NWCWeight, NWCLinear, NWCEmbedding, convert, fuse, fits, choose_block, quantize_fp8   # noqa: E402
from .checkpoint import save_pretrained, load_pretrained, export_bf16                                      # noqa: E402

__all__ = ["NWCWeight", "NWCLinear", "NWCEmbedding", "convert", "fuse", "fits", "choose_block",
           "quantize_fp8", "save_pretrained", "load_pretrained", "export_bf16", "__version__"]
