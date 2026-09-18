"""Compressed checkpoints: save a converted model without the BF16 originals and load it back.

Layout of an NWC checkpoint (directory):
  config.json, generation_config.json, tokenizer files   (HF format, as usual)
  nwc_config.json      format version, compressed matrices (shape, block, bytes) and which modules use them
  model.safetensors    all remaining parameters (BF16) + per matrix the five NWC tensors under "nwc/<id>/<name>"
Loading: build the model on the meta device from the config, insert the NWC modules from the tensors, fill the rest.
"""
import os, json
import torch, torch.nn as nn
from safetensors.torch import save_file, load_file
from .nwc_torch import NWCWeight, NWCLinear, NWCEmbedding, _FusedHead, VERSION

FORMAT = 1
TENSOR_NAMES = ("data", "bases", "hdr", "lut", "low")


def _collect(model):
    """All NWCWeight objects (unique) and the modules that use them."""
    ids, weights, modules = {}, [], []
    def key(w):
        if id(w) not in ids:
            ids[id(w)] = len(weights)
            weights.append({"id": len(weights), "out": w.out_features, "in": w.in_features, "block": w.block, "bytes": w.bytes, "w": w})
        return ids[id(w)]
    groups = set()
    for path, m in model.named_modules():
        if isinstance(m, NWCLinear):
            modules.append({"path": path, "kind": "linear", "weight": key(m.w), "bias": f"{path}.bias" if m.bias is not None else None})
        elif isinstance(m, NWCEmbedding):
            modules.append({"path": path, "kind": "embedding", "weight": key(m.w), "padding_idx": m.padding_idx})
        elif isinstance(m, _FusedHead) and id(m.g) not in groups:
            groups.add(id(m.g))
            lin = m.g["lin"]
            parent = path.rsplit(".", 1)[0] if "." in path else ""
            modules.append({"path": parent, "kind": "fused", "weight": key(lin.w), "names": m.g["names"], "sizes": m.g["sizes"],
                            "bias": f"nwc/fused/{parent}.bias" if lin.bias is not None else None})
    return weights, modules


def save_pretrained(model: nn.Module, path: str, tokenizer=None, base_model: str = None):
    """Write a converted model (after convert/fuse) as an NWC checkpoint. Returns the nwc_config dict."""
    os.makedirs(path, exist_ok=True)
    weights, modules = _collect(model)
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}   # rest + registered biases
    for e in weights:
        for name, t in e["w"].tensors().items(): tensors[f"nwc/{e['id']}/{name}"] = t.cpu().contiguous()
    for m in modules:                                                                    # biases of fused groups (not registered)
        if m["kind"] == "fused" and m["bias"]:
            for p, mod in model.named_modules():
                if isinstance(mod, _FusedHead) and (p.rsplit(".", 1)[0] if "." in p else "") == m["path"]:
                    tensors[m["bias"]] = mod.g["lin"].bias.detach().cpu().contiguous(); break
    save_file(tensors, os.path.join(path, "model.safetensors"), metadata={"format": "pt", "nwc": str(FORMAT)})
    config = {"nwc_checkpoint": FORMAT, "library_version": VERSION, "base_model": base_model,
              "weights": [{k: v for k, v in e.items() if k != "w"} for e in weights], "modules": modules,
              "bytes_nwc": sum(e["bytes"] for e in weights), "bytes_bf16": sum(2 * e["out"] * e["in"] for e in weights)}
    with open(os.path.join(path, "nwc_config.json"), "w", encoding="utf-8") as f: json.dump(config, f, indent=1)
    model.config.save_pretrained(path)
    gc = getattr(model, "generation_config", None)
    if gc is not None: gc.save_pretrained(path)
    if tokenizer is not None: tokenizer.save_pretrained(path)
    return config


def _set(model, path, module):
    parent, _, name = path.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, name, module)


def load_pretrained(path: str, device="cuda", verbose=True):
    """Load an NWC checkpoint: returns the model on `device` with NWC modules in place, no BF16 originals in RAM."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    with open(os.path.join(path, "nwc_config.json"), encoding="utf-8") as f: config = json.load(f)
    if config["library_version"] != VERSION:
        raise RuntimeError(f"checkpoint is format version {config['library_version']}, library is {VERSION}")
    hf_config = AutoConfig.from_pretrained(path)
    with init_empty_weights():
        try: model = AutoModelForCausalLM.from_config(hf_config, dtype=torch.bfloat16)
        except TypeError: model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.bfloat16)
    tensors = load_file(os.path.join(path, "model.safetensors"))
    weights = {}
    for e in config["weights"]:
        t = {n: tensors.pop(f"nwc/{e['id']}/{n}") for n in TENSOR_NAMES}
        weights[e["id"]] = NWCWeight.from_tensors(e["out"], e["in"], e["block"], t["data"], t["bases"], t["hdr"], t["lut"], t["low"], device)
    for m in config["modules"]:
        w = weights[m["weight"]]
        if m["kind"] == "linear":
            bias = tensors.pop(m["bias"], None) if m["bias"] else None
            _set(model, m["path"], NWCLinear(w=w, bias=bias, device=device))
        elif m["kind"] == "embedding":
            _set(model, m["path"], NWCEmbedding(w=w, padding_idx=m["padding_idx"]))
        else:
            bias = tensors.pop(m["bias"], None) if m["bias"] else None
            lin = NWCLinear(w=w, bias=bias, device=device)
            shared = {"lin": lin, "y": None, "x_id": None, "names": m["names"], "sizes": m["sizes"]}
            a = 0
            for i, (name, size) in enumerate(zip(m["names"], m["sizes"])):
                _set(model, f"{m['path']}.{name}" if m["path"] else name, _FusedHead(shared, i, a, a + size)); a += size
    for k, v in tensors.items():                                                         # remaining parameters/buffers
        set_module_tensor_to_device(model, k, device, value=v)
    for n, p in model.named_parameters():
        if p.device.type == "meta": raise RuntimeError(f"parameter {n} missing in the checkpoint")
    model.to(device).eval()
    if verbose:
        print(f"NWC: checkpoint loaded, {config['bytes_bf16']/1e9:.2f} GB BF16 -> {config['bytes_nwc']/1e9:.2f} GB "
              f"(ratio {config['bytes_nwc']/max(config['bytes_bf16'],1):.4f}), VRAM in use {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model
