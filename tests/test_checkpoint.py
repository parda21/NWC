"""Write and load a compressed checkpoint without downloading a model: small Qwen2 model with random weights.
Checks: convert + fuse -> save_pretrained -> load_pretrained -> identical logits, size bookkeeping, embedding gather.
usage: python tests/test_checkpoint.py"""
import os, sys, tempfile, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from transformers import Qwen2Config, AutoModelForCausalLM
from nwc import fuse, convert, save_pretrained, load_pretrained

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=4096, hidden_size=256, intermediate_size=1000, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=True)
model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()   # like load_pretrained(): buffers (inv_freq) stay fp32
ids = torch.randint(0, cfg.vocab_size, (1, 12))
with torch.no_grad(): ref = model(ids).logits.float().cuda()          # BF16 reference on the CPU
state = {k: v.clone() for k, v in model.state_dict().items()}          # originals, for the export check

fuse(model); convert(model)
with torch.no_grad(): a = model(ids.cuda()).logits.float()
d = tempfile.mkdtemp(prefix="nwc_ckpt_")
try:
    c = save_pretrained(model, d, base_model="test")
    print(f"saved: {c['bytes_bf16']/1e6:.2f} MB BF16 -> {c['bytes_nwc']/1e6:.2f} MB, {len(c['weights'])} matrices, {len(c['modules'])} modules")
    del model; torch.cuda.empty_cache()
    m2 = load_pretrained(d, verbose=False)
    with torch.no_grad(): b = m2(ids.cuda()).logits.float()
    same = torch.equal(a, b)
    dref = (b - ref).abs().max().item()
    print(f"logits identical after loading: {same}; max |d| to the BF16 CPU reference {dref:.3e}")
    ok = same and dref < 0.5 and all(p.device.type == "cuda" for p in m2.parameters())
    # export back to a plain BF16 checkpoint: every original parameter must come back bit-identical
    from nwc import export_bf16
    from safetensors.torch import load_file
    e = os.path.join(d, "bf16")
    export_bf16(d, e, verbose=False)
    exported = load_file(os.path.join(e, "model.safetensors"))
    missing = [k for k in state if k not in exported and not (k == "lm_head.weight" and cfg.tie_word_embeddings)]
    differ = [k for k in state if k in exported and not torch.equal(exported[k].to(state[k].dtype), state[k])]
    print(f"export: {len(exported)} tensors, missing {missing[:3]}, differing {differ[:3]}")
    m3 = AutoModelForCausalLM.from_pretrained(e, dtype=torch.bfloat16).eval()
    with torch.no_grad(): c3 = m3(ids).logits.float().cuda()
    print(f"exported model logits identical to the original: {torch.equal(c3, ref)}")
    ok = ok and not missing and not differ and torch.equal(c3, ref)
    # fp8 weight-only: convert -> save -> load must reproduce the converted model's logits exactly
    from nwc.nwc_torch import ELEM
    if ELEM:
        m4 = AutoModelForCausalLM.from_pretrained(e, dtype=torch.bfloat16).eval()
        fuse(m4, elem="fp8"); convert(m4, elem="fp8")
        with torch.no_grad(): a4 = m4(ids.cuda()).logits.float()
        d8 = os.path.join(d, "fp8"); save_pretrained(m4, d8, base_model="test"); del m4; torch.cuda.empty_cache()
        m5 = load_pretrained(d8, verbose=False)
        with torch.no_grad(): b4 = m5(ids.cuda()).logits.float()
        same8 = torch.equal(a4, b4); dq = (b4 - ref).abs().max().item()
        print(f"fp8: logits identical after loading: {same8}; max |d| to the BF16 model {dq:.3e} (quantization, expected > 0)")
        ok = ok and same8
    print("OK" if ok else "FAIL")
finally:
    shutil.rmtree(d, ignore_errors=True)
