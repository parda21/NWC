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
    print("OK" if ok else "FAIL")
finally:
    shutil.rmtree(d, ignore_errors=True)
