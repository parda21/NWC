"""Demo: load a HF model, compress it, generate text, report size and speed.
usage:
  python -m nwc.demo Qwen/Qwen3-4B                     # load, compress, generate 64 tokens (greedy)
  python -m nwc.demo Qwen/Qwen3-4B --native            # comparison: uncompressed
  python -m nwc.demo Qwen/Qwen3-4B --fp8               # weight-only fp8 (per-channel scale), the fp8 values stored lossless
  python -m nwc.demo Qwen/Qwen3-4B --graph             # also tokens/s as a CUDA graph (no Python overhead)
  python -m nwc.demo Qwen/Qwen3-4B --save Qwen3-4B-NWC # write a compressed checkpoint
  python -m nwc.demo Parda21/Qwen3-4B-NWC --load       # load a compressed checkpoint (local dir or HF repo id)
"""
import time, argparse, torch


def main():
    ap = argparse.ArgumentParser(description="NWC demo")
    ap.add_argument("model", help="HF model name/path (BF16), or an NWC checkpoint with --load")
    ap.add_argument("--load", action="store_true", help="model is an NWC checkpoint")
    ap.add_argument("--save", default=None, help="write an NWC checkpoint to this directory")
    ap.add_argument("--native", action="store_true", help="do not compress (comparison)")
    ap.add_argument("--fp8", action="store_true", help="quantize to weight-only fp8 e4m3 first, then compress the fp8 values")
    ap.add_argument("--no-fusion", action="store_true", help="do not fuse q/k/v and gate/up")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--graph", action="store_true", help="time the decode step as a CUDA graph")
    ap.add_argument("--prompt", default="Question: What is a binary tree and its applications? Answer:")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device available; run `python -m nwc.doctor` to see what is missing")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from .checkpoint import resolve
    import os

    t0 = time.time()
    path = resolve(a.model)
    if not a.load and os.path.exists(os.path.join(path, "nwc_config.json")):
        print("NWC: this is a compressed checkpoint, loading it (as with --load)"); a.load = True
    if a.load:
        from .checkpoint import load_pretrained
        model = load_pretrained(path)
        tok = AutoTokenizer.from_pretrained(path)
    else:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
        if a.native:
            model.cuda()
        else:
            from .nwc_torch import convert, fuse
            elem = "fp8" if a.fp8 else "bf16"
            if not a.no_fusion: fuse(model, elem=elem)
            convert(model, elem=elem)
        model.eval()
    print(f"loaded in {time.time()-t0:.0f} s, VRAM in use {torch.cuda.memory_allocated()/1e9:.2f} GB")
    if a.save and not a.native:
        from .checkpoint import save_pretrained
        c = save_pretrained(model, a.save, tokenizer=tok, base_model=None if a.load else a.model)
        print(f"checkpoint written: {a.save} ({c['bytes_nwc']/1e9:.2f} GB of compressed matrices)")

    ids = tok(a.prompt, return_tensors="pt").input_ids.cuda()
    L = ids.shape[1]
    with torch.no_grad():
        model.generate(ids, max_new_tokens=8, do_sample=False)                       # warm-up
        torch.cuda.synchronize(); t0 = time.time()
        out = model.generate(ids, max_new_tokens=a.tokens, do_sample=False)
        torch.cuda.synchronize(); dt = time.time() - t0
    n = out.shape[1] - L
    print(f"generate (HF eager, greedy): {n} tokens in {dt:.2f} s = {n/dt:.1f} tokens/s (wall clock, incl. Python overhead)")
    print("text:", tok.decode(out[0, L:], skip_special_tokens=True)[:400].replace("\n", " "))

    if a.graph:
        from transformers import StaticCache
        cache = StaticCache(config=model.config, max_cache_len=L + a.tokens + 8)
        with torch.no_grad():
            o = model(input_ids=ids, past_key_values=cache, cache_position=torch.arange(L, device="cuda"), use_cache=True)
        cur = o.logits[:, -1].argmax(-1).view(1, 1).clone(); pos = torch.tensor([L], device="cuda")
        def step():
            o = model(input_ids=cur, past_key_values=cache, cache_position=pos, use_cache=True)
            pos.add_(1); cur.copy_(o.logits[:, -1].argmax(-1).view(1, 1))
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3): step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad(): step()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(a.tokens): g.replay()
        torch.cuda.synchronize(); dt = time.time() - t0
        print(f"CUDA graph: {a.tokens} tokens in {dt:.2f} s = {a.tokens/dt:.1f} tokens/s = {dt/a.tokens*1e3:.2f} ms/token")


if __name__ == "__main__":
    main()
