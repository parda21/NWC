import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nwc.gguf import lies_gguf
from nwc import nwc_gpu
P='models/mmproj-Bonsai-27B-BF16.gguf'
f,v,m,t,s=lies_gguf(P)
x=next(k for k in t if k['name']=='mm.2.weight')
f.seek(s+x['off']); d=f.read(40*1024*1024)
p = nwc_gpu.komprimiere(d, x['dims'], 32, 8192, stride_fest=0)
assert nwc_gpu.dekomprimiere(p) == d
open('gpu_nostride.nwc','wb').write(p)
print(f"gpu_nostride.nwc  Rate {len(p)/len(d):.4f}  (mit stride war 0.6525)")
