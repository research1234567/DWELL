import json, numpy as np, warnings; warnings.filterwarnings("ignore")
from real_replay import *
Y=series(); out={}
for a in sorted(Y):
    st={}; ex=replay(Y[a],0.0,1,common=True,exact=True,stats=st); cl=replay(Y[a],0.0,1,common=True,exact=False,zero=True)
    out[a]=dict(neg=100*st['neg']/st['n'],cov_ex=ex[0],pin_ex=ex[2],cov_cl=cl[0],pin_cl=cl[2])
    print(a,out[a],flush=True)
json.dump(out,open('out/real_exact.json','w'),indent=1)
