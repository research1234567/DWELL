"""Protocol 2 resumable runner: python run_caps.py BUDGET"""
import sys,json,os,time,numpy as np
sys.path.insert(0,'.')
from adaptive_cap_sim import *
BUD=float(sys.argv[1]); RGN=['mild','severe','vsevere','extreme']
calm=Env(a=-0.5,b=3.0,d=1); PAN={n:Env(a=a_,b=b_,d=d_) for n,(a_,b_,d_) in eb.TRUTH.items()}
CAPS=[0.003,0.006,0.012,0.025,0.05,0.2]
cp='../../out/rev2_caps_cache.json'; cache=json.load(open(cp)) if os.path.exists(cp) else {}
t0=time.time()
def get(arm,cfg,rn,s):
    k=f'{arm}|{cfg}|{rn}|{s}'
    if k not in cache:
        if time.time()-t0>BUD: raise TimeoutError
        cap=Backoff() if arm=='backoff' else float(cfg)
        cache[k]=[float(x) for x in run(PAN[rn],cap,s,calm=calm)]
    return cache[k]
try:
    DEV=range(5000,5006)
    print('stage1 backoff dev',flush=True)
    for rn in RGN:
        for s in DEV: get('backoff','-',rn,s)
    bd={rn:np.mean([get('backoff','-',rn,s)[2] for s in DEV]) for rn in RGN}; print('backoff dev pinball',bd,flush=True)
    for rn in RGN:
        for c in CAPS:
            for s in DEV: get('const',c,rn,s)
    dev={(rn,c):np.mean([get('const',c,rn,s)[2] for s in DEV]) for rn in RGN for c in CAPS}
    loro={rn:min(CAPS,key=lambda c:np.mean([dev[(r2,c)] for r2 in RGN if r2!=rn])) for rn in RGN}
    print('LORO choice',loro,'| const dev',{f'{k[0]}|{k[1]}':round(v,3) for k,v in dev.items()},flush=True)
    for rn in RGN:
        for s in range(50): get('const',loro[rn],rn,s)
        print('loro eval',rn,time.time()-t0,flush=True)
    for rn in RGN:
        for s in range(50): get('backoff','-',rn,s)
        print('backoff eval',rn,time.time()-t0,flush=True)
    json.dump(dict(loro=loro,dev={f'{k[0]}|{k[1]}':v for k,v in dev.items()}),open('../../out/rev2_caps_selection.json','w'),indent=1); print('ALLDONE')
except TimeoutError: print('budget; cache',len(cache))
json.dump(cache,open(cp,'w'))
