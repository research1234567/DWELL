"""Resumable runner: python run_baselines.py FAM NEV NDEV BUDGET_SECONDS ; caches every (method,cfg,regime,seed) result in ../../out/rev2_FAM_cache.json"""
import sys, json, os, numpy as np, time
sys.path.insert(0,'.')
from baselines_sim import *
fam=sys.argv[1]; NEV=int(sys.argv[2]); NDEV=int(sys.argv[3]); BUD=float(sys.argv[4])
RG=regimes(fam)
if fam=='g2': eb.calm=RG['mild'][0]
cs=csat()
grids={'decay_thr':[dict(kind='decay_thr',c=c) for c in [.1,.2,.5,1,2,5,10]],
 'decay_lvl':[dict(kind='decay_lvl',c=c) for c in [.1,.2,.5,1,2,5]],
 'qt_thr':[dict(kind='qt_thr',eta=e) for e in [.0005,.001,.002,.005,.01,.02,.05,.1]],
 'pid_thr':[dict(kind='pid_thr',KI=k,Csat=c) for k in [.05,.2,1] for c in [cs,2.0]]}
default={'qt_thr_default':dict(kind='qt_thr',eta_frac=0.1)}
cpath=f'../../out/rev2_{fam}_cache.json'
cache=json.load(open(cpath)) if os.path.exists(cpath) else {}
t0=time.time()
def get(mname,i,rn,s):
    k=f'{mname}|{i}|{rn}|{s}'
    if k not in cache:
        if time.time()-t0>BUD: raise TimeoutError
        m=default[mname] if i is None else grids[mname][i]; c,pn=RG[rn]
        cache[k]=[float(x) for x in run(m,pn,s,calm=c)]
    return cache[k]
def save(): json.dump(cache,open(cpath,'w'))
try:
    dev={}
    for mname,g in grids.items():
        for i in range(len(g)):
            for rn in RG:
                dev[(mname,i,rn)]=np.mean([get(mname,i,rn,s)[2] for s in range(5000,5000+NDEV)])
    sel={}
    for mname,g in grids.items():
        sel[mname]={'pooled':min(range(len(g)),key=lambda i:np.mean([dev[(mname,i,rn)] for rn in RG]))}
        for rn in RG: sel[mname][rn]=min(range(len(g)),key=lambda i:dev[(mname,i,rn)])
    print('SEL',json.dumps(sel),flush=True)
    for rn in RG:
        for mname in grids:
            for b in ('pooled',rn):
                for s in range(NEV): get(mname,sel[mname][b],rn,s)
        for s in range(NEV): get('qt_thr_default',None,rn,s)
        print('eval done',rn,time.time()-t0,flush=True)
    json.dump(dict(sel=sel,dev={'|'.join(map(str,k)):v for k,v in dev.items()}),open(f'../../out/rev2_{fam}_selection.json','w'),indent=1)
    print('ALLDONE',flush=True)
except TimeoutError:
    print('budget reached; cache size',len(cache),flush=True)
save()
