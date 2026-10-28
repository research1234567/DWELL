"""rev3 experiments A,B (resumable cache out/rev3_cache.json). Run from code/rev3: python3 w1_run.py BUDGET"""
import sys,json,os,time,numpy as np;sys.path.insert(0,'..')
from core import *
import env_bench as eb
BUD=float(sys.argv[1]);cp='../../out/rev3_cache.json';cache=json.load(open(cp)) if os.path.exists(cp) else {}
caps=json.load(open('../../out/rev3_window_caps.json'));t0=time.time()
REG={'R1':(-2.0,100.0,80),'R3':(-2.0,90.0,120),'R5':(-2.0,120.0,150)}
WIN={'R1':0.5*0.0061799339510553945,'R3':0.5*0.004520406668310491,'R5':None}
def get(k,fn):
    if k not in cache:
        if time.time()-t0>BUD:raise TimeoutError
        cache[k]=[float(x) for x in fn()]
    return cache[k]
try:
    for nm,(a,b,d) in REG.items():
        p=Env(a=a,b=b,d=d);g=0.5*gt_crit(p.rho,d,gmax=0.08,n=1400)
        arms={'f02':0.02,'scalar':g}
        if WIN[nm]:arms['window']=WIN[nm]
        for an,e in arms.items():
            for s in range(6000,6030):get(f'A|{nm}|{an}|{s}',lambda:eb.run(p,None,s,fixed=e))
    for nm in ['mild','severe','vsevere']:
        a,b,d=eb.TRUTH[nm];p=Env(a=a,b=b,d=d);cap=caps[nm]['cap']
        for s in range(50):get(f'B|{nm}|window|{s}',lambda:eb.run(p,cap,s,common=True))
    print('ALLDONE',flush=True)
except TimeoutError:print('budget; cache',len(cache),flush=True)
json.dump(cache,open(cp,'w'))
