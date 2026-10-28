"""Experiment C part: original g2 guard/fixed/uncapped arms + own-error scenario cap for seeds 20-49 (extension; seeds 0-19 untouched). Resumable: python3 g2_ext_guard.py BUDGET"""
import sys,json,os,time,numpy as np,warnings;warnings.filterwarnings("ignore");sys.path.insert(0,'..')
from core import *
import env_bench as eb
BUD=float(sys.argv[1]);t0=time.time();cp='../../out/rev3_g2ext_cache.json';cache=json.load(open(cp)) if os.path.exists(cp) else {}
eb.calm=EnvT(qstar=0.30,rho=0.64,d=1)
TR={'mild':(0.10,1.85,6),'severe':(0.10,2.40,12),'vsevere':(0.10,2.08,24)}
try:
  for name,(q0,rho,d) in TR.items():
    panic=EnvT(qstar=q0,rho=rho,d=d);dbar=int(np.ceil(1.5*d));capEG=eb.robust_cap(1.5*panic.rho,dbar);capU=eb.robust_cap(3.0,24)
    arms={f'Fixed g={g}':(lambda s,g=g:eb.run(panic,None,s,fixed=g)) for g in [0.005,0.01,0.02,0.05]}
    arms['DtACI (published)']=lambda s:eb.run(panic,np.inf,s,common=False)
    arms['DtACI (deployed error)']=lambda s:eb.run(panic,np.inf,s,common=True)
    arms['EG-DtACI (scenario envelope)']=lambda s:eb.run(panic,capEG,s,common=True)
    arms['EG-DtACI (universal envelope)']=lambda s:eb.run(panic,capU,s,common=True)
    arms['own-error + scenario cap']=lambda s:eb.run(panic,capEG,s,common=False)
    for an,fn in arms.items():
      for s in range(20,50):
        k=f'{name}|{an}|{s}'
        if k not in cache:
          if time.time()-t0>BUD:raise TimeoutError
          cache[k]=[float(x) for x in fn(s)]
  print('ALLDONE')
except TimeoutError:print('budget; cache',len(cache))
json.dump(cache,open(cp,'w'))
