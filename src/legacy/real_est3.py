import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from real_est2 import *
import real_est2; cfg=sys.argv[1]; B=int(sys.argv[2]); MARGFLAG=(len(sys.argv)>3 and sys.argv[3]=='marg'); real_est2.MARG=MARGFLAG; p=build(); X=prep(p); regs=[f'SB{lo}_{hi}' for lo,hi in WIN]
X['_p10']=X.groupby('asset').h1.transform(lambda v:v.quantile(0.10)); X['_p25']=X.groupby('asset').h1.transform(lambda v:v.quantile(0.25))
C={'low10_90':(X[X.h1<=X._p10],0.9),'low10_99':(X[X.h1<=X._p10],0.99),'low25_90':(X[X.h1<=X._p25],0.9),'all90':(X,0.9),'all99':(X,0.99),'scarce90':(X[X.h1<0.05],0.9),'scarce99':(X[X.h1<0.05],0.99),'svb90':(X[(X.hour>=216)&(X.hour<384)],0.9),'svb99':(X[(X.hour>=216)&(X.hour<384)],0.99),
   'calm90':(X[~((X.hour>=216)&(X.hour<384))],0.9)}
Xs,tau=C[cfg]; print(cfg,"n=",len(Xs),"assets",Xs.asset.nunique(),flush=True)
b=qr(Xs,tau,regs)[regs].values; bs=boot(Xs,tau,regs,B=B)
# total long-run response: sum of the three window coefficients (a persistent 1-unit headroom shock raises each window sum by ~ window-length*..)
# phi_total approx = coefficient of the longest window (steady state) ; we report all and a conservative total = sum of positive-part upper bounds
lo,hi=np.percentile(bs,[2.5,97.5],axis=0); tot=bs.sum(1)
res=dict(n=len(Xs),tau=tau,est=b.tolist(),lo=lo.tolist(),hi=hi.tolist(),sum_est=float(b.sum()),sum_lo=float(np.percentile(tot,2.5)),sum_hi=float(np.percentile(tot,97.5)),sum_p975=float(np.percentile(tot,97.5)),B=len(bs))
json.dump(res,open(f'out/real_{cfg}{"_marg" if MARGFLAG else ""}.json','w'),indent=1); print(json.dumps(res))
