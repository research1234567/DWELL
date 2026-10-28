import numpy as np, pandas as pd, warnings, sys, time, json; warnings.filterwarnings("ignore")
from statsmodels.regression.quantile_regression import QuantReg
from real_panel import build
WIN=[(1,3),(4,12),(13,24)]; P=6
def prep(p):
    rows=[]
    for a,g in p.groupby('asset'):
        g=g.sort_values('hour').reset_index(drop=True); X=pd.DataFrame({'hour':g.hour,'asset':a,'y':g.y,'h1':g.h.shift(1)})
        for lo,hi in WIN: X[f'SB{lo}_{hi}']=sum(g.B.shift(k) for k in range(lo,hi+1))
        for k in range(1,5): X[f'lead{k}']=g.B.shift(-k)          # placebo leads
        for k in range(1,P+1): X[f'y{k}']=g.y.shift(k)
        X['vol24']=g.y.rolling(24).mean().shift(1); X['hod']=g.ts.dt.hour; X['dow']=g.ts.dt.dayofweek; X['week']=g.hour//168
        X['dh']=g.h.diff(); X['B0']=g.B; rows.append(X)
    return pd.concat(rows).dropna().reset_index(drop=True)
MARG=False
def qr(X,tau,regs):
    if MARG:
        Z=X[regs].copy(); parts=[Z]
    else:
        Z=X[regs+[f'y{k}' for k in range(1,P+1)]+['vol24']].copy()
        parts=[Z,pd.get_dummies(X.hod,prefix='h',drop_first=True).astype(float),pd.get_dummies(X.dow,prefix='d',drop_first=True).astype(float)]
    if X.asset.nunique()>1: parts.append(pd.get_dummies(X.asset,drop_first=True).astype(float))
    M=pd.concat(parts,axis=1); M.insert(0,'const',1.0)
    r=QuantReg(X.y.values,M.values.astype(float)).fit(q=tau,max_iter=800,p_tol=1e-6); return pd.Series(r.params,index=M.columns)
def boot(X,tau,regs,B=40,seed=0):
    rng=np.random.default_rng(seed); wk=X.week.unique(); out=[]
    groups={w:X[X.week==w] for w in wk}
    for b in range(B):
        ch=rng.choice(wk,len(wk),replace=True); Xb=pd.concat([groups[w] for w in ch]).reset_index(drop=True)
        try: out.append(qr(Xb,tau,regs)[regs].values)
        except Exception: pass
    return np.array(out)
if __name__=="__main__":
    p=build(); X=prep(p); print("n",len(X))
    # first stage & placebo
    fs=np.polyfit(X.B0,X.dh,1)[0]; print("first-stage slope d(headroom)/B (OLS) = %.3f"%fs)
    regs=[f'SB{lo}_{hi}' for lo,hi in WIN]
    t0=time.time(); b=qr(X,0.9,regs); print("fit time %.1fs"%(time.time()-t0)); print("tau=.9 pooled",b[regs].round(4).to_dict())
    bl=qr(X,0.9,regs+[f'lead{k}' for k in range(1,5)]); print("placebo leads",bl[[f'lead{k}' for k in range(1,5)]].round(4).to_dict())
