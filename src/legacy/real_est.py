import numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings("ignore")
from statsmodels.regression.quantile_regression import QuantReg
from real_panel import build
TAU=0.9
def design(p,K=24,P=6,assets=None):
    rows=[]
    for a,g in p.groupby('asset'):
        if assets and a not in assets: continue
        g=g.sort_values('hour').reset_index(drop=True); X=pd.DataFrame({'hour':g.hour,'asset':a,'y':g.y})
        for k in range(1,K+1): X[f'B{k}']=g.B.shift(k)
        for k in range(1,P+1): X[f'y{k}']=g.y.shift(k)
        X['vol24']=g.y.rolling(24).mean().shift(1)
        X['hod']=g.ts.dt.hour; X['dow']=g.ts.dt.dayofweek; X['ts']=g.ts
        rows.append(X)
    X=pd.concat(rows).dropna().reset_index(drop=True); return X
def fit(X,K=24,P=6,tau=TAU,fe_asset=True):
    cols=[f'B{k}' for k in range(1,K+1)]+[f'y{k}' for k in range(1,P+1)]+['vol24']
    Z=X[cols].copy()
    D=pd.get_dummies(X.hod,prefix='h',drop_first=True).astype(float); W=pd.get_dummies(X.dow,prefix='d',drop_first=True).astype(float)
    parts=[Z,D,W]
    if fe_asset and X.asset.nunique()>1: parts.append(pd.get_dummies(X.asset,drop_first=True).astype(float))
    M=pd.concat(parts,axis=1); M.insert(0,'const',1.0)
    # standardise B columns (scale 1e-2) for numerical stability, keep track
    r=QuantReg(X.y.values,M.values.astype(float)).fit(q=tau,max_iter=1000,p_tol=1e-6)
    names=list(M.columns); b=pd.Series(r.params,index=names); return b
if __name__=="__main__":
    p=build(); K=24
    for a in [None,'USDC','DAI','USDT','WETH','WBTC']:
        X=design(p,K=K,assets=[a] if a else None); b=fit(X,K=K)
        phi=np.array([b[f'B{k}'] for k in range(1,K+1)])
        print(f"{a or 'pooled':7s} n={len(X)} phi_k (k=1,2,3,4,6,8,12,16,20,24): "+" ".join(f"{phi[k-1]:+.3f}" for k in [1,2,3,4,6,8,12,16,20,24])+f" | mean(k>=13)={phi[12:].mean():+.3f} sum={phi.sum():+.2f}",flush=True)
