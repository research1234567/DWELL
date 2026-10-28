"""Pre-registered field-experiment design: randomise the deployed headroom by iid perturbations u_t ~ N(0, s^2) (fractions of supply) for H hours;
estimate rho = -d Q_0.9(y)/d headroom by quantile regression of y on window sums of u (lags 1-3, 4-12, 13-24) with the same controls as the observational
estimate.  Null runs (rho=0) give the estimator's sampling spread -> minimum detectable rho (80% power, 5% test).  Outcomes: real USDC series (true marginals);
reps vary the randomisation draw.  Optional injected effect for an alternative-power check:  y' = y*exp(-(rho/Q0) * u_{t-d})."""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from statsmodels.regression.quantile_regression import QuantReg
from real_panel import build
WIN=[(1,3),(4,12),(13,24)]; P=6; TAU=0.9
def load(asset='USDC'):
    p=build(); g=p[p.asset==asset].sort_values('hour'); return g.y.fillna(0).values, g.ts.dt.hour.values
def fit(y,u,hod):
    n=len(y); cols={}
    for lo,hi in WIN: cols[f'U{lo}_{hi}']=sum(pd.Series(u).shift(k) for k in range(lo,hi+1)).values
    for k in range(1,P+1): cols[f'y{k}']=pd.Series(y).shift(k).values
    cols['vol']=pd.Series(y).rolling(24).mean().shift(1).values
    X=pd.DataFrame(cols); X['hod']=hod; X['y']=y; X=X.dropna()
    M=X[[c for c in X.columns if c.startswith('U')]].copy(); M.insert(0,'const',1.0)      # marginal estimand: randomisation => no outcome controls needed
    r=QuantReg(X.y.values,M.values.astype(float)).fit(q=TAU,max_iter=600,p_tol=1e-6)
    idx=[list(M.columns).index(f'U{lo}_{hi}') for lo,hi in WIN]; wts=np.array([hi-lo+1 for lo,hi in WIN],float); b=r.params[idx]; C=r.cov_params()[np.ix_(idx,idx)]
    return -float(wts@b), float(np.sqrt(wts@C@wts))      # length-weighted: total response = sum_k rho_k = sum_w |w|*gamma_w
if __name__=="__main__":
    s=float(sys.argv[1]); reps=int(sys.argv[2]); rho_inj=float(sys.argv[3]) if len(sys.argv)>3 else 0.0; d_true=3
    y_all,hod_all=load(); Q0=float(np.quantile(y_all,TAU)); R={'s':s,'Q0':Q0,'rho_inj':rho_inj,'rows':[]}
    for H in [1000,2000,4000,8760]:
        est=[];se=[]
        for rep in range(reps):
            rng=np.random.default_rng(100*rep+int(H/100)); u=s*rng.standard_normal(H); y=y_all[:H].copy()
            if rho_inj>0: y=y*np.exp(np.clip(-(rho_inj/Q0)*np.r_[np.zeros(d_true),u[:-d_true]],-5,5))
            e,sd=fit(y,u,hod_all[:H]); est.append(e); se.append(sd)
        est=np.array(est); se=np.array(se); sdhat=est.std(ddof=1)
        R['rows'].append(dict(H=H,mean_est=float(est.mean()),sd_est=float(sdhat),mean_se=float(se.mean()),mde=float(2.8*sdhat),reject=float(np.mean(np.abs(est/se)>1.96))))
        print(f"s={s} rho_inj={rho_inj} H={H}: mean rho_hat {est.mean():+.4f} | sd {sdhat:.4f} | mean model se {se.mean():.4f} | MDE(80%,5%) {2.8*sdhat:.4f} | rejection rate {np.mean(np.abs(est/se)>1.96):.2f}",flush=True)
    json.dump(R,open(f'out/field_s{s}_r{rho_inj}.json','w'))
