import numpy as np, pandas as pd
def build(path=None):
    import os
    for cand in [path,'data/real_12m_continuous_panel.csv','../data/real_12m_continuous_panel.csv']:
        if cand and os.path.exists(cand): path=cand; break
    d=pd.read_csv(path).sort_values(['asset','hour']).reset_index(drop=True)
    d['ts']=pd.Timestamp('2023-03-01')+pd.to_timedelta(d.hour,unit='h')
    out=[]
    for a,g in d.groupby('asset'):
        g=g.copy(); bad=(g.cash<=0)|(g.total_aToken<=0)|(g.debt<0)
        g.loc[bad,['cash','debt','total_aToken']]=np.nan
        S=g.total_aToken; g['S']=S; g['h']=g.cash/S                      # headroom fraction of supply
        g['dS']=S.diff(); g['dD']=g.debt.diff()
        g['y']=(-g.dS/S.shift()).clip(lower=0)                           # net supply outflow rate (withdrawals net of deposits)
        g['yraw']=-g.dS/S.shift()
        g['B']=g.dD/S.shift()                                             # net borrowing (borrow - repay) as fraction of supply
        g['dh']=g.h.diff()
        out.append(g)
    return pd.concat(out).reset_index(drop=True)
if __name__=="__main__":
    p=build(); pd.set_option('display.width',200)
    print(p.groupby('asset').agg(n=('y','count'),y_mean=('y','mean'),y_p90=('y',lambda x:x.quantile(.9)),y_p99=('y',lambda x:x.quantile(.99)),y_max=('y','max'),
        B_sd=('B','std'),B_p01=('B',lambda x:x.quantile(.01)),B_p99=('B',lambda x:x.quantile(.99)),h_min=('h','min'),h_med=('h','median'),h_max=('h','max'),nan=('h',lambda x:x.isna().sum())).to_string())
    # jumps: supply changes bigger than 5% in an hour
    big=p[(p.yraw.abs()>0.05)]; print("hours with |dS/S|>5%:",len(big)); print(big[['ts','asset','S','yraw','B','h']].head(12).to_string())
    # interest drift check: median dS/S
    print(p.groupby('asset').yraw.median().to_string())
    # correlation of y with lagged B, and h autocorr
    for a,g in p.groupby('asset'):
        print(a,"corr(y_t,B_{t-1})=%.3f corr(y_t,y_{t-1})=%.3f ac(h)=%.4f ac(B)=%.3f"%(g.y.corr(g.B.shift()),g.y.corr(g.y.shift()),g.h.corr(g.h.shift()),g.B.corr(g.B.shift())))
