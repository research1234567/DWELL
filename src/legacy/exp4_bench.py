import json, sys, numpy as np
from core import *
from smaci import SMACI
T=6400; BLK=800; NS=int(sys.argv[1]) if len(sys.argv)>1 else 12
calm=Env(a=-0.5,b=3.0,d=1); panic=Env(a=-1.0,b=24.0,d=6)
envf=lambda t: panic if ((t-1)//BLK)%2==1 else calm
isp=np.array([((t-1)//BLK)%2==1 for t in range(1,T+1)])
GC={id(calm):gt_crit(calm.rho,calm.d),id(panic):gt_crit(panic.rho,panic.d)}
print("true gt_c calm %.3f panic %.3f ; QT: calm %.3f panic %.3f"%(GC[id(calm)],GC[id(panic)],GC[id(calm)]/calm.f,GC[id(panic)]/panic.f))
def oracle(c=0.5):
    def f_(q,y,miss,t,st):
        e=envf(t); g=min(1.0,c*GC[id(e)])/e.f; st['g']=g; return q+g*(miss-ALPHA)
    return f_
def dtaci(gams=(0.005,0.01,0.02,0.05,0.1,0.2),eta=60.,share=1e-3):
    st0=dict(w=np.ones(len(gams))/len(gams),qs=None)
    def f_(q,y,miss,t,st):
        if st0['qs'] is None: st0['qs']=np.full(len(gams),q)
        qs=st0['qs']; loss=(y-qs)*((1-ALPHA)-(y<qs))
        w=st0['w']*np.exp(-eta*loss); w/=w.sum(); w=(1-share)*w+share/len(gams); st0['w']=w
        qs=np.clip(qs+np.array(gams)*((y>qs).astype(float)-ALPHA),0,1); st0['qs']=qs
        st['g']=float(w@np.array(gams)); return float(w@qs)
    return f_
methods={}
for g in [0.005,0.01,0.02,0.05,0.1,0.3]: methods[f"Fixed ACI g={g}"]=lambda g=g:(make_fixed(g),0.0)
methods["Decaying g0=0.5"]=lambda:(make_decay(0.5),0.0)
methods["DtACI-QT (6 experts)"]=lambda:(dtaci(),0.0)
def capped(base_fn,c=0.5):
    inner={}
    def f_(q,y,miss,t,st):
        e=envf(t); cap=c*GC[id(e)]/e.f
        if 'b' not in inner: inner['b']=base_fn()
        qn=inner['b'](q,y,miss,t,st); g=st.get('g',np.nan)
        step=qn-q
        if np.isfinite(g) and g>cap: qn=q+step*cap/g
        st['g']=min(g,cap) if np.isfinite(g) else cap; return qn
    return f_
methods["Fixed g=0.05 + stability cap"]=lambda:(capped(lambda:(lambda q,y,m,t,st:(st.__setitem__('g',0.05),q+0.05*(m-ALPHA))[1])),0.0)
methods["DtACI-QT + stability cap"]=lambda:(capped(lambda:dtaci()),0.0)
def idcap(base_fn):
    sm=SMACI(); inner={}
    def f_(q,y,miss,t,st):
        if 'b' not in inner: inner['b']=base_fn()
        sm(q,y,miss,t,st)                       # updates identification, sets st['dither']
        cap=sm.gt/sm.f
        qn=inner['b'](q,y,miss,t,st); g=st.get('g',np.nan); step=qn-q
        if np.isfinite(g) and g>cap: qn=q+step*cap/g
        st['g']=min(g,cap) if np.isfinite(g) else cap; return qn
    return f_
methods["DtACI-QT + identified cap (ours)"]=lambda:(idcap(lambda:dtaci()),0.015)
methods["Oracle-scheduled (c=0.5)"]=lambda:(oracle(),0.0)
methods["SM-ACI (ours, identified)"]=lambda:(SMACI(),0.015)
R={}
for name,mk in methods.items():
    rows=[]
    for s in range(NS):
        ctrl,dith=mk(); o=run(envf,ctrl,T,100+s,dither=dith)
        cov=o['cov']; pin=o['pin']; err=(o['q']-o['qstar'])**2
        rows.append([abs(cov.mean()-.9)*100,abs(cov[isp].mean()-.9)*100,abs(cov[~isp].mean()-.9)*100,pin.mean()*100,err[~isp].mean()*1e4,err[isp].mean()*1e4,np.nanmean(o['g'][~isp]) if not np.all(np.isnan(o['g'])) else np.nan,np.nanmean(o['g'][isp]) if not np.all(np.isnan(o['g'])) else np.nan])
    r=np.array(rows); R[name]=dict(mean=np.nanmean(r,0).tolist(),se=(np.nanstd(r,0,ddof=1)/np.sqrt(NS)).tolist())
    m=R[name]['mean']; print(f"{name:28s} cov|dev| all {m[0]:.2f} panic {m[1]:.2f} calm {m[2]:.2f} pp | pinball {m[3]:.3f} | MSE calm {m[4]:.1f} panic {m[5]:.1f} e-4 | mean gamma calm {m[6]:.3f} panic {m[7]:.3f}",flush=True)
json.dump(R,open('out/exp4.json','w'),indent=1)
