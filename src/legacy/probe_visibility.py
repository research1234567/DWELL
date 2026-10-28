import numpy as np
from core import *
for b in [3,12,24,48]:
    e=Env(b=b,a=-0.5 if b<=3 else -0.5*b/3*0+(-1.0)); 
    print(f"b={b} a={e.a} q*={e.qstar:.4f} f={e.f:.2f} theta={e.theta:.2f} rho={e.rho:.2f} K={e.K:.2f} gQT_c={2/e.K:.4f}  f*gQT_c={e.f*2/e.K:.3f}")
# strong-reactivity env
e=Env(a=-1.0,b=24.0); gc=2/e.K
print("env",e.qstar,e.f,e.theta,e.rho,gc)
for r in [0.3,0.6,0.9,1.1,1.5,2.0,3.0]:
    g=r*gc; ac=[];v=[];pk=[]
    for s in range(8):
        o=run(lambda t:e,make_fixed(g),30000,s); x=o['q'][3000:]-e.qstar
        v.append(np.mean(x**2)); xc=x-x.mean(); ac.append(np.sum(xc[1:]*xc[:-1])/np.sum(xc**2))
        P=np.abs(np.fft.rfft(xc[:16384]*np.hanning(16384)))**2; fr=np.fft.rfftfreq(16384)*2
        pk.append(P[fr>0.9].sum()/P.sum())   # share of power near Nyquist (period-2)
    print(f"g/gc={r:.1f} g={g:.4f} ac1={np.mean(ac):+.3f} var={np.mean(v):.5f} nyq_power_share={np.mean(pk):.3f}  linPhi={1-g*e.K:+.3f}")
