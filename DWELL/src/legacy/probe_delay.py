import numpy as np
from core import *
def peak(x):
    n=min(len(x),16384); xc=x[:n]-x[:n].mean(); P=np.abs(np.fft.rfft(xc*np.hanning(n)))**2; fr=np.fft.rfftfreq(n)*2
    i=np.argmax(P[1:])+1; return fr[i], P[i]/P.sum()
for (a,b,d) in [(-1.0,24.0,6),(-1.0,24.0,12),(-1.0,24.0,24)]:
    e0=Env(a=a,b=b); 
    gt=gt_crit(e0.rho,d); gc=gt/e0.f
    print(f"\nb={b} d={d} rho={e0.rho:.2f} f={e0.f:.2f} gt_c={gt:.4f} gQT_c={gc:.4f} f*gc={gt:.3f}  d=0 gt_c={2/(1+e0.rho):.3f}")
    e=Env(a=a,b=b,d=d)
    for r in [0.3,0.7,1.0,1.3,2.0,3.0]:
        g=r*gc; v=[];pw=[];fq=[]
        for s in range(6):
            o=run(lambda t:e,make_fixed(g),40000,s); x=o['q'][4000:]-e.qstar
            v.append(np.mean(x**2)); f_,p_=peak(x); fq.append(f_); pw.append(p_)
        print(f"  g/gc={r:.1f} gQT={g:.4f} var={np.mean(v):.5f} peak_freq/pi={np.mean(fq):.3f} peak_power_share={np.mean(pw):.3f}")
