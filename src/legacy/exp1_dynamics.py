import json, numpy as np
from core import *
R={}
e=Env(); q0,K,gc,c1,h2,h3=flip_normal_form(e)
R.update(qstar=q0,f=e.f,theta=e.theta,K=K,gc=gc,c1=c1,rho=e.rho,h2=h2,h3=h3)
# (a) amplitude: numerical vs normal form
rows=[]
for g in [0.76,0.78,0.80,0.85,0.90]:
    o=skeleton_orbit(e,g); amp=(o.max()-o.min())/2; mu=g-gc
    th=np.sqrt(4*mu/(gc*c1)); rows.append((g,amp,th,abs(amp-th)/amp))
R['amp_table']=rows
# Lyapunov at a few points
R['lyap']={str(g):lyapunov(e,g) for g in [0.6,0.7,0.74,0.76,0.80,0.9,1.0]}
# nonlinear stability AT the bifurcation point (c1>0): converges slowly (cubic)
o=skeleton_orbit(e,gc,n_burn=200000,n_keep=2); R['at_gc_dist']=float(abs(o[0]-q0))
# (b) delay: dominant crossing frequency for d=1..6 (rho = calibrated)
dl={}
for d in range(0,9):
    g=gt_crit(e.rho,d); A=companion(g*1.0005,e.rho,d); ev=np.linalg.eigvals(A); lam=ev[np.argmax(abs(ev))]
    dl[d]=dict(gt=g,gQT=g/e.f,angle_over_pi=float(abs(np.angle(lam))/np.pi))
R['delay_calibrated']=dl
# paper's NS claim at d=1 : closed form 1/theta ; complexity cond f<3theta
R['d1_ns_closed']=1/e.theta; R['d1_cond_f_lt_3theta']=bool(e.f<3*e.theta)
# check by simulating the nonlinear d=2 skeleton just above threshold: dominant frequency
def delayed_skeleton(env,g,d,T=40000,q0=None):
    q=np.full(T+d+1,env.qstar+1e-3 if q0 is None else q0)
    for t in range(d,T+d):
        q[t+1]=q[t]+g*((1-env.alpha)-env.F(q[t],q[t-d]))
    return q[d+1:]
def peak_freq(x):
    x=x[-8192:]-x[-8192:].mean(); P=np.abs(np.fft.rfft(x*np.hanning(len(x))))**2; fr=np.fft.rfftfreq(len(x)); return fr[np.argmax(P)]*2  # in units of pi
pk={}
for d,gm in [(1,0.97),(2,0.76),(3,0.80),(4,0.76)]:
    x=delayed_skeleton(e,gm,d); pk[d]=dict(g=gm,peak_over_pi=float(peak_freq(x)),amp=float((x[-4000:].max()-x[-4000:].min())/2))
R['delay_sim_peak']=pk
# (c) stochastic chain: variance & lag-1 autocorr vs gamma (d=0), 40 seeds x 20000 steps
res=[]
for g in [0.02,0.05,0.08,0.2,0.4,0.6,0.74,0.9,1.2]:
    v=[];ac=[]
    for s in range(12):
        o=run(lambda t:e,make_fixed(g),20000,s); x=o['q'][2000:]-q0
        v.append(np.mean(x**2)); xc=x-x.mean(); ac.append(np.sum(xc[1:]*xc[:-1])/np.sum(xc**2))
    th=g*0.09/(2*K-g*K**2) if g<gc else np.nan
    res.append((g,float(np.mean(v)),float(th),float(np.mean(ac)),float(1-g*K)))
R['stoch']=res
json.dump(R,open('out/exp1.json','w'),indent=1,default=float)
for k,v in R.items(): print(k,v if not isinstance(v,(list,dict)) else '')
for r in R['amp_table']: print('amp',r)
print(R['lyap']); print('at_gc',R['at_gc_dist'])
for d,v in dl.items(): print('d',d,v)
print(pk)
for r in res: print('g=%.2f var=%.5f lin_theory=%s ac1=%.3f phi_lin=%.3f'%(r[0],r[1],'%.5f'%r[2] if r[2]==r[2] else '  nan ',r[3],r[4]))
