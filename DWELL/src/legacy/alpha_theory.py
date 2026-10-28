import numpy as np, json
from scipy.linalg import solve_discrete_lyapunov as dlyap
from core import *
def alpha_system(env,gamma,W):
    """Exact linearisation of rolling-window level-update ACI.
    state x=(a_t, u_{t-1},...,u_{t-D}), D=d+W ;  u_t = -a_t/f - rho*M_t,  M_t=(1/W) sum_{j=1..W} u_{t-j-d}
    a_{t+1}=a_t+gamma*(f u_t + theta u_{t-d}) + noise"""
    d=env.d; D=d+W; n=1+D; f,th,rho=env.f,env.theta,env.rho
    c=np.zeros(n); c[0]=-1/f
    for j in range(1,W+1): c[1+(j+d)-1]+= -rho/W          # u_{t-(j+d)} sits at position 1+(j+d)-1
    A=np.zeros((n,n))
    row=gamma*f*c.copy(); row[0]+=1
    if d==0: row+=gamma*th*c
    else: row[1+d-1]+=gamma*th                              # u_{t-d}
    A[0]=row
    for k in range(1,D): A[1+k,1+k-1]=0
    # shift: new u_{t} enters position 1 at next step, others move down
    A[1]=c
    for k in range(2,n): A[k,k-1]=1
    return A,c
def stats(env,gamma,W,var=True):
    A,c=alpha_system(env,gamma,W); sr=max(abs(np.linalg.eigvals(A)))
    if not var or sr>=1: return sr,np.inf
    B=np.zeros((A.shape[0],1)); B[0,0]=gamma*np.sqrt(env.alpha*(1-env.alpha))
    P=dlyap(A,B@B.T); return sr,float(c@P@c)
if __name__=="__main__":
    R=json.load(open('out/alpha_form.json')); out=[]
    for (b_,d,W,r,g,sim,th_qt) in R:
        e=Env(a=-0.5 if b_==3.0 else -1.0,b=b_,d=d)
        sr,v=stats(e,g,W,var=(W<=200))
        out.append((b_,d,W,r,g,sim,th_qt,v,sr))
        print(f"b={b_} d={d} W={W:5d} g/gc={r:.2f} sim={sim:.3e} QT-equiv={th_qt:.3e} window-theory={v:.3e} ratio_win={sim/v if np.isfinite(v) else float('nan'):.2f} specrad={sr:.4f}")
    json.dump(out,open('out/alpha_theory.json','w'))
    # thresholds: first gamma where spectral radius >=1 (window system) vs QT-equivalent
    for (a_,b_,d) in [(-0.5,3.0,0),(-1.0,24.0,6)]:
        e=Env(a=a_,b=b_,d=d); gtc=gt_crit(e.rho,d) if d>0 else 2/(1+e.rho)
        for W in [50,200,1000]:
            gs=np.linspace(0.02,3,150)*gtc; crit=next((g for g in gs if max(abs(np.linalg.eigvals(alpha_system(e,g,W)[0])))>=1),np.inf)
            print(f"threshold b={b_} d={d} W={W}: window-system gamma_c={crit:.4f}  vs QT-equivalent {gtc:.4f}  (ratio {crit/gtc:.2f})")
