import json,glob,numpy as np
C=json.load(open('../../out/rev3_cache.json'));out=[]
def arr(prefix,n,rng):return np.array([C[f'{prefix}|{s}'] for s in rng])
def pct(B,A):
    d=100*(B[:,2]-A[:,2])/A[:,2];return d.mean(),1.96*d.std(ddof=1)/np.sqrt(len(d))
out.append('A: fixed-step controllers, seeds 6000-6029; cols pinball(x100) | %change vs fixed.02 (+/-1.96se) | %change vs scalar-half | reactive cov dev pp | reactive MSE x1e4')
for nm in ['R1','R3','R5']:
    A=arr(f'A|{nm}|f02',30,range(6000,6030));S=arr(f'A|{nm}|scalar',30,range(6000,6030))
    for an,X in [('fixed .02',A),('scalar-half',S)]+([('window-half',arr(f'A|{nm}|window',30,range(6000,6030)))] if f'A|{nm}|window|6000' in C else []):
        m,c=pct(X,A);m2,c2=pct(X,S)
        out.append(f'{nm} {an:12s} pin {X[:,2].mean():.3f} | vs.02 {m:+.1f}+/-{c:.1f}% | vs scalar {m2:+.1f}+/-{c2:.1f}% | cov {X[:,1].mean():.2f} | mse {X[:,4].mean():.0f}')
out.append('B: EG-DtACI with window cap vs stored EG scalar cap and fixed .02 (seeds 0-49), pinball x100, reactive cov dev pp')
for nm in ['mild','severe','vsevere']:
    W=arr(f'B|{nm}|window',50,range(50))
    st=[json.load(open(f'../../out/f2_{nm}_{s}.json')) for s in range(0,50,10)]
    def get(k):return np.array([r for d in st for r in d[k]])
    sc=get('EG-DtACI (scenario envelope 1.5x)');f2=get('Fixed g=0.02');un=get('EG-DtACI (universal envelope)')
    m,c=pct(W,sc);m2,c2=pct(W,f2)
    out.append(f'{nm}: window-cap EG {W[:,2].mean():.3f} (se {W[:,2].std(ddof=1)/50**.5:.3f}) | scalar-cap EG {sc[:,2].mean():.3f} | universal {un[:,2].mean():.3f} | fixed.02 {f2[:,2].mean():.3f} | window vs scalar {m:+.2f}+/-{c:.2f}% | vs fixed.02 {m2:+.1f}+/-{c2:.1f}% | cov dev {W[:,1].mean():.3f} vs scalar {sc[:,1].mean():.3f} | mse {W[:,4].mean():.0f} vs {sc[:,4].mean():.0f}')
open('../../out/rev3_AB_analysis.txt','w').write('\n'.join(out));print('\n'.join(out))
