"""From code/rev2: python3 summarize_all.py. No simulations or selection changes."""
import json,numpy as np
from pathlib import Path
out=Path('../../out'); RG=['mild','severe','vsevere','extreme'];report=[]
for fam,N,nds in [('f2',50,10),('g2',20,5)]:
 c=json.load(open(out/f'rev2_{fam}_cache.json')); selection=json.load(open(out/f'rev2_{fam}_selection.json'))['sel']; regimes=RG if fam=='f2' else RG[:3]
 report.append(f'{fam}: eval seeds0-{N-1}, dev5000-{4999+nds}, cache{len(c)}')
 for m in ['decay_thr','decay_lvl','qt_thr','pid_thr']:
  for budget in ['pooled','perreg']:
   vals=[]
   for r in regimes:
    i=selection[m]['pooled' if budget=='pooled' else r];x=np.array([c[f'{m}|{i}|{r}|{seed}'] for seed in range(N)])
    vals.append(f'{r}: pin{x[:,2].mean():.6f} se{x[:,2].std(ddof=1)/np.sqrt(N):.6f} cov_abs_reactive{x[:,1].mean():.6f}')
   report.append(f'{m} {budget}: '+ '; '.join(vals))
 for r in regimes:
  x=np.array([c[f'qt_thr_default|None|{r}|{seed}'] for seed in range(N)]);report.append(f'qt_default {r}: {x.mean(0).tolist()}')
c=json.load(open(out/'rev2_caps_cache.json')); selection=json.load(open(out/'rev2_caps_selection.json'));report.append(f'caps cache count{len(c)}');summ={}
for r in RG:
 summ[r]={}
 for arm,cfg in [('const',selection['loro'][r]),('backoff','-')]:
  x=np.array([c[f'{arm}|{cfg}|{r}|{seed}'] for seed in range(50)]);summ[r][arm]={'mean':x.mean(0).tolist(),'se':(x.std(0,ddof=1)/np.sqrt(50)).tolist()};report.append(f'{r} {arm}: '+str(summ[r][arm]))
json.dump(summ,open(out/'rev2_caps_summary.json','w'),indent=1)
(out/'rev2_regenerated_analysis.txt').write_text('\n'.join(report)+'\n');print('\n'.join(report))
