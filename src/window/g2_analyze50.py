import json,numpy as np
RG=['mild','severe','vsevere']
S=json.load(open('../../out/rev2_g2_selection.json'))['sel']
cb=json.load(open('../../out/rev2_g2_cache.json'));ce=json.load(open('../../out/rev3_g2ext_cache.json'))
def stored(rn):
    M={}
    for s0 in (0,10):
        d=json.load(open(f'../../out/g2_{rn}_{s0}.json'))
        for k,v in d.items():
            if k!='truth':M.setdefault(k,[]).extend(v)
    return {k:np.array(v) for k,v in M.items()}
ARMS=['Fixed g=0.005','Fixed g=0.01','Fixed g=0.02','Fixed g=0.05','DtACI (published)','DtACI (deployed error)','EG-DtACI (scenario envelope)','EG-DtACI (universal envelope)']
R={}
for rn in RG:
    st=stored(rn);r={}
    for a in ARMS: r[a]=np.vstack([st[a],np.array([ce[f'{rn}|{a}|{s}'] for s in range(20,50)])])
    # own-error+cap only in ext; skip for 0-49 (orig stored elsewhere)
    for m in ['decay_thr','decay_lvl','qt_thr','pid_thr']:
        for b,lab in (('pooled','pooled'),(rn,'perreg')):
            r[f'{m} [{lab}]']=np.array([cb[f'{m}|{S[m][b]}|{rn}|{s}'] for s in range(50)])
    r['qt_thr [default]']=np.array([cb[f'qt_thr_default|None|{rn}|{s}'] for s in range(50)])
    R[rn]=r
def line(n,sl,idx=2):
    return ['%.3f+-%.3f'%(R[r][n][sl,idx].mean(),R[r][n][sl,idx].std(ddof=1)/np.sqrt(len(R[r][n][sl,idx]))) for r in RG]
out=[]
for lab,sl in (('seeds 0-19',slice(0,20)),('seeds 0-49',slice(0,50)),('seeds 20-49 only',slice(20,50))):
    out.append(f'== pinball x100 {lab} {RG}')
    for k in R['mild']:out.append('%-32s'%k+'  '.join('%-14s'%x for x in line(k,sl))+'  cov '+' '.join('%.2f'%R[r][k][sl,1].mean() for r in RG))
out.append('== paired diff vs EG scenario, seeds 0-49 (mean +-1.96se; neg = arm better)')
for k in R['mild']:
    if k.startswith(('EG','Fixed','DtACI')):continue
    o=[]
    for r in RG:
        d=R[r][k][:,2]-R[r]['EG-DtACI (scenario envelope)'][:,2];o.append('%+.3f+-%.3f'%(d.mean(),1.96*d.std(ddof=1)/np.sqrt(50)))
    out.append('%-32s'%k+'  '.join(o))
out.append('== paired diff: EG scenario minus Fixed 0.02 (seeds 0-49) and vs Fixed .005')
for ref in ('Fixed g=0.02','Fixed g=0.005'):
    o=[]
    for r in RG:
        d=R[r]['EG-DtACI (scenario envelope)'][:,2]-R[r][ref][:,2];o.append('%+.3f+-%.3f'%(d.mean(),1.96*d.std(ddof=1)/np.sqrt(50)))
    out.append(ref+'  '+'  '.join(o))
# orig-seed preservation check
out.append('seeds0-19 EG scenario means match rev2: '+str([round(float(R[r]['EG-DtACI (scenario envelope)'][:20,2].mean()),3) for r in RG]))
open('../../out/rev3_g2_analysis50.txt','w').write('\n'.join(out));print('\n'.join(out))
