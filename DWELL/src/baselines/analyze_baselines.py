import json,numpy as np
RG=['mild','severe','vsevere','extreme']
c=json.load(open('../../out/rev2_f2_cache.json')); S=json.load(open('../../out/rev2_f2_selection.json'))['sel']
def stored(name):
    M={}
    for s0 in (0,10,20,30,40):
        d=json.load(open(f'../../out/f2_{name}_{s0}.json'))
        for k,v in d.items():
            if k!='truth': M.setdefault(k,[]).extend(v)
    return {k:np.array(v) for k,v in M.items()}
rows={}
for rn in RG:
    st=stored(rn); r={}
    for k in ['Fixed g=0.02','Fixed g=0.005','DtACI (uncapped, own error)','EG-DtACI (scenario envelope 1.5x)','EG-DtACI (universal envelope)']: r[k]=st[k]
    for m in ['decay_thr','decay_lvl','qt_thr','pid_thr']:
        for b,lab in (('pooled','pooled'),(rn,'perreg')):
            r[f'{m} [{lab}]']=np.array([c[f'{m}|{S[m][b]}|{rn}|{s}'] for s in range(50)])
    r['qt_thr [default 0.1*Bt]']=np.array([c[f'qt_thr_default|None|{rn}|{s}'] for s in range(50)])
    rows[rn]=r
keys=list(rows['mild'].keys())
print('%-36s'%'pinball x100 (mean+-se)',*['%14s'%r for r in RG],sep='')
for k in keys:
    print('%-36s'%k,*['%8.3f+-%.3f'%(rows[r][k][:,2].mean(),rows[r][k][:,2].std(ddof=1)/np.sqrt(50)) for r in RG],sep=' ')
print('\nreactive-phase cov dev (pp) / panic MSE x1e4')
for k in keys:
    print('%-36s'%k,*['%5.2f/%6.1f'%(rows[r][k][:,1].mean(),rows[r][k][:,4].mean()) for r in RG],sep=' ')
print('\npaired diff vs EG scenario (pinball, mean+-1.96se; negative=baseline better)')
for k in keys:
    if k.startswith('EG') or k.startswith('Fixed') or k.startswith('DtACI'): continue
    out=[]
    for r in RG:
        d=rows[r][k][:,2]-rows[r]['EG-DtACI (scenario envelope 1.5x)'][:,2]; out.append('%+.3f+-%.3f'%(d.mean(),1.96*d.std(ddof=1)/np.sqrt(50)))
    print('%-36s'%k,*out,sep=' ')
print('\nselected cfgs',json.dumps(S))
json.dump({r:{k:[v[:,2].mean(),v[:,2].std(ddof=1)/np.sqrt(50),v[:,1].mean(),v[:,4].mean()] for k,v in rows[r].items()} for r in RG},open('../../out/rev2_f2_summary.json','w'),indent=1)
