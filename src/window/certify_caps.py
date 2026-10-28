import sys,json,numpy as np;sys.path.insert(0,'..')
from window_boundary import *
S=json.load(open('../../out/rev3_envelope_scan_double.json'));out={}
for nm in ['mild','severe','vsevere']:
    rows=S[nm]['rows'];ok=[r for r in rows if r['stable_start'] and r['eta_first']];m=min(ok,key=lambda r:r['eta_first'])
    r=first_loss(S[nm]['rho_bar'],m['d'],500);out[nm]=dict(rho_bar=S[nm]['rho_bar'],d_min=m['d'],double_eta=m['eta_first'],mp_status=r['status'],mp_eta=r['eta_c'],cap=0.5*r['eta_c'] if r['eta_c'] else None,scalar_cap=0.5*min(x['scalar'] for x in rows))
    print(nm,out[nm],flush=True)
out['extreme']=dict(status='UNDEFINED: %d of %d delays in 1..%d have no stable start'%(sum(not r['stable_start'] for r in S['extreme']['rows']),len(S['extreme']['rows']),S['extreme']['d_bar']),first_no_start=[r['d'] for r in S['extreme']['rows'] if not r['stable_start']][:3])
json.dump(out,open('../../out/rev3_window_caps.json','w'),indent=1)
