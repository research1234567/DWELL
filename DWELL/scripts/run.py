"""Run archived scientific scripts in an isolated compatibility layout."""
from pathlib import Path
import argparse,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('group',choices=['legacy','baselines','window','transfer','population'])
p.add_argument('script',help='Python filename within the selected source group')
p.add_argument('args',nargs=argparse.REMAINDER)
a=p.parse_args()
if Path(a.script).name!=a.script or not (ROOT/'src'/a.group/a.script).is_file():p.error('Choose an existing source filename.')
w=ROOT/'work'; c=w/'conformal_feedback_revised'; c.mkdir(parents=True,exist_ok=True)
# Preserve frozen results: archived scripts only read/write copied working files.
for src,dst in [(ROOT/'src/legacy',c/'code'),(ROOT/'src/baselines',c/'code/rev2'),(ROOT/'src/window',c/'code/rev3'),(ROOT/'src/transfer',w/'transfer_results'),(ROOT/'src/population',w/'population_proof_addendum')]:
 shutil.copytree(src,dst,dirs_exist_ok=True)
if not (c/'out').exists():shutil.copytree(ROOT/'results/benchmarks',c/'out')
(c/'figs').mkdir(exist_ok=True);(c/'data').mkdir(exist_ok=True)
external=ROOT/'data/external/real_12m_continuous_panel.csv'
if external.exists():shutil.copy2(external,c/'data'/external.name)
for sub,src in [('transfer_results','transfer'),('population_proof_addendum','population')]:
 for f in (ROOT/'results'/src).iterdir():
  dst=w/sub/f.name
  if not dst.exists():shutil.copy2(f,dst)
for f in (ROOT/'data/synthetic/transfer').glob('*.npz'):
 dst=w/'transfer_results'/f.name
 if not dst.exists():shutil.copy2(f,dst)
folders={'legacy':c,'baselines':c/'code/rev2','window':c/'code/rev3','transfer':w/'transfer_results','population':w/'population_proof_addendum'}
script=(c/'code'/a.script) if a.group=='legacy' else folders[a.group]/a.script
sys.exit(subprocess.call([sys.executable,str(script),*a.args],cwd=folders[a.group]))
