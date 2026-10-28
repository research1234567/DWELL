"""Fast deterministic checks; no full experiment reruns."""
from pathlib import Path
import hashlib,json,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/legacy'))
sys.path.insert(0,str(ROOT/'src/window'))
from core import Env,gt_crit,companion
from window_boundary import schur_mp
class SmokeTests(unittest.TestCase):
 def test_source_manifest(self):
  for row in json.loads((ROOT/'provenance/source_manifest.json').read_text())['files']:
   self.assertEqual(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(),row['sha256'],row['path'])
 def test_scalar_boundary(self):
  e=Env(a=-1,b=24,d=6);g=gt_crit(e.rho,e.d)
  self.assertLess(max(abs(np.linalg.eigvals(companion(.99*g,e.rho,e.d)))),1)
  self.assertGreater(max(abs(np.linalg.eigvals(companion(1.01*g,e.rho,e.d)))),1)
 def test_window_counterexample(self):
  self.assertFalse(schur_mp(.2,.9,0,1))
 def test_transfer_arrays(self):
  files=list((ROOT/'data/synthetic/transfer').glob('*.npz'));self.assertEqual(len(files),6)
  for p in files:
   with np.load(p,allow_pickle=False) as z:
    self.assertIn('mse',z.files);self.assertEqual(z['mse'].shape,(7,5,64));self.assertTrue(np.isfinite(z['mse']).all())
if __name__=='__main__':unittest.main()
