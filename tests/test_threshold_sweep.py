import unittest
import numpy as np
from dreamer_agent.calibration import select_threshold,metrics
class ThresholdSweepTests(unittest.TestCase):
 def test_matches_exhaustive_with_ties(self):
  rng=np.random.default_rng(9)
  for size in [20,100,300]:
   for _ in range(8):
    y=rng.integers(0,2,size);p=np.round(.5*y+.5*rng.random(size),1)
    candidates=[]
    for t in np.unique(p):
     r=metrics(p,y,float(t))
     if r["tp"]+r["fp"]>=3 and r["precision_interval"][0]>=.5:candidates.append(r)
    expected=max(candidates,key=lambda r:(r["recall"] or 0,r["precision_interval"][0])) if candidates else None
    actual=select_threshold(p,y,precision_lower=.5,min_predictions=3)
    self.assertEqual(actual["threshold"],expected["threshold"] if expected else None)
if __name__=="__main__":unittest.main()
