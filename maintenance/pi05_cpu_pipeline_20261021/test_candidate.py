import sys,unittest,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/cross_vla_krfm_five_epochs'))
from candidate import state_prompt,TimedIterator
class CandidateContracts(unittest.TestCase):
 def test_prompt_exact_matches_frozen_format_and_bins(self):
  state=torch.tensor([[-1.,0.,1.,-.5,.5,.99999],[0.,0.,0.,0.,0.,0.]])
  tasks=[' pick\nobject ','place']
  padded=torch.nn.functional.pad(state,(0,32-state.shape[-1]));bins=torch.linspace(-1,1,257)[:-1]
  values=(torch.bucketize(padded,bins,right=True)-1).cpu().tolist()
  expected=[f'Task: {t.strip().replace(chr(10), " ")}, State: {" ".join(map(str,s))};\nAction: ' for t,s in zip(tasks,values)]
  self.assertEqual(state_prompt(tasks,state),expected)
 def test_timing_does_not_change_stream_or_rng(self):
  state=torch.get_rng_state().clone();it=TimedIterator(iter([1,2,3]));self.assertEqual(list(it),[1,2,3]);self.assertEqual(it.count,3);self.assertGreaterEqual(it.wait_seconds,0);self.assertTrue(torch.equal(state,torch.get_rng_state()))
if __name__=='__main__':unittest.main()
