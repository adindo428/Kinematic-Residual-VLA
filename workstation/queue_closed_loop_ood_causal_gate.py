"""Run the same causal-gate ablation on the frozen 1 m/s OOD cases."""
from pathlib import Path
import queue_closed_loop_causal_gate as gate

gate.PARENT = gate.PARENT.parent/'closed-loop-speed-ood-20261112'
gate.ROOT = gate.PARENT.parent/'closed-loop-ood-causal-gate-20261112'
gate.runner.ROOT = gate.ROOT
_prepare = gate.prepare
_verify = gate.runner.verify

def prepare():
    existed = (gate.ROOT/'plan.json').exists()
    plan = _prepare()
    if not existed:
        assert plan['protocol']['initial_speed_mps'] == 1.0
        plan['files_sha256'][str(Path(__file__).resolve())] = gate.low.sha(__file__)
        gate.low.write(gate.ROOT/'plan.json', plan)
        request = gate.low.read(gate.ROOT/'request.json')
        request.update(next=None, initial_speed_mps=1.0)
        gate.low.write(gate.ROOT/'request.json', request)
    return plan

def verify(plan):
    result = _verify(plan)
    path = gate.ROOT/'comparison.md'
    path.write_text(path.read_text().replace('initial speed 0.10 m/s', 'initial speed 1.00 m/s')+
        '\nInference-only causal gating; same KRV epoch-3 weights as the ungated OOD run.\n')
    return result

gate.prepare = prepare
gate.runner.verify = verify

if __name__ == '__main__': gate.main()
