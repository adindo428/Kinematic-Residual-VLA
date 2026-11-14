"""Native resident worker with compatibility for the released DOM policy."""
from pathlib import Path

_source = Path(__file__).with_name('resident_table1_worker.py').read_text()
_old = "        dummy.update(policy.extract_residual_batch_fields(dummy))"
assert _source.count(_old) == 1
_source = _source.replace(_old, "        if hasattr(policy, 'extract_residual_batch_fields'): dummy.update(policy.extract_residual_batch_fields(dummy))")
_source = _source.replace("cfg.residual_phase_gate_mode=='causal_rules'", "getattr(cfg, 'residual_phase_gate_mode', 'constant')=='causal_rules'")
exec(compile(_source, str(Path(__file__).with_name('resident_table1_worker.py')), 'exec'))
