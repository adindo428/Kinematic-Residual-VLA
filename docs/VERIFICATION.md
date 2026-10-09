# Verification and known limits

The publication adds local launchers and documentation while preserving distinct tracking and DOM training implementations. No checkpoint weights, assets or completed rollouts are embedded in Git.

From the repository root, CPU tests for official manipulation accounting and the portable launcher's command/overwrite behavior, plus current tracking scoring:

```bash
python -m pytest -q evaluation/official_dom/tests \
  experiments/tracking75/source/tests/test_benchmark.py
```

**34 tests passed** when preparing this publication: 11 official/local-launcher tests and 23 tracking-scoring tests. They reject missing/extra/duplicated trials, no-action infrastructure outcomes, inconsistent times, corrupt telemetry grids, and incorrect partial-episode aggregation. They do not establish successful GPU simulation.

Training's focused phase-gate checks use the DOM source import path:

```bash
PYTHONPATH=third_party/DynamicVLA python -m pytest -q \
  third_party/DynamicVLA/tests/test_dom_phase_gate.py \
  third_party/DynamicVLA/tests/test_dom_stopping.py
```

**14 phase-gate and 6 early-stopping tests passed** locally. Broader training tests need the policy environment, including Torch, Safetensors, PyArrow and decoder dependencies. An attempted broader training check in this Mac's base Python could not collect because those dependencies/import paths were absent; it is not reported as passed. The DOM100 documentation retains prior training validation evidence and its hardware/runtime scope.

Python compilation and launcher `--help` are checked. Cluster shell scripts are syntax checked, not submitted by this publication. Additional tracking trainer/model tests are included in `source/tests`; use that source directory as the working directory so imports match the tracking fork.

The local workstation pilot is still required: this Mac cannot execute NVIDIA CUDA/Isaac simulation, and the RTX 3080's combined policy/camera memory and runtime behavior have not been tested. Official evaluation records actual GPU identity, separate policy inference VRAM peaks, and sampled total-device usage during the pilot. Underlying simulator assets and runtime compatibility are validated by the real pilot rather than inferred from CPU tests.

The official benchmark keeps upstream source/test/checkpoint revisions and streaming execution. Its Table 1 differences are descriptive; a different GPU or latency may affect outcomes. The tracking stress suite uses synchronous simulation paused during inference, one room and declared tracking proxies. Its occupancy scores are not manipulation success percentages and cannot be compared directly to the paper's Table 1.
