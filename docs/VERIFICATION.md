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

## Linux RTX PRO workstation preparation

The October 9 workstation update adds CUDA 12.8 setup scripts, explicit simulator-environment selection, a VS Code workspace, policy/simulator probes, and a portable three-arm grasp diagnostic. Run:

```bash
python -m pytest -q workstation/tests evaluation/official_dom/tests
```

**17 tests passed**: six workstation checks plus the eleven existing official/local checks. New checks cover preservation of the pinned baseline while creating diagnostic copies, refusal to overwrite runs, source/config drift rejection, durable preflight failures, path handling with spaces, and explicit simulator-environment selection. The shell launchers passed syntax checks; new Python and the adapted diagnostic scripts compiled. GPU probes and Linux package installers have not been executed on the target RTX PRO machine. The simulator probe is an actual-startup check to run there, not a result already obtained on this Mac.

The subsequent contact-friction update passed **20 combined workstation/official CPU tests**. Added cases verify that contact updates affect only mapped finger collision shapes, preserve restitution and other robot materials, reject decreasing friction, and stage the optional sweep without modifying the pinned baseline. The native PhysX checks and VLA outcomes are pending evaluation job 16899593; CPU passes do not establish grasp improvement.
