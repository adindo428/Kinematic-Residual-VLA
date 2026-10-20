# Recorded verification evidence

These small reports preserve the September 30, 2026 local pilot verification and Hugging Face publication results without putting datasets/videos in Git.

- `pilot-report.json`: collection quality, balance, coverage and accepted episode counts.
- `expert-smoke-suite.json`: five expert-driven Isaac Sim cases, not learned-policy results.
- `dataset-validation.json`: full export validation and generic LeRobot 0.3.3 reader check.
- `upstream-reader-validation.json`: 18 actual upstream reader samples after the isolated instruction/cache patches.
- `huggingface-upload.json`: published dataset revision, remote inventory and downloaded-sample checks, public access confirmation.

Absolute paths in historical reports describe the original machine, not prerequisites for a fresh checkout. Retrieve the dataset at the revision in the upload receipt and follow the README to produce new reports. Generated local reports stay under ignored `runs/` or `exports/` directories.

October 2 publication checks: tools-environment pytest passed **24 tests, one skipped** (`pxr` unavailable in that interpreter). New evaluator tests cover checkpoint dimensions/history, absolute/delta conversion, geodesic angle wrapping, terminal padding and exclusion of target labels from model inputs. These tests use deterministic test policies, not trained checkpoints. No learned-policy simulator run or actual checkpoint forward pass was performed.
