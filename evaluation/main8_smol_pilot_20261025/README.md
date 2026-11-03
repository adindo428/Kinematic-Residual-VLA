# Completed SmolVLA closed loop integration pilot

Job18158208 requests one L40S,8 CPUs,64GiB and at most two hours. Nine released
DOM configurations, one per dimension and one trial each, are unchanged against
the test archive. This is an integration diagnostic, not a full success-rate
comparison or exact original-paper reproduction. Simulator5.1/Lab2.3.2 and its
known gripper/physics reproduction limitations are retained. No physics fix is
silently applied. The original simulator observer records trial status/time.

The model is the completed20k/10epoch SmolVLA baseline checkpoint. All saved
parameters load strictly; checkpoint SHA, preparation/source hashes, identity,
completion/update/epoch markers are checked. A decoded held-out physical20x7
prediction must pass on the actual GPU before simulator trials. Training source,
outputs, other jobs and the other clusters are untouched; weights stay on Palmetto.

The adapter preserves single current observation, native preprocessing and saved
normalizers, two cameras,20-action queue, physical delta Euler pose/gripper,
and applies chunk-start pose exactly once through the pinned RPC converter.
Four CPU contracts passed for state/image conversion, dummy target exclusion,
queue order, and episode reset. GPU/simulator behavior remains pending allocation.
KR is intentionally excluded until current simulator velocity has been verified;
zero velocity is only a shape placeholder for this non-KR policy and never feeds
its neural prefix. No readiness weights, future labels or phase gate are used.

No full-suite or other model evaluation is submitted by this pilot. Successful
or unsuccessful manipulation is distinct from infrastructure integrity. Every
trial must be accounted for, including client acknowledgements and valid simulator
termination. A failure/no-action outcome is not silently scored as model success.

Submission intent/result and actual job ID are recorded in submission.json. Never
rerun stage.py against a recorded or uncertain submission. Preserve failed results
and use a fresh frozen attempt for repairs. No checkpoint is stored on the Mac.
