# SmolVLA closed loop pilot R2

Fresh isolated retry of failed evaluation18158208. Its failure before simulator
startup was missing libnppicc.so.12 from the batch LD_LIBRARY_PATH. The existing
CUDA12.3 lib64 path used by training is now included; no package, weights,
normalizer, recipe, training source or simulator physics was changed.

Actual CPU native TorchCodec import and two-camera validation decode passed.
Four adapter contracts and shell/scheduler test-only checks passed. Actual
GPU completed-checkpoint loading, finite physical inference and nine-trial
simulator verification remain mandatory in the job. No simulator success
result exists yet. The pilot requests one L40S,8 CPUs,64GiB and at most two
hours. It reads the completed SmolVLA baseline10-epoch checkpoint only.

Job18363872 was immediately recorded in submission.json. Do not blindly rerun
stage.py or create another evaluation while this job is pending/active.
All prior failure logs are preserved in previous-failure.json and original
remote directory. No checkpoint or dataset bytes are stored on the Mac.
