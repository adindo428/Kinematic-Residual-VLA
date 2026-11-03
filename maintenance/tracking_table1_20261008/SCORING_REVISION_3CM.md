Scoring version 4

For each episode, calculate 100 times the duration with 3D EE-to-hover-target distance <=0.03 m divided by the episode duration; average episode percentages with equal weight. The entire episode, including the initial second, counts. No binary pass/fail threshold or orientation gate. A policy termination earns no tracking credit for the unexecuted remainder of its requested horizon; completion rate and observed-only coverage are reported separately.

Time to first entry within 3 cm, capped at the requested horizon if never acquired. This is acquisition time, not task completion time. Simulation time excludes expert warmup and inference pauses.

Deployed to the active Palmetto evaluation: full array 16830026, summary 16830027. No trajectories, observation conditions, checkpoints or inference were changed. Old scoring source/protocols and per-rollout metric histories are retained. The verifier rescales old live-worker outputs from telemetry and the publisher rejects mixed scoring versions.

Validation: 23 boundary/aggregation/telemetry tests pass locally and on Palmetto; 13 pilot shards / 52 rollouts independently verified using v4. The 120 saved comparison trials were rescored. A synthetic 7200-record report verifies equal episode weights and separate completion reporting.
