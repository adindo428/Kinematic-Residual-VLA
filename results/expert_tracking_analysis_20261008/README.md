Expert hover tracking reference

100 original accepted training-collection episodes, 10 seconds each.
Mean full-episode coverage within 2 cm: 99.972%. Lowest episode: 97.2%.
99 episodes have 100% sampled coverage; all 10 held-out expert episodes have 100%.
Mean post-first-second coverage: 99.968889%.
Pooled position RMSE: 3.020735 mm at 25 Hz; original 200 Hz collection report: 3.022992 mm.

Computed from current measured observation-tool positions and the matched hover target. Action labels (future poses), expert warmup, and continuation frames were excluded. The fixed target offset was checked across all original test trajectories, and all 10 expert/evaluation initial positions agree to 0.1 micrometer tolerance.

Coverage uses the 25 Hz stored observations. Raw 200 Hz expert traces are not present in the cache. These observations do not measure expert performance under the nine newly generated stress conditions.

Recommended expert-level criterion for discussion: full completion and at least 95% whole-episode time within 2 cm. A 50% criterion would be substantially more lenient than the demonstrated expert reference. The user subsequently selected at least 25% whole-episode coverage for this exploratory evaluation. That criterion is now deployed as scoring version 3.
