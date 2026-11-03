# DeltaAI interactive stress and speed sweep

User requested worst-case interactive profiling before spending full training time. Production3381061 was held before starting. Interactive3381095 allocated gh092,four GH200s,32CPU,384GB for45minutes.

Separate frozen family domv-stress-ddp-20261014 and setup-stress-domv-20261014 preserve audited bytes, all nine approved exclusions, matched split, normalization and frozen model/training source f991015a.... No repeated audit or Isaac Sim.

Stress scenarios in order: microbatch88/workers4 (largest batch dividing global352 at4ranks),44/workers4,44/workers2,22/workers4. Each gets9minutes and the complete sweep35minutes. Separate fresh checkpoint/RNG smoke and18-update full-data processes. Full-data check keeps live validation prefetch, performs2 post-training validation batches with optimizer state and train workers still present, and measures full startup/training/validation GPU and host peaks. Require15percent headroom and finite validation metrics. Source math/sampler/optimizer unchanged. Failures recorded and excluded; only actual complete safe checkpoint/RNG/full-data/validation profiles can select production runtime. Short stress checks cannot establish stability for all epochs or every dataset example.

Comparisons to original75.46 and155.9 samples/s are historical references, not new measurements. Actual scenario receipts determine selection. Keep production held until sweep completes. Any replacement must point to this frozen family and record a confirmed submission before cancelling held3381061. Never release old3381061 with assumptions about a new family's runtime.
