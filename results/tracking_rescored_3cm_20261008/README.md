Mean time within 3 cm of the moving hover target (%)

For each episode, calculate 100 times the duration with 3D EE-to-hover-target distance <=0.03 m divided by the episode duration; average episode percentages with equal weight. The entire episode, including the initial second, counts. No binary pass/fail threshold or orientation gate. A policy termination earns no tracking credit for the unexecuted remainder of its requested horizon; completion rate and observed-only coverage are reported separately.

These are the earlier saved comparisons, not results of the new nine-dimension benchmark. All 120 episodes were rescored without rerunning policy inference or changing originals.

| Model | 1× | 1.5× | 2× | Average over all 30 episodes |
|---|---:|---:|---:|---:|
| DOM-VLA | 6.02 | 4.92 | 4.21 | 5.05 |
| DOM-VLA+Velocity | 25.45 | 21.43 | 7.02 | 17.96 |
| KR-VLA-Coordinate | 63.99 | 27.76 | 7.34 | 33.03 |
| KR-VLA-Coordinate+Velocity | 56.45 | 29.03 | 10.89 | 32.12 |

All episodes completed except one Coordinate episode at 2×. The remaining requested duration gets zero tracking credit; completion is reported separately.

Time to first entry within 3 cm, capped at the requested horizon if never acquired. This is acquisition time, not task completion time. Simulation time excludes expert warmup and inference pauses. All earlier trials started within 3 cm after expert warmup, so acquisition time is zero and does not distinguish these models.
