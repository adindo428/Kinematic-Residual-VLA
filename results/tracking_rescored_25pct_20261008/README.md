Earlier comparison rescored with the accepted 25% success criterion.

Complete the full horizon and keep 3D EE-to-hover-target distance <=0.02 m for at least 25% of the entire episode, including the initial second. Intervals need not be consecutive. Orientation is not a success gate.

These are saved 10-trial comparisons at each speed, not results of the new nine-condition evaluation. No simulation was rerun.

| Method | 1x success | 1.5x success | 2x success |
|---|---:|---:|---:|
| DOM-VLA | 0% | 0% | 0% |
| DOM-VLA+Velocity | 10% | 0% | 0% |
| KR-VLA-Coordinate | 40% | 0% | 0% |
| KR-VLA-Coordinate+Velocity | 50% | 10% | 0% |

Time to accumulate at least 25% of the requested episode duration within 2 cm. Failed/terminated trials receive their full requested horizon; successful-only time is also reported. Simulation time, excluding expert warmup and inference pauses.
