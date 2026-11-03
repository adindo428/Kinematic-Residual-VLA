CR, DA and LS: completed tracking evaluation results

Success (%) is the equal-weight mean of each episode’s percentage of time within ≤3 cm of the moving hover target, including the first second. These are three completed dimensions, not the final nine-dimension benchmark.

Each model has 200 episodes per dimension; all 2,400 rollouts completed and verified with scoring version 4. CR and DA last 10 seconds; LS lasts 30 seconds.

| Model | CR (%) | DA (%) | LS (%) | Mean over these three (%) |
|---|---:|---:|---:|---:|
| DOM-VLA | 10.94 | 10.40 | 8.84 | 10.06 |
| DOM-VLA+Velocity | 53.66 | 45.73 | 46.81 | 48.74 |
| KR-VLA-Coordinate | 51.38 | 50.52 | 49.71 | 50.54 |
| KR-VLA-Coordinate+Velocity | 52.87 | 55.18 | 52.10 | 53.38 |

Mean per-episode hover position RMSE (mm; lower is better):

| Model | CR | DA | LS |
|---|---:|---:|---:|
| DOM-VLA | 66.25 | 68.10 | 68.16 |
| DOM-VLA+Velocity | 34.87 | 38.20 | 36.62 |
| KR-VLA-Coordinate | 34.93 | 36.73 | 34.45 |
| KR-VLA-Coordinate+Velocity | 33.41 | 34.16 | 33.17 |

Mean object speed: CR 6.06 cm/s; DA 6.18 cm/s; LS 5.76 cm/s. All models use matched episodes. Completion rate is 100% for all cells; completion means the episode ran to its horizon, independently of tracking coverage. LS is continuous tracking, not manipulation sequencing. These are tracking adaptations in a single simulated room.

Source: Palmetto evaluation array 16830026, full shards 000–029. The overall nine-dimension table remains pending.
