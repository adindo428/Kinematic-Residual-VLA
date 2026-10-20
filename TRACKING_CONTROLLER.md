# Tracking controller: collection, training, and evaluation

This branch preserves the Franka hover-tracking controller, its published pilot dataset recipe, the 75-epoch DynamicVLA training snapshot, and the closed-loop tracking protocol. It is a tracking adaptation using DOM assets, not the official DOM pick-and-place benchmark.

| Step | Start here |
|---|---|
| Collect expert trajectories and export the dataset | [Collector guide](third_party/Data_Collection_Dynamic_Objects/README.md) and [implementation details](third_party/Data_Collection_Dynamic_Objects/implementation.md) |
| Reproduce the 75-epoch model family | [Training guide](experiments/tracking75/TRAINING.md) and [frozen source](experiments/tracking75/source/) |
| Evaluate tracking in closed loop | [Evaluation protocol](experiments/tracking75/README.md) |

The pilot dataset is [`adi2440/dom-franka-tracking`](https://huggingface.co/datasets/adi2440/dom-franka-tracking) at revision `5c46a9f26b5f85edafa53885187cc03fa8706219`. It contains 100 accepted expert episodes; the frozen training split is 80/10/10 episodes. Source code, configurations, split definitions, small verification reports, and provenance are in this branch. Large videos, USD assets, checkpoints, and environments are obtained separately using the linked guides.

The collector runs a 200 Hz rule-based expert and records synchronized 25 Hz RGB and robot/object state. Its LeRobot v2.1 action labels are measured future tool poses, 0.2 seconds ahead, rather than controller commands. The 75-epoch policies use 20-action chunks. The current four-model comparison covers DOM-VLA, DOM-VLA+Velocity, KR-VLA-Coordinate, and KR-VLA-Coordinate+Velocity. The historical Field variant remains in the frozen source.

The training and closed-loop comparisons currently use recorded or simulator-provided object velocity. A camera-only velocity estimator is not included, so the velocity-using models should not be described as vision-only. The tracking score measures time near a moving hover target; it is not grasp success.
