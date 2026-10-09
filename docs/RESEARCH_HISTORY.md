> Historical recovery notes from September 2026. For current executable workflows, use [the root README](../README.md).

# Kinematic Residual VLA

Research workspace for **Kinematic Residual Flow Matching for Vision-Language-Action Policies**. The project studies dynamic manipulation, where a target object moves between observation and contact. The recovered DynamicVLA fork and paper source are included; no trained checkpoint or completed empirical result is included.

See [the project knowledge base](../KNOWLEDGE_BASE.md) for the method-to-code map, training history, recovery limits, and unresolved work. For the recovered paired training commands, start with [the recovery guide](../third_party/DynamicVLA/RECOVERY.md), not the older [third-party README](../third_party/DynamicVLA/README.md).

## Idea

A conventional flow-matching VLA learns a distribution over action chunks conditioned on image, instruction, and robot state. When the target moves, its velocity shifts the appropriate action. Merely supplying velocity as another input leaves the action generator to learn this predictable shift alongside grasp geometry and contact strategy.

Kinematic Residual Flow Matching (KRFM) separates the two:

1. Estimate target-object velocity and project it into the robot's action-chunk coordinates with a context-dependent kinematic map.
2. Subtract the projected displacement from demonstrated action chunks.
3. Train a conditional flow-matching network on these residual action targets.
4. At inference, generate a residual chunk and add the projected displacement back before execution.

For a fixed context \(c\), object velocity \(v^o\), raw target \(x_1\), and projected displacement \(d_c(v^o)\), the mean-preserving version uses

\[
\bar d_c = \mathbb E[d_c(V^o)\mid c], \qquad
\widetilde x_1 = x_1 - d_c(v^o) + \bar d_c.
\]

The residual flow learns \(\widetilde x_1\); inference reconstructs \(\widehat x_1=\widehat{\widetilde x}_1+d_c(v^o)-\bar d_c\). The draft also permits the simpler uncentered version with \(\bar d_c=0\). For joint-space actions, one proposed local projection is \(L_H J^\dagger(q)R_c v^o\), with physical-time scaling incorporated into the projection or its flow-time integral.

## Theoretical scope

The paper assumes a **local additive model** \(X_1=R_c+d_c(V^o)\), with residual action \(R_c\) conditionally independent of object velocity given context. Under that assumption, subtracting the centered velocity displacement makes the velocity-conditioned residual action laws identical. The draft proves zero velocity-induced Wasserstein variability for the residual target, versus \(\mathbb E[\|d_c(V^o)-\bar d_c\|^2\mid c]\) for the raw target, and a corresponding reduction in covariance trace when projected displacement varies. These are conditional theoretical results, not measured performance gains or guarantees outside the assumption.

The approximation can break when acceleration, contact dynamics, occlusion, joint limits, collision constraints, or velocity-dependent grasp strategies matter. Velocity-estimation bias can also shift the final action incorrectly.

## Proposed evaluation

The draft outlines conveyor grasping, dynamic handover, interception/catching, and a quasi-static control setting, particularly testing held-out object speeds. Proposed comparisons include a current-frame VLA, a velocity-conditioned flow-matching VLA without target residualization, oracle-velocity KRFM, and estimated-velocity KRFM. Proposed metrics include task success, contact-position and timing error, miss and collision rates, smoothness, latency, and raw-versus-residual target covariance. Ablations examine centering, analytic drift, projection choice, and velocity noise.

**No empirical results are reported in the recovered paper.** The experiments section is a protocol, not evidence that these baselines or ablations have been run.

## Source and recovery status

- Overleaf project: [Kinematic-Residual VLA](https://www.overleaf.com/project/6a29d86d9b1841920b3ce81a).
- Paper source: [`paper/iclr2025_conference.tex`](../paper/iclr2025_conference.tex), recovered from Overleaf. A copy also lives on Palmetto at `/home/aparame/Research/Dynamic_VLA/paper/`.
- Recovered implementation: [`third_party/DynamicVLA`](../third_party/DynamicVLA), reconstructed from upstream and surviving development-session records. The recovered source is copied here, excluding nested Git history, caches, and run artifacts.
- As of September 22, 2026, no lost trained checkpoint weights were recovered. The historical epoch-15 residual checkpoint mentioned in a prior chat is not available in this repository.

The paper is an unfinished draft: the introduction has a truncated sentence; later sections still use the earlier name “ProactiveVLA”; the experiment table and ablation list reference belief tokens and an uncertainty-aware gripper loss that are not defined by the current method; and author/template placeholders remain. The preliminary flow-matching loss also appears to use \(x_s-x_0\) where its stated linear interpolation target is \(x_1-x_0\). These discrepancies should be resolved before treating the draft as a definitive specification.

## Next steps

Recover any surviving checkpoint or run artifacts from backups, Git remotes, W&B, or other storage; then document the actual environment, commands, datasets, and results here. Implementations and empirical claims should be added only after they can be verified.
