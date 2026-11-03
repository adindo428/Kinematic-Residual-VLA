# Finger–object contact friction diagnostic

This experiment changes the **contact material** of the two Franka finger collision bodies. The zero-friction correction in the earlier gripper diagnostic referred to **joint friction**, a separate parameter. NVIDIA's [gripper rigging guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/rig_closed_loop_structures.html) describes increasing fingertip contact friction. Our update uses native PhysX material buffers and follows the body-to-shape indexing used by [Isaac Lab's material event](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/envs/mdp/events.py).

The exploratory conditions are current finger materials, static/dynamic coefficients **1.0/1.0**, and **2.0/2.0**. Joint friction, finger speed/effort, object contact material, nonfinger robot materials, restitution and combine modes remain fixed. The effective contact-pair coefficient also depends on the other object's material and combination rule; the values above are the **finger material's coefficients**, not an independently measured effective pair coefficient. Upstream currently sets object static/dynamic friction to 0.9/1.0; we retain that setting.

`friction.py` selects only `panda_leftfinger` and `panda_rightfinger`, verifies the shape mapping, changes static/dynamic columns only, reads back all robot materials, and verifies the object's materials were unchanged. It refuses requested values below the existing finger values. Observers also check the finger readback after each trial. An absent or overwritten physics change becomes an infrastructure error, not a policy failure.

## Palmetto experiment

Job **16899593**, submitted October 9, 2026: one shared L40S, three released configurations (apple, bottle, potato), two repeats per condition = 18 closed-loop trials. All conditions use the previous corrected gripper and aligned action queue, released DOM weights, unchanged latency sleep, identical diagnostic seeding rule, and official manipulation success criteria. A fresh current-material control is rerun in this allocation. The new physics conditions are not the original benchmark environment and must be labeled diagnostic.

At submission the job was pending priority; this file is not a claim of completed results. Check its current state and `verification.json` before reporting scores.

Results:
`/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_friction/20261009`

[Browse results](https://ondemand.rcd.clemson.edu/pun/sys/dashboard/files/fs//home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_friction/20261009)

Source stage:
`/scratch/aparame/Dynamic_VLA/runs/official_dom_friction_20261009`

`control/`, `contact_mu1/`, `contact_mu2/` contain trial results, physics coefficient readbacks in each trial record, per-step telemetry, client index traces, logs and videos. The aggregate is `diagnostic_summary.json` / `comparison.csv`. Archive job **16899599** waits for the evaluation, backs up verified completed arms with the existing private Hugging Face workflow, and may replace home folders with scratch symlinks. Do not assume a backup succeeded until its receipt is verified.

## Linux workstation continuation

Using the prepared public source/checkpoint and assets from [the workstation guide](../../../../workstation/README.md), select new stage/result paths:

```bash
export DIAG_STAGE="$KRVLA_WORK/contact-friction-01"
export DIAG_RESULTS="$KRVLA_WORK/results/contact-friction-01"
"$POLICY_PYTHON" workstation/diagnostic.py prepare --base "$STAGE" \
  --official-root "$RESULTS" --stage "$DIAG_STAGE" --root "$DIAG_RESULTS" \
  --finger-friction 1 2
"$POLICY_PYTHON" workstation/diagnostic.py run --stage "$DIAG_STAGE" \
  --root "$DIAG_RESULTS" --assets "$ASSETS" --isaaclab "$ISAACLAB" \
  --policy-python "$POLICY_PYTHON" --labels gripper_aligned contact_mu1 contact_mu2
```

`gripper_aligned` is the unchanged-contact control in the workstation comparison. Preparation also keeps original and gripper-only copies available, but the command above runs only the matched three friction conditions. No pre-existing source or result folder is overwritten. The target simulator remains Sim 5.1 / Lab 2.3.2, and the workstation runtime is not yet validated by these CPU checks.

The focused CPU tests verify finger-only shape selection, preservation of other materials and restitution, refusal to decrease friction, and portable staging. Actual simulator readback and learned-policy outcomes require the running evaluation.
