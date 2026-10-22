# Evaluation and remaining work

## Evidence status

- Full training run: 25001 updates, 600024 control steps per environment, 4096 environments, 2457698304 total transitions. Training completion and finite final weights: [training_25001.json](../results/training_25001.json).
- Published evaluation: checkpoint **22000**, whose metadata records **22001 completed updates** (zero-based filename). The final checkpoint 25000 has not been evaluated with this protocol.
- Checkpoint SHA256: `715412d1a71e2def33814e95d4900683b3780bf2e57bddfd2099b6cb35b22f93`.
- 19 conditions × 16 samples; 18 seconds each; seed42; deterministic mean actions. 0–3s stand at 0.78m, step to test commands at 3s, measure from 6s. Only samples before the first fall count.
- Retain friction and hand/torso mass variation. Disable pushes, observation noise, COM shifts and encoder bias. The arm-disturbance cases use ratio1 and no mild mixture; others use default PD arm targets.
- All 304 episodes survived. This is a short test, not evidence of unrestricted robustness. Different conditions have different random samples; the comparison is not a paired causal ablation.

| Case | Target vx/vy/yaw | Mean vx/vy/yaw | Target / measured height (m) | Survival |
|---|---|---|---|---|
| stand | +0.00 / +0.00 / +0.00 | +0.001 / -0.000 / +0.004 | 0.78 / 0.776 | 16/16 |
| forward05 | +0.50 / +0.00 / +0.00 | +0.533 / -0.001 / -0.007 | 0.78 / 0.780 | 16/16 |
| backward03 | -0.30 / +0.00 / +0.00 | -0.147 / +0.001 / -0.001 | 0.78 / 0.779 | 16/16 |
| left02 | +0.00 / +0.20 / +0.00 | +0.089 / +0.161 / +0.014 | 0.78 / 0.779 | 16/16 |
| right02 | +0.00 / -0.20 / +0.00 | +0.102 / -0.167 / -0.016 | 0.78 / 0.781 | 16/16 |
| yaw_left03 | +0.00 / +0.00 / +0.30 | +0.090 / -0.047 / +0.209 | 0.78 / 0.780 | 16/16 |
| yaw_right03 | +0.00 / +0.00 / -0.30 | +0.089 / +0.053 / -0.201 | 0.78 / 0.780 | 16/16 |
| curve_left | +0.50 / +0.00 / +0.30 | +0.538 / -0.001 / +0.197 | 0.78 / 0.780 | 16/16 |
| curve_right | +0.50 / +0.00 / -0.30 | +0.538 / +0.005 / -0.201 | 0.78 / 0.780 | 16/16 |
| squat60 | +0.00 / +0.00 / +0.00 | -0.005 / -0.000 / +0.000 | 0.60 / 0.608 | 16/16 |
| squat38 | +0.00 / +0.00 / +0.00 | -0.013 / -0.001 / -0.013 | 0.38 / 0.382 | 16/16 |
| squat24 | +0.00 / +0.00 / +0.00 | -0.016 / -0.002 / -0.021 | 0.24 / 0.270 | 16/16 |
| walk_low60 | +0.50 / +0.00 / +0.00 | +0.428 / -0.010 / -0.006 | 0.60 / 0.606 | 16/16 |
| stand_arms | +0.00 / +0.00 / +0.00 | +0.002 / -0.001 / +0.003 | 0.78 / 0.777 | 16/16 |
| forward_arms | +0.50 / +0.00 / +0.00 | +0.496 / -0.002 / +0.004 | 0.78 / 0.782 | 16/16 |
| yaw_left_arms | +0.00 / +0.00 / +0.30 | +0.058 / -0.022 / +0.191 | 0.78 / 0.780 | 16/16 |
| yaw_right_arms | +0.00 / +0.00 / -0.30 | +0.055 / +0.025 / -0.185 | 0.78 / 0.780 | 16/16 |
| squat38_arms | +0.00 / +0.00 / +0.00 | -0.013 / -0.001 / -0.013 | 0.38 / 0.381 | 16/16 |
| squat24_arms | +0.00 / +0.00 / +0.00 | -0.014 / -0.002 / -0.012 | 0.24 / 0.260 | 16/16 |

## Limitations and next experiments

1. Turning reaches only 67–70% of ±0.3rad/s and translates about 0.92–0.96m over the 12s measurement window. Full arm disturbance gives +0.191/−0.185rad/s with about0.63m translation.
2. Lateral commands induce +0.089/+0.102m/s forward velocity. Backward −0.3m/s reaches only −0.147m/s. Report the unwanted axes and transient error rather than just the commanded-axis average.
3. Deep squat0.24m settles around0.270m with default arms and0.260m with arm motion. Recheck soft limits, posture penalties and geometric reachability with a slow height sweep before adjusting weights.
4. Normal walking has near-equal left/right contact duty0.638/0.640 and foot-point height P95 about4.49/4.45cm. Mean stance-foot horizontal speed is0.026m/s; low-height walking rises to0.054m/s. These metrics do not prove zero slip or toe clearance.
5. Arm disturbance walking tracks0.5m/s at0.496m/s, but arm joint target MAE is about0.136rad. Locomotion robustness and accurate arm tracking are separate criteria.
6. Next compare the final model under the same protocol; then isolate command coverage versus reward tradeoffs for pure turning/lateral motion. No reward changes were made during this evaluation.

## Replay integrity

Videos use recorded PhysX poses, with MuJoCo used only for graphics. No `mj_step` is called. The displayed body frames match recorded frames with maximum position error9.22e-6m. Each clip uses the first environment in its group, not a selected best rollout. Every18s clip begins with3s at zero velocity/0.78m.

- Locomotion126s: forward, left, right, left turn, right turn, left curve, right curve.
- Height72s:0.60m,0.38m,0.24m, then walking at0.60m.
- Arms54s: forward, left turn, deep squat.
- [Raw aggregate metrics](../results/duet_22000/evaluation.json), [render verification](../results/duet_22000/RENDER_PASS.json), [artifacts](ARTIFACTS.md).

The PhysX evaluation above is separate from the preliminary MuJoCo deployment checks in [DEPLOY.md](DEPLOY.md). A standalone experimental HG DDS controller is now included; no hardware experiments or complete teleoperation pipeline have been validated.
