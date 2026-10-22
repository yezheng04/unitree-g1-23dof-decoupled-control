# Repository contents

| Directory | Included |
|---|---|
| source/ | One registered locomotion and arm-disturbance task, body/controller mapping, sensors, rewards, curriculum, observations, symmetry and runner |
| scripts/rsl_rl/ | Task-specific train/play |
| scripts/evaluation/ |19-case policy evaluation with trajectory capture, generic checkpoint-labeled offline replay |
| scripts/validation/ | CPU source/link/schema checks and upstream-equation/GPU integration validator |
| scripts/setup/ | Pinned29-file robot asset downloader and hash verifier |
| deploy/ | Standalone TorchScript export, MuJoCo physics, experimental HG DDS controller and CPU fault-injection tests |
| docs/ | Installation, training recipe, architecture graph, results and artifact guide |
| docs/media/ | Measured-policy preview GIF and representative frames |
| results/ | Sanitized22000 evaluation, render checks, completed25001-run marker and training provenance |
| licenses/ | Upstream license texts and source attribution |
| .github/workflows/ | CPU checks and wheel-build CI; no unverified GPU-convergence claim |

Excluded: old HOMIE/29DoF/mimic tasks, unrelated deployment stacks, native mjlab project, personal server/W&B details, virtual environments, large logs, caches and credentials. Large files are kept in the separate release bundle; robot assets are downloaded from their pinned official source.

## Packaging validation

See [PORTFOLIO_VALIDATION.json](../results/PORTFOLIO_VALIDATION.json): pinned asset download, CPU source checks, two-update policy training, checkpoint restoration and finite trained-policy inference, curriculum/equation validation, graphics replay and wheel contents passed on the tested stack. These are functionality checks, not a new trained-policy evaluation.
