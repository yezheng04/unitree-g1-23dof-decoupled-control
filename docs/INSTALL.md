# Installation and commands

## Tested stack

| Component | Version |
|---|---|
| OS / GPU | Ubuntu 24.04 / RTX 5080 16 GB |
| Python | 3.11 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | v2.3.2 (`37ddf626871758333d6ed89cf64ad702aef127d0`) |
| PyTorch | 2.7.0+cu128 |
| RSL-RL | 3.1.2 |
| W&B | 0.24.2 |
| Offline replay only | MuJoCo 3.5.0, imageio/FFmpeg, Pillow |

Install Isaac Lab and Isaac Sim using the instructions in the [pinned Isaac Lab release](https://github.com/isaac-sim/IsaacLab/tree/v2.3.2). Isaac Sim is an external prerequisite, not bundled or installed automatically by this repository. Activate that environment before all simulator commands. Verify `nvidia-smi` and `torch.cuda.is_available()` first.

This package preserves the Python import name `unitree_rl_lab` for upstream compatibility. Use a dedicated environment, or uninstall another editable `unitree_rl_lab` package before installing this one; do not unknowingly import a different checkout.

```bash
python -m pip install -r requirements-runtime.txt
python -m pip install -e .
python scripts/setup/fetch_robot_assets.py
python scripts/setup/fetch_robot_assets.py --verify-only
python scripts/validation/check_release.py
python scripts/list_envs.py
```

The downloader fetches 29 files (~25 MB) from the pinned official Unitree ROS commit, checks every SHA256, and writes to ignored `assets/g1/`. Sources, hashes and license are recorded in `robot_assets.json` and `licenses/unitree_ros.txt`. No symlink to a particular server is needed. USD conversion is cached under ignored `resources/usd_cache/`.

Optional path overrides: `G1_ASSET_DIR`, `G1_23DOF_URDF`, `G1_23DOF_MJCF`, `G1_USD_CACHE`; see `.env.example` (it is a template, not automatically loaded). Prefer editable installation. A non-editable wheel installation additionally requires `G1_PROJECT_ROOT` pointing at the repository/asset workspace. Custom URDF/MJCF pairs must preserve matching geometry and joint semantics; replay checks reject inconsistent FK.

## Training and playback

```bash
# Installation smoke only: 64 environments, two updates.
python scripts/rsl_rl/train_duet.py --headless --num_envs 64 --max_iterations 2 --offline --run_name smoke
# Full recipe. Configure W&B once with `wandb login`, or add --offline.
python scripts/rsl_rl/train_duet.py --headless
# Resume: budget means TOTAL completed updates, not additional updates.
python scripts/rsl_rl/train_duet.py --headless --checkpoint artifacts/checkpoints/model_22000.pt --max_iterations 25001
# Interactive viewer; use --headless for a display-free inference check.
python scripts/rsl_rl/play_duet.py --checkpoint artifacts/checkpoints/model_22000.pt --vx 0.5 --base_height 0.78 --arms
```

Use the task-specific train/play entry points above: the runner preserves entropy/curriculum state and validates the 71/96/13 contract. Weights from native MuJoCo/mjlab are not load-compatible.

```bash
python scripts/evaluation/evaluate_duet.py --headless --checkpoint artifacts/checkpoints/model_22000.pt --output outputs/eval_22000
# Optional graphics-only replay; can run in a separate Python 3.11 environment.
python -m pip install -e '.[render]'
python scripts/evaluation/render_duet_replay.py --input outputs/eval_22000
```

Evaluation defaults: seed42, 19 cases × 16 samples, 18 seconds each. Video labels are derived from the selected checkpoint. `--smoke` on the renderer only produces a short preview, not a full video suite.

## Upstream equation / resume validation

For this optional check, clone the upstream training project listed in [Acknowledgments](../README.md#acknowledgments) at the revision recorded in [source provenance](../THIRD_PARTY_NOTICES.md), then run:

```bash
python scripts/validation/check_duet_port.py --headless --checkpoint artifacts/checkpoints/model_22000.pt --upstream /absolute/path/to/upstream-repo --output outputs/validation
```

Read the PASS marker, not only the process exit code: Isaac Sim shutdown can obscure Python exceptions. CPU CI checks syntax, links, result schema and wheel construction; actual physics/inference checks require the tested GPU stack.
