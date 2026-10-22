# Installation and commands

## Setup

First install **Isaac Sim 5.1.0 and Isaac Lab 2.3.2 with Python 3.11**, following the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html) and selecting the [v2.3.2 release](https://github.com/isaac-sim/IsaacLab/tree/v2.3.2). Then clone this repository outside the Isaac Lab directory:

```bash
git clone https://github.com/yezheng04/unitree-g1-23dof-decoupled-control.git
cd unitree-g1-23dof-decoupled-control
```

Create the project environment from your configured Isaac Lab environment (replace `env_isaaclab` with its actual name):

```bash
conda create -n g1-23dof --clone env_isaaclab -y
conda activate g1-23dof
python -m pip install -r requirements-runtime.txt -e .
python scripts/setup/download_assets.py
```

The clone includes all project source code. The download script fetches the robot assets and **checkpoint 22000** with its metadata. Full videos are optional: use `python scripts/setup/download_assets.py --videos` or download them from the [Release](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/tag/v0.1.0-22000).

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

## Deployment

For checkpoint export, MuJoCo sim2sim and the experimental G1 EDU 23DoF hardware controller, see [Deploy](DEPLOY.md). The standalone deployment environment uses `requirements-deploy.txt` and does not require Isaac Sim.
