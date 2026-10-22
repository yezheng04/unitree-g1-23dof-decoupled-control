# G1 23DoF Locomotion & Arm-Disturbance Control

An Isaac Lab / PhysX implementation of flat-ground locomotion and decoupled arm-disturbance training for the Unitree G1 23DoF robot.

[中文说明](README.zh-CN.md) · [Installation](docs/INSTALL.md) · [Architecture](docs/ARCHITECTURE.md)

![Recorded IsaacLab policy replay at checkpoint 22000](docs/media/demo.gif)

The preview shows recorded **PhysX trajectories**, rendered offline without MuJoCo simulation steps. It uses checkpoint **22000**. The evaluated model and all three full videos are available in the [checkpoint 22000 release](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/tag/v0.1.0-22000); see [download and extraction instructions](docs/ARTIFACTS.md).

## What this project demonstrates

- Adapting locomotion and arm-disturbance training to Isaac Lab's manager-based environments.
- Explicit joint-order mapping for a 23DoF embodiment, including robot limits, PD gains, contact sensors and virtual foot reference points.
- Decoupled control: a 13-action policy controls legs and waist yaw; an independent trajectory generator drives the 10 arm joints.
- Staged velocity, squat-height, arm-motion and push curricula; symmetry augmentation and loss; resumable curriculum and entropy schedules.
- A reproducible evaluation pipeline with fixed commands, first-fall masking, explicit failure counts and verified trajectory replay.

This is a **learning and engineering portfolio** focused on simulation adaptation, training and evaluation for the G1 23DoF robot. Implementation details and physical differences are documented in [Architecture](docs/ARCHITECTURE.md).

## Included task

| Task ID | Policy action | Role |
|---|---:|---|
| `Unitree-G1-23dof-Duet-Velocity` | 13 | Legs + waist locomotion under 10-joint arm disturbances and height commands |

Observations: **71 actor / 96 critic**, single-frame, no HIM encoder. Actor and critic use MLPs `[512, 256, 128]` with ELU and observation normalization.

## Installation

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

## Training

```bash
# Quick installation check
python scripts/rsl_rl/train_duet.py --headless --num_envs 64 --max_iterations 2 --offline --run_name smoke
# Full training
wandb login
python scripts/rsl_rl/train_duet.py --headless
```

## Measured results

The full run completed **25001 PPO updates** on one RTX 5080. The published quantitative evaluation currently covers **checkpoint 22000**, not the final checkpoint.

| Test | Command | Measured mean |
|---|---:|---:|
| Forward | 0.50 m/s | 0.533 m/s |
| Forward with arm motion | 0.50 m/s | 0.496 m/s |
| Backward | −0.30 m/s | −0.147 m/s |
| Turn in place, left / right | ±0.30 rad/s | +0.209 / −0.201 rad/s |
| Squat height | 0.38 m | 0.382 m |
| Deep squat height | 0.24 m | 0.270 m |

## Code map

```text
source/unitree_rl_lab/unitree_rl_lab/
  paths.py                       # Portable assets and USD cache
  tasks/locomotion/robots/g1/
    common/                      # Robot, contacts and simulation support; no registered task
    duet_velocity/               # 13+10 control, rewards, curricula and runner
scripts/
  setup/                         # Pinned robot-asset download with SHA256 checks
  rsl_rl/                        # Train and play entry points
  evaluation/                    # Fixed-command evaluation and offline replay
  validation/                    # Source checks and GPU integration validation
deploy/                          # Standalone policy export, MuJoCo and experimental HG DDS control
docs/                            # Recipe, design, reproducibility and limitations
results/                         # Small, sanitized evaluation and completion evidence
licenses/                        # Retained third-party license texts
```

## Deploy

Standalone deployment tools support **MuJoCo sim2sim** and an **experimental G1 EDU 23DoF HG DDS controller**. Export the checkpoint with its observation normalization before deployment. Use Python 3.11; see [environment setup and full deployment guide](docs/DEPLOY.md).

```bash
python -m pip install -r requirements-deploy.txt
python -m deploy.export_policy --checkpoint artifacts/checkpoints/model_22000.pt
# Actual MuJoCo physics; omit --headless to open the viewer.
python -m deploy.sim2sim --headless --vx 0.5 --duration 18
# Hardware controller's default mode: offline inference, no DDS connection.
python -m deploy.sim2real
```

For hardware, install `deploy/requirements-hardware.txt`, confirm the network interface and IMU frame, and begin with a read-only check:

```bash
python -m deploy.sim2real --read-only --interface YOUR_ROBOT_NIC --imu-frame pelvis --duration 10
# After the setup and operator checks in docs/DEPLOY.md:
python -m deploy.sim2real --execute --interface YOUR_ROBOT_NIC --imu-frame pelvis
```

Hold L1 and press Start to move to the default pose; keep L1 held and press A to start the policy. B/Select or releasing L1 requests a latched damping stop. Motion-service takeover requires an explicit `--release-mode`.

## Acknowledgments

Thanks to [DUET](https://github.com/bae-air-lab/DUET) and [HOMIE](https://github.com/InternRobotics/OpenHomie) for their open-source work.
