# Model and video artifacts

The evaluated **checkpoint 22000** and all three full demonstration videos are published in the [GitHub Release](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/tag/v0.1.0-22000). The training code is included in the repository and the release's Source code archives; start with [train_duet.py](../scripts/rsl_rl/train_duet.py) and the [training recipe](TRAINING_RECIPE.md).

## Downloads

| File | Content |
|---|---|
| [Complete artifact bundle](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/download/v0.1.0-22000/g1-23dof-checkpoint-22000.tar.gz) | Model, contract, three videos, manifest and license texts |
| [model_22000.pt](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/download/v0.1.0-22000/model_22000.pt) | Evaluated PyTorch checkpoint |
| [model_22000.contract.json](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/download/v0.1.0-22000/model_22000.contract.json) | Joint order, observation/action dimensions and control metadata |
| [Locomotion video](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/download/v0.1.0-22000/duet_22000_locomotion.mp4) | Walking and turning, 126 seconds |
| [Height video](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/download/v0.1.0-22000/duet_22000_height.mp4) | Squatting and low-height walking, 72 seconds |
| [Arm-disturbance video](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/download/v0.1.0-22000/duet_22000_arms.mp4) | Locomotion and squatting with arm motion, 54 seconds |
| [manifest_22000.json](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/download/v0.1.0-22000/manifest_22000.json) | Original artifact paths, SHA256 hashes and sizes |
| [SHA256SUMS](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/download/v0.1.0-22000/SHA256SUMS) | Checksums for the bundle and individual downloads |

## Extract and use

Download the complete bundle into the repository root, then extract it:

```bash
tar -xzf g1-23dof-checkpoint-22000.tar.gz
```

The bundle restores the paths used by the training, playback and evaluation commands:

```text
artifacts/
  checkpoints/
    model_22000.pt
    model_22000.contract.json
  videos/
    duet_22000_locomotion.mp4
    duet_22000_height.mp4
    duet_22000_arms.mp4
  manifest.json
```

License texts and third-party notices are also included. If downloading individual files, place the model and contract together under `artifacts/checkpoints/`, and videos under `artifacts/videos/`.

After installing the [tested environment](INSTALL.md) and fetching robot assets, play the evaluated policy with:

```bash
python scripts/rsl_rl/play_duet.py --checkpoint artifacts/checkpoints/model_22000.pt --vx 0.5 --base_height 0.78 --arms
```

## Provenance

The model SHA256 is `715412d1a71e2def33814e95d4900683b3780bf2e57bddfd2099b6cb35b22f93`. Its bytes match the evaluated checkpoint. The contract uses a portable URDF path; robot hashes and control/observation metadata are unchanged.

Checkpoint filenames use zero-based iteration labels: model 22000 has 22001 completed PPO updates. This release contains only the evaluated 22000 model; the final 25000 checkpoint is not included. Evaluation protocol and limitations are documented in [RESULTS.md](RESULTS.md).

The preview GIF is tracked in Git. Checkpoints and full MP4s are distributed as Release attachments; training logs, replay arrays and USD caches are excluded.
