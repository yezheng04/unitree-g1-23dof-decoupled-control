# Sources, attribution and licenses

This repository is a derivative engineering project. Original copyright and SPDX headers are retained. The root [Apache-2.0 license](LICENSE) is inherited from Unitree RL Lab; BSD-3-Clause notices on Isaac Lab-derived portions remain applicable. See the accompanying license texts for each component.

| Upstream | Pinned revision | Role | License text |
|---|---|---|---|
| [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) | `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` | Package, manager/task and RSL-RL integration scaffolding | [LICENSE](LICENSE) |
| [Unitree RL mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) | `1425b15f73bd4095f0df53709d7c389c3eb9e790` | Robot/controller conventions and adapted physical support | [license](licenses/unitree_rl_mjlab.txt) |
| [DUET](https://github.com/bae-air-lab/DUET) | `439c21bd2817feedd519475e8301195b95cb6fcb` | Arm trajectories, reward/curriculum/symmetry design | [license](licenses/duet.txt) |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | `37ddf626871758333d6ed89cf64ad702aef127d0` / v2.3.2 | Simulation framework and adapted scripts | [license](licenses/isaaclab.txt) |
| [Unitree ROS](https://github.com/unitreerobotics/unitree_ros) | `5994d4faef0a9cadd3287f8de0199a67eeb2a259` | Downloaded URDF, MJCF and meshes | [license](licenses/unitree_ros.txt) |

## Adaptations in this repository

- The upstream training recipe was adapted to IsaacLab/PhysX and G1 Rev1.0, including explicit joint ordering and controller mapping.
- Contact slots/sites were adapted to PhysX filtered contact sensors and virtual foot reference points; actual mass-weighted CoM and payload semantics were implemented.
- The upstream 13/10 control split, 22 reward terms, mirror semantics and curriculum stages were retained, with an RSL-RL3 runner bridge for entropy and resume state.
- Fixed-command evaluation, metadata, finite-value checks and FK-verified offline replay were added.
- Publication cleanup removes machine-specific paths, unrelated tasks and operational history; adds pinned asset retrieval, packaging, documentation and source CI.

This work does not claim authorship of upstream algorithms, software or robot models. No official endorsement is implied. Native MuJoCo/mjlab repositories, Isaac Sim binaries, private accounts and hardware deployment code are not redistributed here.
