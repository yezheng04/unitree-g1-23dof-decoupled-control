# G1 23自由度：IsaacLab 行走与上肢扰动控制

这是一个机器人强化学习实践项目：基于 IsaacLab/PhysX，为 G1 23DoF Rev1.0 模型适配上下肢解耦训练，并进行行走、下蹲和上肢扰动下的控制验证。

![22000轮模型回放](docs/media/demo.gif)

[下载 22000 轮模型与三段完整视频](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/tag/v0.1.0-22000) · [下载与解压说明](docs/ARTIFACTS.md)

## 工程内容

- 仅注册一个行走与上肢扰动控制任务：13维腿/腰策略加10维外部手臂轨迹；机器人和接触适配保留在不注册任务的common模块。
- 明确的关节顺序、动作缩放、PD/armature、脚部参考点和接触传感器适配。
- 速度、高度、上肢运动和推扰课程；PPO镜像增强、镜像损失和熵调度。
- 保存/恢复课程步数、checkpoint元数据和训练源码快照。
- 19个固定命令场景的评估脚本，以及由实际PhysX记录驱动的视频回放。

本仓库记录学习过程中的仿真适配、训练、评测和工程实现，具体实现与仿真差异见[架构说明](docs/ARCHITECTURE.md)。

## 使用入口

训练代码已包含在仓库中：[训练入口](scripts/rsl_rl/train_duet.py)、[任务环境与 PPO 实现](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/duet_velocity/)、[机器人与仿真适配](source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/common/)。

先按[安装说明](docs/INSTALL.md)配置IsaacLab 2.3.2 / IsaacSim 5.1 / Python 3.11环境，然后：

```bash
python scripts/setup/fetch_robot_assets.py
python -m pip install -e .
python scripts/rsl_rl/train_duet.py --headless --num_envs 64 --max_iterations 2 --offline --run_name smoke
```

正式训练：`python scripts/rsl_rl/train_duet.py --headless`。默认4096环境、seed 1、25001轮，每500轮保存，W&B在线记录；首次使用前运行`wandb login`。不上传密钥，离线可加`--offline`。

[训练配方与22项奖励](docs/TRAINING_RECIPE.md) · [代码知识图谱与迁移差异](docs/ARCHITECTURE.md) · [完整评估](docs/RESULTS.md) · [视频和模型附件](docs/ARTIFACTS.md)

## 致谢

感谢 [DUET](https://github.com/bae-air-lab/DUET) 和 [HOMIE](https://github.com/InternRobotics/OpenHomie) 项目的开源工作。
