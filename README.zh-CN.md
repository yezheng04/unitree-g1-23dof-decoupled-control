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

## 安装

克隆前，请先按 [Isaac Lab 官方安装指南](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html) 配置 **Isaac Sim 5.1.0、Isaac Lab 2.3.2 和 Python 3.11**，Isaac Lab 选择 [v2.3.2](https://github.com/isaac-sim/IsaacLab/tree/v2.3.2)。然后在 Isaac Lab 目录之外克隆本仓库：

```bash
git clone https://github.com/yezheng04/unitree-g1-23dof-decoupled-control.git
cd unitree-g1-23dof-decoupled-control
```

从已配置好的 Isaac Lab 环境创建项目环境，将 `env_isaaclab` 替换成你的实际环境名：

```bash
conda create -n g1-23dof --clone env_isaaclab -y
conda activate g1-23dof
python -m pip install -r requirements-runtime.txt -e .
python scripts/setup/download_assets.py
```

克隆包含全部项目源码；下载脚本补齐机器人资源、**22000 checkpoint** 及配套元数据。完整视频可额外运行 `python scripts/setup/download_assets.py --videos` 下载，也可从 [Release](https://github.com/yezheng04/unitree-g1-23dof-decoupled-control/releases/tag/v0.1.0-22000) 单独下载。

## 训练

```bash
# 快速检查安装
python scripts/rsl_rl/train_duet.py --headless --num_envs 64 --max_iterations 2 --offline --run_name smoke
# 正式训练
wandb login
python scripts/rsl_rl/train_duet.py --headless
```

## Deploy / 部署

已提供 **MuJoCo sim2sim** 和 **G1 EDU 23 自由度真机 HG DDS 控制**入口。部署前将 checkpoint 导出为包含观测归一化的 TorchScript 策略；环境安装、参数和操作步骤见[部署说明](docs/DEPLOY.md)。

```bash
python -m pip install -r requirements-deploy.txt
python -m deploy.export_policy --checkpoint artifacts/checkpoints/model_22000.pt
# MuJoCo 物理闭环；删除 --headless 可打开可视化窗口
python -m deploy.sim2sim --headless --vx 0.5 --duration 18
# 默认只进行离线推理，不连接机器人
python -m deploy.sim2real
```

安装 `deploy/requirements-hardware.txt` 中的真机 SDK 后，先确认网口与 IMU 坐标系：

```bash
python -m deploy.sim2real --read-only --interface YOUR_ROBOT_NIC --imu-frame pelvis --duration 10
# 按部署说明完成现场检查后，再显式启用控制
python -m deploy.sim2real --execute --interface YOUR_ROBOT_NIC --imu-frame pelvis
```

`YOUR_ROBOT_NIC` 替换为实际网口。按住 L1 并按 Start 进入默认姿态，继续按住 L1 并按 A 启动策略；松开 L1 或按 B/Select 请求锁存的阻尼停止。接管运动服务需要显式添加 `--release-mode`。

## 致谢

感谢 [DUET](https://github.com/bae-air-lab/DUET) 和 [HOMIE](https://github.com/InternRobotics/OpenHomie) 项目的开源工作。
