# Deploy / 部署

本部署目录面向 **Unitree G1 EDU 23 自由度**，使用本仓库的 IsaacLab checkpoint。策略控制腿和腰的 13 个关节，手臂 10 个关节独立保持默认姿态；MuJoCo 可额外启用手臂运动测试。

| 入口 | 用途 | 当前验证范围 |
|---|---|---|
| `python -m deploy.export_policy` | 导出含观测归一化的 TorchScript 策略 | 与 RSL-RL 原策略的 900 帧观测输出完全一致 |
| `python -m deploy.sim2sim` | MuJoCo 实时物理闭环 | 已完成 9 个 18 秒单次测试，见下文 |
| `python -m deploy.sim2real` | G1 HG DDS 真机控制 | 离线推理与模拟故障测试；未做真机验收 |

## 1. 环境和模型导出

先完成[项目安装](INSTALL.md)，再从仓库根目录执行。部署不需要启动 Isaac Sim。

```bash
conda activate g1-23dof
python -m pip install -r requirements-deploy.txt
python -m deploy.export_policy --checkpoint artifacts/checkpoints/model_22000.pt --output artifacts/deploy/22000
```

仅部署时也可单独创建 Python 3.11 环境（`conda create -n g1-deploy python=3.11 -y`），激活后安装 `requirements-deploy.txt` 并运行 `python scripts/setup/download_assets.py`，再导出策略。

生成 `policy.pt`（TorchScript 推理文件）和 `deploy.json`（参数与 SHA256）。原来的 `model_22000.pt` 仍是训练 checkpoint，可以恢复训练；部署文件只包含推理所需内容。导出采用 `weights_only=True` 读取当前 checkpoint，并校验机器人模型和配套元数据。

导出保留 `[512, 256, 128]` ELU actor 和 `(obs - mean) / (std + 0.01)` 归一化，不依赖 IsaacLab 或 RSL-RL 运行时。观测是 71 维：机体系角速度 3、投影重力 3、速度命令 3、步态相位 2、23 关节相对位置、23 关节速度、13 维上次原始动作、高度命令 1。所有观测使用训练时的单位尺度；步态周期 0.6 秒，速度命令范数小于 0.1 时相位置零。策略频率 50 Hz。

## 2. Sim2Sim：MuJoCo

有图形界面的电脑可直接打开交互窗口：

```bash
# 站立
python -m deploy.sim2sim --duration 60
# 前进；前 3 秒保持零速度，然后切换命令
python -m deploy.sim2sim --vx 0.5 --duration 60
# 转向、下蹲、独立上肢扰动
python -m deploy.sim2sim --yaw 0.3
python -m deploy.sim2sim --height 0.38
python -m deploy.sim2sim --vx 0.5 --arm-motion
```

服务器无显示器时：

```bash
python -m deploy.sim2sim --headless --vx 0.5 --duration 18 \
  --trace --output outputs/sim2sim/forward.json
# 额外检查硬件控制所用的位置和目标变化率约束
python -m deploy.sim2sim --headless --vx 0.3 --hardware-limits \
  --output outputs/sim2sim/hardware_limits.json
```

这是实际执行 `mj_step` 的策略闭环。旧的 `render_duet_replay.py` 仅渲染已记录的 PhysX 轨迹，两者用途不同。MuJoCo 默认步长 0.002 秒、每 10 步推理一次；使用锁定的 Rev1.0 MJCF，按训练 contract 配置默认姿态、PD、armature 和 URDF 力矩限制，踝部使用 35 Nm。保留四球足碰撞，并添加训练中的相邻链节碰撞排除。

输出 JSON 记录命令、存活状态、测量速度、根节点高度和倾角。`--trace` 另外保存 NPZ 观测/动作轨迹。跌倒或提前关闭窗口返回非零退出码。`--arm-motion` 使用幅度 0.15 rad、频率 0.25 Hz 的独立正弦测试轨迹，不能替代完整随机扰动评测。`--hardware-limits` 包含位置和目标变化率限制，未仿真 DDS 时延、固件或无线手柄。

## 3. Sim2Real：G1 EDU 23DoF

### 安装与离线检查

真机驱动使用 [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)，固定到 `814556d15970dd2ecf1c9984e845ca02ab07e206`。Cyclone DDS 的系统依赖按该 SDK 的安装说明配置，再执行：

```bash
python -m pip install -r deploy/requirements-hardware.txt
# 不初始化 DDS，不连接机器人，执行 100 次本地推理
python -m deploy.sim2real
python -m unittest discover -s deploy/tests -v
```

训练的 23 个关节不能直接当作 HG 消息中的前 23 个槽位。映射固定为：

```text
腿和腰：0..12
左臂：15..19
右臂：22..26
```

未使用的腰 roll/pitch、腕 pitch/yaw 以及额外槽位始终禁用。控制采用 `unitree_hg` 的 `rt/lowstate` / `rt/lowcmd` 和 PR 模式，`mode_machine` 从有效状态包读取，校验收发 CRC。不能用 29 自由度机器人代替此配置。

### 先只读，再控制

网口名称尚待提供。下面的 `ROBOT_IFACE` 必须替换成部署电脑实际连接机器人的网口；IMU 坐标系也需在只读阶段确认。官方 G1 配置使用 pelvis；如果实际固件提供的是 torso 数据，选择 `--imu-frame torso`，程序会用腰 yaw 角度和角速度变换到 pelvis。

```bash
ROBOT_IFACE=enp3s0  # 示例值，替换为你的实际网口
# 只接收状态，输出关节、重力、角速度、模式与手柄按键；不创建 lowcmd 发布器
python -m deploy.sim2real --read-only \
  --interface "$ROBOT_IFACE" --imu-frame pelvis --duration 10
```

连接信息确认后，先在有防跌落支撑、操作员可用物理急停的条件下核对关节方向、IMU、手柄和默认姿态。启动策略时脚应接触地面；悬空不能检验行走策略。当前代码未经过真机测试，因此这些命令是待现场验证的实验控制入口。

```bash
# 机器人应已进入允许低层控制的状态，且没有其他 lowcmd 写入者
python -m deploy.sim2real --execute \
  --interface "$ROBOT_IFACE" --imu-frame pelvis --duration 60
# 若需要由本程序显式请求释放现有运动服务，额外添加 --release-mode
```

程序先检查运动服务。存在活动服务时默认退出；仅显式传入 `--release-mode` 才调用 `ReleaseMode`，并再次确认服务已释放。不要同时运行其他底层控制器。

控制顺序如下：

1. `WAIT`：输出阻尼。按住 **L1**，再按 **Start**，进入 3 秒默认姿态插值。
2. `READY`：继续按住 **L1**，默认姿态跟踪误差小于 0.3 rad 后按 **A** 启动策略。
3. `POLICY`：左摇杆控制前后/横向，右摇杆控制转向；上限分别为 0.3 m/s、0.15 m/s、0.3 rad/s。速度命令经过渐变处理，手臂保持训练默认姿态。
4. 松开 **L1**、按 **B / Select**、Ctrl+C 或到达运行时长后停止。故障锁存并请求阻尼，不自动重新启用策略或原运动服务。切换为阻尼时机器人可能失去站立支撑，操作员仍需负责支撑与物理急停。

`--height` 默认 0.78 m，真机命令限制在 0.60–0.78 m；初次现场验证使用默认值。真机不执行随机手臂扰动；当前未接入 VLA、IK 或外部手臂目标流。

独立的 250 Hz 发布线程监测状态/控制更新时间（100 ms）、连续状态 tick、CRC、机型模式变化、45° 倾角、关节位置/速度和 70°C 温度阈值。策略推理仍为 50 Hz；超过控制期限会锁存停止。目标位置保留 0.02 rad 机械限位余量，目标变化率上限为腿/腰 8 rad/s、手臂 2 rad/s，并使用最新关节状态限制 PD 力矩估计。这个估计不能替代固件的电流/力矩保护；Python 调度、网络中断或进程被强制终止也不保证阻尼指令送达。状态包的新鲜度不等于无线手柄链路的新鲜度，仍需验证固件的失联处理。

## 4. 验证记录

验证环境：Python 3.11、PyTorch 2.7.0、MuJoCo 3.3.6；模型为已发布的 22000 checkpoint。9 个单次 18 秒物理测试均未跌倒，0–3 秒为站立，6 秒后统计速度。它们是初步迁移检查，不是随机化鲁棒性或真机验收。

| 条件 | 目标 vx / vy / yaw | 测量均值 vx / vy / yaw |
|---|---|---|
| 站立 | 0 / 0 / 0 | -0.000 / 0.000 / -0.000 |
| 前进 | 0.50 / 0 / 0 | 0.551 / 0.001 / -0.001 |
| 手臂扰动前进 | 0.50 / 0 / 0 | 0.549 / 0.003 / -0.005 |
| 后退 | -0.30 / 0 / 0 | -0.117 / 0.001 / -0.003 |
| 横移 | 0 / 0.20 / 0 | 0.116 / 0.140 / 0.015 |
| 左转 | 0 / 0 / 0.30 | 0.117 / -0.043 / 0.198 |
| 0.38 m 下蹲 | 0 / 0 / 0 | 约 0 / 0 / 0 |
| 加目标约束的站立与前进 | 见原始记录 | 见原始记录 |

完整数据见 [deployment_22000.json](../results/deployment_22000.json)。其中根节点世界高度不等于训练奖励中的足部参考点相对高度，不能直接用它计算原高度指标。发布的旧演示视频仍然是 PhysX 回放，不是新 MuJoCo 测试录像。

## 5. 实现参考

部署流程参考 [DUET 的 Deploy](https://github.com/bae-air-lab/DUET/tree/439c21bd2817feedd519475e8301195b95cb6fcb/deploy)。本版使用 Python/TorchScript，按当前 IsaacLab checkpoint 重建观测和控制接口，未复用上游训练好的策略或部署默认高度。

HG 消息、PR 模式、机型字段和 CRC 参考 [官方 G1 低层示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/example/g1/low_level/g1_low_level_example.py)；手柄字节布局和 IMU 坐标处理参考 [Unitree RL Gym 部署](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b/deploy/deploy_real)。
