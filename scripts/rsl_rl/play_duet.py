"""Inspect the DUET policy with explicit velocity/height commands; no export."""
import argparse
import types
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--vx", type=float, default=0.5)
p.add_argument("--vy", type=float, default=0.0)
p.add_argument("--yaw", type=float, default=0.0)
p.add_argument("--base_height", type=float, default=0.78)
p.add_argument("--arms", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if not 0.24 <= args.base_height <= 0.78:
    p.error("Height must be inside the trained [0.24, 0.78] m envelope.")
app = AppLauncher(args).app
env = None
try:
    import gymnasium as gym
    import torch
    import unitree_rl_lab.tasks
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.velocity_env_cfg import RobotPlayEnvCfg
    from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.rsl_rl_ppo_cfg import DuetPPORunnerCfg
    from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.runner import DuetRunner

    cfg = RobotPlayEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 3.0
    cfg.actions.upper_body_pose.enabled = args.arms
    cfg.actions.upper_body_pose.ratio_stages = ((0, 1.0),)
    cfg.actions.upper_body_pose.clean_fraction_stages = ((0, 0.0),)
    cfg.viewer.eye = (3.0, 3.0, 2.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.7)
    if args.device:
        cfg.sim.device = args.device
    env = gym.make("Unitree-G1-23dof-Duet-Velocity", cfg=cfg).unwrapped
    wrapper = RslRlVecEnvWrapper(env)
    runner = DuetRunner(wrapper, DuetPPORunnerCfg().to_dict(), log_dir=None, device=env.device)
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    target = torch.tensor([args.vx, args.vy, args.yaw], device=env.device)

    def fixed_twist(self):
        self.vel_command_b[:] = target
        self.is_standing_env[:] = target.norm() < 1e-8

    def fixed_height(self):
        self._height[:] = args.base_height

    twist = env.command_manager.get_term("twist")
    height = env.command_manager.get_term("base_height")
    twist._update_command = types.MethodType(fixed_twist, twist)
    height._update_command = types.MethodType(fixed_height, height)
    with torch.inference_mode():
        wrapper.reset()
        for _ in range(args.steps):
            if not app.is_running():
                break
            fixed_twist(twist)
            fixed_height(height)
            obs = wrapper.get_observations()
            actions = policy(obs)
            assert torch.isfinite(actions).all()
            wrapper.step(actions)
    print("DUET_PLAY_PASS", flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
