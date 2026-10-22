"""Explicit PhysX adapters for DUET's foot sites, sensors and state layout."""
from types import SimpleNamespace
import torch
from isaaclab.envs import mdp as base
from isaaclab.managers import SceneEntityCfg
from ..common import mdp as port
from ..common.asset_cfg import JOINTS, BODY_NAMES


def ordered_ids(env):
    if not hasattr(env, 'duet_joint_ids'):
        env.duet_joint_ids = env.scene['robot'].find_joints(JOINTS, preserve_order=True)[0]
    return env.duet_joint_ids


def q_rel(env):
    d = env.scene['robot'].data
    ids = ordered_ids(env)
    return d.joint_pos[:, ids] - d.default_joint_pos[:, ids]


def dq_rel(env):
    d = env.scene['robot'].data
    ids = ordered_ids(env)
    return d.joint_vel[:, ids] - d.default_joint_vel[:, ids]


def site_pos(env):
    return port.feet_state(env)[0]


def site_vel(env):
    return port.feet_state(env)[1]


def feet_sensor(env):
    sensors = port.feet_sensors(env)
    return SimpleNamespace(data=SimpleNamespace(
        found=port.foot_contact(env), force=port.foot_force(env),
        current_air_time=torch.cat([s.data.current_air_time for s in sensors], 1),
        current_contact_time=torch.cat([s.data.current_contact_time for s in sensors], 1)))


def self_collision_cost(env, sensor_name='self_collision', force_threshold=50.):
    # PhysX filtered net force per link differs from MuJoCo contact-slot force.
    histories = [env.scene[f'self_{i}'].data.force_matrix_w_history.sum(dim=3).norm(dim=-1).squeeze(-1) > force_threshold for i in range(len(BODY_NAMES))]
    return torch.stack(histories, -1).any(-1).float().sum(1)


def whole_com(env):
    robot = env.scene['robot']
    if not hasattr(env, 'duet_body_mass'):
        env.duet_body_mass = robot.root_physx_view.get_masses().to(env.device)
    mass = env.duet_body_mass[..., None]
    return (robot.data.body_com_pos_w * mass).sum(1) / mass.sum(1)


class payload(base.randomize_rigid_body_mass):
    def __call__(self, env, env_ids, asset_cfg, mass_distribution_params, operation='add', recompute_inertia=False):
        super().__call__(env, env_ids, asset_cfg=asset_cfg,
            mass_distribution_params=mass_distribution_params, operation=operation,
            recompute_inertia=recompute_inertia)
        env.duet_body_mass = env.scene['robot'].root_physx_view.get_masses().to(env.device)


def push_and_relatch_anchor(env, env_ids, velocity_range):
    base.push_by_setting_velocity(env, env_ids, velocity_range)
    if 'idle_position_anchor' in env.reward_manager.active_terms:
        env.reward_manager.get_term_cfg('idle_position_anchor').func.reset(env_ids)


def arm_traj_vel(env, action_term_name='upper_body_pose'):
    return env.action_manager.get_term(action_term_name).traj_vel


def diagnostics(env):
    d = env.scene['robot'].data
    cmd = env.command_manager.get_command('twist')
    h = env.command_manager.get_command('base_height')[:, 0]
    measured = d.root_link_pos_w[:, 2] - site_pos(env)[:, :, 2].min(1).values + .02
    arm = env.action_manager.get_term('upper_body_pose')
    from .rewards import _com_support_ratio
    ratio, double = _com_support_ratio(env, 'feet_ground_contact', SceneEntityCfg('robot'), .09, .04)
    g = port.quat_apply_inverse(d.body_link_quat_w[:, port.torso_id(env)], d.GRAVITY_VEC_W)
    return {'lin_vel_track_error': (cmd[:, :2]-d.root_link_lin_vel_b[:, :2]).norm(dim=1).mean(),
        'yaw_track_error': (cmd[:, 2]-d.root_link_ang_vel_b[:, 2]).abs().mean(),
        'height_track_error': (measured-h).abs().mean(),
        'arm_traj_speed': arm.traj_vel.abs().mean(), 'arm_traj_accel': arm.traj_acc.abs().mean(),
        'torso_pitch_abs': torch.atan2(g[:, 0], -g[:, 2]).abs().mean(),
        'torso_roll_abs': torch.atan2(g[:, 1], -g[:, 2]).abs().mean(),
        'com_support_error': (ratio*double).mean(),
        'command_height_floor': torch.tensor(env.command_manager.get_term('base_height').current_floor, device=env.device),
        'command_yaw_max': torch.tensor(env.command_manager.get_term('twist').cfg.ranges.ang_vel_z[1], device=env.device),
        'arm_curriculum_ratio': torch.tensor(arm.curriculum_ratio, device=env.device)}
