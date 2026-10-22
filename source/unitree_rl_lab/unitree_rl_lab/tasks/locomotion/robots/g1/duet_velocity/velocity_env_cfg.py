"""DUET Flat on the existing Rev1.0/PhysX body; source recipe is pinned in JSON."""
import json, math
from pathlib import Path
from isaaclab.utils import configclass
from isaaclab.envs import mdp as base
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.managers import (ObservationGroupCfg, ObservationTermCfg as ObsTerm,
    RewardTermCfg, CurriculumTermCfg, EventTermCfg, SceneEntityCfg)
from ..common.scene_cfg import BaseEnvCfg as PortEnvCfg, SceneCfg as PortSceneCfg, PolicyCfg as PortPolicyCfg, EventsCfg as PortEventsCfg
from ..common.asset_cfg import ROBOT_CFG, JOINTS, ACTION_SCALE
from ..common import mdp as port
from . import rewards, adapters, curriculums
from .actions import UpperBodyPoseActionCfg
from .commands import BaseHeightCommandCfg
SOURCE = json.loads(Path(__file__).with_name('upstream_config.json').read_text())
LOWER = JOINTS[:13]
ARMS = JOINTS[13:]

def entity(params):
    if isinstance(params, list): return [entity(v) for v in params]
    if not isinstance(params, dict): return params
    if 'name' in params and 'joint_ids' in params:
        kw = {k: params[k] for k in ('joint_names', 'body_names') if params.get(k)}
        if 'joint_names' in kw: kw['joint_names'] = LOWER
        return SceneEntityCfg(params['name'], preserve_order=True, **kw)
    return {k: entity(v) for k, v in params.items()}

def build_rewards():
    out = {}
    for name, row in SOURCE['rewards'].items():
        fn = row['func'].split(':')[-1]
        func = adapters.self_collision_cost if fn == 'self_collision_cost' else getattr(rewards, fn, None)
        if func is None: func = getattr(base, fn)
        out[name] = RewardTermCfg(func=func, weight=row['weight'], params=entity(row['params']))
    return out

def build_curriculum():
    return {name: CurriculumTermCfg(func=getattr(curriculums, row['func'].split(':')[-1]), params=row['params']) for name, row in SOURCE['curriculum'].items()}

def build_arm():
    row = SOURCE['actions']['upper_body_pose']
    cfg = UpperBodyPoseActionCfg(asset_name='robot', joint_names=ARMS)
    for key, value in row.items():
        if key not in ('class_type','entity_name','joint_names') and hasattr(cfg, key): setattr(cfg, key, value)
    return cfg

def build_height():
    row = SOURCE['commands']['base_height']
    cfg = BaseHeightCommandCfg(resampling_time_range=(3.,8.), debug_vis=False)
    for key, value in row.items():
        if key not in ('class_type','debug_vis') and hasattr(cfg, key): setattr(cfg, key, value)
    return cfg

@configclass
class SceneCfg(PortSceneCfg):
    robot = ROBOT_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')

@configclass
class ActionsCfg:
    joint_pos = port.BiasedJointPositionActionCfg(asset_name='robot', joint_names=LOWER, preserve_order=True,
        scale={n: ACTION_SCALE[n] for n in LOWER}, use_default_offset=True)
    upper_body_pose = build_arm()

@configclass
class CommandsCfg:
    twist = UniformVelocityCommandCfg(asset_name='robot', resampling_time_range=(3.,8.), rel_standing_envs=.2,
        rel_heading_envs=1., heading_command=True, heading_control_stiffness=.5, debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-.3,.6),lin_vel_y=(-.2,.2),ang_vel_z=(-.4,.4),heading=(-math.pi,math.pi)))
    base_height = build_height()

@configclass
class PolicyCfg(PortPolicyCfg):
    # All 23 measured joint positions AND velocities, only 13 previous RL actions.
    joint_pos = ObsTerm(func=adapters.q_rel, noise=None)
    joint_vel = ObsTerm(func=adapters.dq_rel)
    height_command = ObsTerm(func=base.generated_commands, params={'command_name':'base_height'})
    def __post_init__(self):
        super().__post_init__()
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg
        self.joint_pos.noise = AdditiveUniformNoiseCfg(n_min=-.01,n_max=.01)
        self.joint_vel.noise = AdditiveUniformNoiseCfg(n_min=-1.5,n_max=1.5)

@configclass
class CriticCfg(PolicyCfg):
    base_lin_vel = ObsTerm(func=port.imu_lin_vel)
    foot_height = ObsTerm(func=port.foot_height)
    foot_air_time = ObsTerm(func=port.foot_air_time)
    foot_contact = ObsTerm(func=port.foot_contact)
    foot_contact_forces = ObsTerm(func=port.foot_contact_forces)
    arm_traj_vel = ObsTerm(func=adapters.arm_traj_vel)
    def __post_init__(self):
        super().__post_init__(); self.enable_corruption=False

@configclass
class ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class EventsCfg(PortEventsCfg):
    push_robot = EventTermCfg(func=adapters.push_and_relatch_anchor,mode='interval',interval_range_s=(3.,5.),
        params={'velocity_range':SOURCE['events']['push_robot']['params']['velocity_range']})
    base_com = EventTermCfg(func=base.randomize_rigid_body_com,mode='startup',params={
        'asset_cfg':SceneEntityCfg('robot',body_names=['torso_link']),
        'com_range':{'x':(-.1,.1),'y':(-.06,.06),'z':(-.05,.05)}})
    # Resolve the actual rubber-hand rigid bodies, matching upstream payload semantics.
    hand_payload = EventTermCfg(func=adapters.payload,mode='reset',params={
        'asset_cfg':SceneEntityCfg('robot',body_names=['left_wrist_roll_rubber_hand','right_wrist_roll_rubber_hand']),
        'mass_distribution_params':(.25,1.75), 'operation':'add'})
    torso_payload = EventTermCfg(func=adapters.payload,mode='reset',params={
        'asset_cfg':SceneEntityCfg('robot',body_names=['torso_link']), 'mass_distribution_params':(-.5,2.), 'operation':'add'})
    hand_com = EventTermCfg(func=base.randomize_rigid_body_com,mode='startup',params={
        'asset_cfg':SceneEntityCfg('robot',body_names=['left_wrist_roll_rubber_hand','right_wrist_roll_rubber_hand']),
        'com_range':{'x':(-.05,.05),'y':(-.05,.05),'z':(-.05,.05)}})

@configclass
class RobotEnvCfg(PortEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096,env_spacing=3.,lazy_sensor_update=False)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards = build_rewards()
    curriculum = build_curriculum()
    def __post_init__(self):
        super().__post_init__()
        self.terminations.fell_over.params['limit_angle']=math.radians(60.)

@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__(); self.scene.num_envs=32
        self.observations.policy.enable_corruption=False
        self.events.push_robot=None; self.curriculum={}
        self.actions.upper_body_pose.enabled=False
        self.commands.twist.ranges.lin_vel_x=(-1.,1.)
        self.commands.twist.ranges.lin_vel_y=(-.5,.5)
        self.commands.twist.ranges.ang_vel_z=(-1.,1.)
        self.commands.base_height.floor_curriculum_start=.24
        self.commands.base_height.walk_min_height=.60
        self.commands.base_height.nominal_env_fraction=0.
