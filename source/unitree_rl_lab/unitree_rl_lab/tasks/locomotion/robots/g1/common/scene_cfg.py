"""Non-registered scene and observation building blocks for DUET."""
import math

import isaaclab.sim as sim

from isaaclab.assets import AssetBaseCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, mdp as base

from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, TerminationTermCfg as DoneTerm, EventTermCfg as EventTerm, SceneEntityCfg

from isaaclab.scene import InteractiveSceneCfg

from isaaclab.terrains import TerrainImporterCfg

from isaaclab.utils import configclass

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Noise

from .asset_cfg import ROBOT_CFG, JOINTS, ACTION_SCALE, BODY_NAMES

from . import mdp

@configclass
class SceneCfg(InteractiveSceneCfg):
    robot=ROBOT_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    terrain=TerrainImporterCfg(prim_path='/World/ground',terrain_type='plane',collision_group=-1,
        physics_material=sim.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.,friction_combine_mode='multiply',restitution_combine_mode='multiply'))
    light=AssetBaseCfg(prim_path='/World/light',spawn=sim.DomeLightCfg(intensity=750.))
    left_ground=mdp.FilteredContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/left_ankle_roll_link',filter_prim_paths_expr=['/World/ground/terrain/GroundPlane/CollisionPlane'],track_air_time=True,track_friction_forces=True,force_threshold=.001,history_length=4,update_period=.005,max_contact_data_count_per_prim=16)
    right_ground=left_ground.replace(prim_path='{ENV_REGEX_NS}/Robot/right_ankle_roll_link')
    def __post_init__(self):
        # IsaacLab contact filtering requires exactly one source link per sensor.
        for i,name in enumerate(BODY_NAMES):
            setattr(self,f'self_{i}',mdp.FilteredContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+name,
                filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/'+other for other in BODY_NAMES if other!=name],track_friction_forces=True,
                history_length=4,update_period=.005,max_contact_data_count_per_prim=16))

@configclass
class PolicyCfg(ObsGroup):
    base_ang_vel=ObsTerm(func=mdp.imu_ang_vel,noise=Noise(n_min=-.2,n_max=.2))
    projected_gravity=ObsTerm(func=base.projected_gravity,noise=Noise(n_min=-.05,n_max=.05))
    command=ObsTerm(func=mdp.command)
    phase=ObsTerm(func=mdp.phase)
    joint_pos=ObsTerm(func=mdp.q_rel,noise=Noise(n_min=-.01,n_max=.01))
    joint_vel=ObsTerm(func=mdp.dq_rel,noise=Noise(n_min=-1.5,n_max=1.5))
    actions=ObsTerm(func=base.last_action)
    def __post_init__(self):
        self.concatenate_terms=True;self.enable_corruption=True;self.history_length=1

@configclass
class EventsCfg:
    reset_base=EventTerm(func=base.reset_root_state_uniform,mode='reset',params={'pose_range':{'x':(-.5,.5),'y':(-.5,.5),'z':(0.,0.),'yaw':(-3.14,3.14)},'velocity_range':{}})
    reset_joints=EventTerm(func=base.reset_joints_by_offset,mode='reset',params={'position_range':(0.,0.),'velocity_range':(0.,0.)})
    push_robot=EventTerm(func=base.push_by_setting_velocity,mode='interval',interval_range_s=(5.,6.),params={'velocity_range':{'x':(-.5,.5),'y':(-.5,.5),'z':(-.4,.4),'roll':(-.52,.52),'pitch':(-.52,.52),'yaw':(-.78,.78)}})
    foot_friction=EventTerm(func=mdp.foot_friction,mode='startup')
    encoder_bias=EventTerm(func=mdp.encoder_bias,mode='startup')
    base_com=EventTerm(func=base.randomize_rigid_body_com,mode='startup',params={'asset_cfg':SceneEntityCfg('robot',body_names=['torso_link']),'com_range':{'x':(-.05,.05),'y':(-.05,.05),'z':(-.05,.05)}})

@configclass
class TerminationsCfg:
    time_out=DoneTerm(func=base.time_out,time_out=True)
    fell_over=DoneTerm(func=base.bad_orientation,params={'limit_angle':math.radians(70.)})

@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Simulation settings only; concrete actions/rewards/commands belong to DUET."""
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3., lazy_sensor_update=False)
    terminations: TerminationsCfg = TerminationsCfg()
    def __post_init__(self):
        self.decimation=4;self.episode_length_s=20.;self.sim.dt=.005;self.sim.render_interval=4
        self.sim.physics_material=self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count=10*2**15
        self.sim.physx.gpu_max_rigid_contact_count=2**23
