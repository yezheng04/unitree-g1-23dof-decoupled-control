"""PhysX contact, kinematics, observations and randomization used by DUET.

Adapted from Unitree mjlab 1425b15 / mjlab 1.2.0 (Apache-2.0).
"""
import torch

from isaaclab.envs import mdp as base


from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


from isaaclab.sensors import ContactSensor, ContactSensorCfg

from isaaclab.utils import configclass

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .asset_cfg import JOINTS, BODY_NAMES, FOOT_NAMES, FOOT_OFFSET, IMU_OFFSET

class FilteredContactSensor(ContactSensor):
    """One source body per sensor; modes and history use filtered normal+friction forces.

    MuJoCo found>0 is approximated by force norm > 1e-3 N in PhysX.
    Self-collision reduction is per-body net force, not MuJoCo per-contact slots.
    """
    def _update_buffers_impl(self, env_ids):
        if self.cfg.track_air_time:
            old_air=self._data.current_air_time[env_ids].clone()
            old_contact=self._data.current_contact_time[env_ids].clone()
            elapsed=(self._timestamp[env_ids]-self._timestamp_last_update[env_ids]).unsqueeze(-1)
        super()._update_buffers_impl(env_ids)
        total=self._data.force_matrix_w[env_ids]+self._data.friction_forces_w[env_ids]
        self._data.force_matrix_w[env_ids]=total
        self._data.net_forces_w[env_ids]=total.sum(dim=2)
        if self.cfg.history_length>0:
            self._data.force_matrix_w_history[env_ids,0]=total
        if self.cfg.track_air_time:
            contact=self._data.net_forces_w[env_ids].norm(dim=-1)>self.cfg.force_threshold
            self._data.last_air_time[env_ids]=torch.where((old_air>0)&contact,old_air+elapsed,self._data.last_air_time[env_ids])
            self._data.last_contact_time[env_ids]=torch.where((old_contact>0)&~contact,old_contact+elapsed,self._data.last_contact_time[env_ids])
            self._data.current_air_time[env_ids]=torch.where(~contact,old_air+elapsed,0.)
            self._data.current_contact_time[env_ids]=torch.where(contact,old_contact+elapsed,0.)

@configclass
class FilteredContactSensorCfg(ContactSensorCfg):
    class_type: type=FilteredContactSensor

class BiasedJointPositionAction(JointPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        if not hasattr(env,'encoder_bias'):
            env.encoder_bias=torch.zeros((env.num_envs,env.scene['robot'].num_joints),device=env.device)
    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions-self._env.encoder_bias[:,self._joint_ids],joint_ids=self._joint_ids)

@configclass
class BiasedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type=BiasedJointPositionAction

def command(env):return env.command_manager.get_command('twist')

def joint_ids(env):return env.action_manager.get_term('joint_pos')._joint_ids

def imu_ang_vel(env):return env.scene['robot'].data.root_link_ang_vel_b

def imu_lin_vel(env):
    d=env.scene['robot'].data
    offset=torch.tensor(IMU_OFFSET,device=env.device).expand(env.num_envs,-1)
    return d.root_link_lin_vel_b+torch.cross(d.root_link_ang_vel_b,offset,dim=-1)

def q_rel(env):
    d=env.scene['robot'].data; ids=joint_ids(env)
    # Source calls joint_pos_rel with biased=False despite startup encoder bias.
    return d.joint_pos[:,ids]-d.default_joint_pos[:,ids]

def dq_rel(env):
    d=env.scene['robot'].data; ids=joint_ids(env)
    return d.joint_vel[:,ids]-d.default_joint_vel[:,ids]

def phase(env):
    t=(env.episode_length_buf*env.step_dt)%.6/.6
    p=torch.stack((torch.sin(2*torch.pi*t),torch.cos(2*torch.pi*t)),dim=-1)
    return torch.where((command(env).norm(dim=1)<.1)[:,None],0.,p)

def feet_state(env):
    if not hasattr(env,'mj_foot_ids'):
        env.mj_foot_ids=env.scene['robot'].find_bodies(FOOT_NAMES,preserve_order=True)[0]
    d=env.scene['robot'].data; ids=env.mj_foot_ids
    offset=torch.tensor(FOOT_OFFSET,device=env.device).expand(env.num_envs,2,-1)
    r=quat_apply(d.body_link_quat_w[:,ids],offset)
    pos=d.body_link_pos_w[:,ids]+r
    vel=d.body_link_lin_vel_w[:,ids]+torch.cross(d.body_link_ang_vel_w[:,ids],r,dim=-1)
    return pos,vel

def feet_sensors(env):return [env.scene['left_ground'],env.scene['right_ground']]

def foot_height(env):return feet_state(env)[0][:,:,2]-env.scene.env_origins[:,2:3]

def foot_air_time(env):return torch.cat([s.data.current_air_time for s in feet_sensors(env)],dim=1)

def foot_contact(env):return torch.cat([(s.data.net_forces_w.norm(dim=-1)>1e-3).float() for s in feet_sensors(env)],dim=1)

def foot_force(env):return torch.cat([s.data.net_forces_w for s in feet_sensors(env)],dim=1)

def foot_contact_forces(env):
    f=foot_force(env).flatten(1)
    return f.sign()*torch.log1p(f.abs())

def torso_id(env):
    if not hasattr(env,'mj_torso_id'):env.mj_torso_id=env.scene['robot'].find_bodies(['torso_link'])[0][0]
    return env.mj_torso_id

def encoder_bias(env,env_ids):
    if env_ids is None:env_ids=slice(None)
    env.encoder_bias[env_ids]=torch.empty_like(env.encoder_bias[env_ids]).uniform_(-.015,.015)

def foot_friction(env,env_ids):
    """One common sampled coefficient for all foot shapes, static=dynamic."""
    robot=env.scene['robot'];view=robot.root_physx_view
    mats=view.get_material_properties().clone()
    ids=torch.arange(env.num_envs,dtype=torch.int32) if env_ids is None else env_ids.cpu().to(torch.int32)
    # Query each link's shape count; do not assume uniform number of colliders.
    counts=[]
    for link_path in robot.root_physx_view.link_paths[0]:
        rv=robot._physics_sim_view.create_rigid_body_view(link_path)
        counts.append(rv.max_shapes)
    samples=torch.empty((len(ids),1,1)).uniform_(.3,1.6)
    offset=0
    for n,count in zip(robot.body_names,counts):
        if n in FOOT_NAMES:
            mats[ids,offset:offset+count,:2]=samples
        offset+=count
    assert offset==mats.shape[1],(offset,mats.shape)
    view.set_material_properties(mats,ids)
    env.mj_sampled_friction=samples[:,0,0].to(env.device)
