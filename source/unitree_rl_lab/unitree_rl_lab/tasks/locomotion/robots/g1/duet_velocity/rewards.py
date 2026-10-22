"""Adapted from bae-air-lab/DUET (Apache-2.0); IsaacLab API port.
See DUET_ISAACLAB_PORT.md for physics and asset differences.
"""
from __future__ import annotations
import torch
from isaaclab.assets import Articulation as Entity
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, wrap_to_pi
from isaaclab.utils.string import resolve_matching_names_values
from .adapters import site_pos, site_vel, feet_sensor, whole_com
_DEFAULT_ASSET_CFG = SceneEntityCfg('robot')

def track_linear_velocity(env: ManagerBasedRlEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + 2 * z_error
    return torch.exp(-lin_vel_error / std ** 2)

def track_angular_velocity(env: ManagerBasedRlEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + 0.05 * xy_error
    return torch.exp(-ang_vel_error / std ** 2)

def track_base_height(env: ManagerBasedRlEnv, command_name: str, ankle_sole_distance: float=0.0, asset_cfg: SceneEntityCfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Reward tracking the commanded base (pelvis) height above the feet.

  Base height is measured as the root link height above the **lower (stance) foot**:
  ``exp(-4 * |base_height - h_cmd + sole_dist|)``, with ``slice(None)``
  pointing to the foot sites.

  Using the stance (lower) foot -- not the higher one -- keeps the reference
  identical whether standing (both feet down) or walking (one foot swinging up).
  With the higher foot as reference, the swing foot raises the reference mid-stride,
  so the policy lifts the pelvis to hold the commanded height when it starts walking
  and drops it again when it stops. The stance foot is always planted, so walk and
  stand share the same height reference and the bobbing goes away.
  """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    root_z = asset.data.root_link_pos_w[:, 2]
    foot_z = site_pos(env)[:, slice(None), 2]
    base_height = root_z - torch.min(foot_z, dim=1).values
    height_error = torch.abs(base_height - command[:, 0] + ankle_sole_distance)
    return torch.exp(-height_error * 4.0)

def body_orientation_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg=_DEFAULT_ASSET_CFG, height_command_name: str | None=None, relax_below: float=0.3, full_above: float=0.55, relax_factor: float=0.3) -> torch.Tensor:
    """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.

  Optional height gating (``height_command_name``): scale the penalty down when
  a deep squat is commanded. Reaching toward the floor requires leaning the
  torso forward, but on the 23-DOF G1 there is no waist pitch -- torso pitch is
  the whole body rotating about the ankles -- so an ungated uprightness penalty
  opposes ground-reaching directly. The penalty keeps full strength while
  standing tall or walking (where uprightness is what keeps the robot alive) and
  falls to ``relax_factor`` for commanded heights at or below ``relax_below``,
  interpolating linearly in between.

  Note this PERMITS lean rather than creating it: static forward lean is
  ultimately bounded by keeping the CoM over the foot support polygon.
  """
    asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
        body_quat_w = body_quat_w.squeeze(1)
        gravity_w = asset.data.GRAVITY_VEC_W
        projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    else:
        xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    if height_command_name is not None:
        h_cmd = env.command_manager.get_command(height_command_name)
        if h_cmd is not None:
            span = max(full_above - relax_below, 1e-06)
            frac = ((h_cmd[:, 0] - relax_below) / span).clamp(0.0, 1.0)
            xy_squared = xy_squared * (relax_factor + (1.0 - relax_factor) * frac)
    return xy_squared

def body_angular_velocity_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Penalize excessive body angular velocities."""
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
    ang_vel = ang_vel.squeeze(1)
    ang_vel_xy = ang_vel[:, :2]
    return torch.sum(torch.square(ang_vel_xy), dim=1)

def feet_air_time(env: ManagerBasedRlEnv, sensor_name: str, threshold: float=0.4, command_name: str | None=None, command_threshold: float=0.1) -> torch.Tensor:
    """Reward feet air time."""
    sensor: ContactSensor = feet_sensor(env)
    sensor_data = sensor.data
    air_time = sensor_data.current_air_time
    contact_time = sensor_data.current_contact_time
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
    mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    error = torch.abs(mode_time - threshold)
    reward = torch.clamp(threshold - error, min=0.0)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward

def feet_clearance(env: ManagerBasedRlEnv, target_height: float, command_name: str | None=None, command_threshold: float=0.1, asset_cfg: SceneEntityCfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Penalize deviation from target clearance height, weighted by foot velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_z = site_pos(env)[:, slice(None), 2]
    foot_vel_xy = site_vel(env)[:, slice(None), :2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)
    delta = torch.abs(foot_z - target_height)
    cost = torch.sum(delta * vel_norm, dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost

def feet_gait(env: ManagerBasedRlEnv, period: float, offset: list[float], threshold: float, command_threshold: float, command_name: str, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = feet_sensor(env)
    is_contact = sensor.data.current_contact_time > 0
    global_phase = (env.episode_length_buf * env.step_dt / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = leg_phase < threshold
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward

def feet_slip(env: ManagerBasedRlEnv, sensor_name: str, command_name: str, command_threshold: float=0.01, always_active: bool=False, asset_cfg: SceneEntityCfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact).

  ``always_active`` removes the zero-command gate. By default this term is
  switched OFF while the robot stands still, which leaves sliding feet entirely
  unpenalized in exactly the regime where a stationary base matters most. A
  planted foot that slips is equally bad standing or walking, so for precise
  standing work this should be True; the default preserves the original
  behaviour for existing tasks.
  """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = feet_sensor(env)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = torch.ones_like(total_command) if always_active else (total_command > command_threshold).float()
    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()
    foot_vel_xy = site_vel(env)[:, slice(None), :2]
    vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)
    vel_xy_norm_sq = torch.square(vel_xy_norm)
    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
    num_in_contact = torch.sum(in_contact)
    mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(num_in_contact, min=1)
    env.extras['log']['Metrics/slip_velocity_mean'] = mean_slip_vel
    return cost

class variable_posture(ManagerTermBase):
    """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        asset: Entity = env.scene[cfg.params['asset_cfg'].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos
        _, joint_names = asset.find_joints(cfg.params['asset_cfg'].joint_names, preserve_order=True)
        _, _, std_standing = resolve_matching_names_values(data=cfg.params['std_standing'], list_of_strings=joint_names)
        self.std_standing = torch.tensor(std_standing, device=env.device, dtype=torch.float32)
        _, _, std_walking = resolve_matching_names_values(data=cfg.params['std_walking'], list_of_strings=joint_names)
        self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)
        _, _, std_running = resolve_matching_names_values(data=cfg.params['std_running'], list_of_strings=joint_names)
        self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)
        self.std_squatting = None
        if cfg.params.get('std_squatting'):
            _, _, std_squatting = resolve_matching_names_values(data=cfg.params['std_squatting'], list_of_strings=joint_names)
            self.std_squatting = torch.tensor(std_squatting, device=env.device, dtype=torch.float32)

    def __call__(self, env: ManagerBasedRlEnv, std_standing, std_walking, std_running, asset_cfg: SceneEntityCfg, command_name: str, walking_threshold: float=0.5, running_threshold: float=1.5, std_squatting=None, height_command_name: str | None=None, squat_below: float=0.35, full_above: float=0.6) -> torch.Tensor:
        del std_standing, std_walking, std_running, std_squatting
        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None
        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed
        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float()
        running_mask = (total_speed >= running_threshold).float()
        std = self.std_standing * standing_mask.unsqueeze(1) + self.std_walking * walking_mask.unsqueeze(1) + self.std_running * running_mask.unsqueeze(1)
        if self.std_squatting is not None and height_command_name is not None:
            h_cmd = env.command_manager.get_command(height_command_name)
            if h_cmd is not None:
                span = max(full_above - squat_below, 1e-06)
                alpha = ((full_above - h_cmd[:, 0]) / span).clamp(0.0, 1.0).unsqueeze(1)
                std = std * (1.0 - alpha) + self.std_squatting * alpha
        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)
        return torch.exp(-torch.mean(error_squared / std ** 2, dim=1))

def idle_base_motion(env: ManagerBasedRlEnv, command_name: str, command_threshold: float=0.1, asset_cfg: SceneEntityCfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Penalize base motion (xy lin vel + yaw rate) when no locomotion is commanded.

    This is the CoM-anchoring term for stationary manipulation: while the robot
    stands (possibly squatting, reaching, or holding a payload) the pelvis must
    stay planted. It (a) removes the post-stop drift where the robot needs an
    extra command to settle, and (b) forces the legs to counter arm-motion /
    payload-induced momentum in place instead of stepping to catch it. After a
    push it also drives the fastest possible return to rest.
    """
    asset: Entity = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_link_lin_vel_b
    ang_vel = asset.data.root_link_ang_vel_b
    lin_cost = torch.sum(torch.square(lin_vel[:, :2]), dim=1)
    ang_cost = 0.5 * torch.square(ang_vel[:, 2])
    command = env.command_manager.get_command(command_name)
    if command is not None:
        lin_idle = (torch.norm(command[:, :2], dim=1) <= command_threshold).float()
        yaw_idle = (torch.abs(command[:, 2]) <= command_threshold).float()
        return lin_cost * lin_idle + ang_cost * yaw_idle
    return lin_cost + ang_cost

class idle_position_anchor(ManagerTermBase):
    """Penalize base DISPLACEMENT from where the robot last came to a stop.

  Why this is not redundant with ``idle_base_motion``. That term penalizes
  velocity, quadratically: at a slow 0.05 m/s creep it costs 0.0025, which after
  the manager's dt scaling is negligible -- yet sustained for ten seconds it is
  half a metre of drift. Velocity penalties are simply the wrong instrument for
  position error, because the quantity that matters is the integral, and its
  gradient vanishes exactly where precision is needed.

  So this latches the base xy pose the moment the velocity command goes to zero
  and penalizes deviation from that anchor for as long as the robot stays idle.
  The cost is the L2 NORM (not its square), giving a constant gradient that keeps
  pushing toward zero error instead of going slack near it -- the same reasoning
  that makes exponential tracking rewards unsuitable for the last centimetre.

  This is what makes the robot hold station for precise manipulation, and in
  particular what forces the legs to reject the forward CoM shift when the arms
  extend -- a disturbance that otherwise expresses itself purely as slow drift
  and is therefore almost invisible to a velocity penalty.
  """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.anchor_xy = torch.zeros(env.num_envs, 2, device=env.device)
        self.anchor_yaw = torch.zeros(env.num_envs, device=env.device)
        self.was_lin_idle = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.was_yaw_idle = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.was_lin_idle[env_ids] = False
        self.was_yaw_idle[env_ids] = False

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, command_threshold: float=0.1, yaw_weight: float=0.5, max_error: float=1.0, xy_deadband: float=0.0, yaw_deadband: float=0.0, asset_cfg: SceneEntityCfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
        """See the class docstring.

    ``xy_deadband`` / ``yaw_deadband``: displacement inside the band costs
    nothing; outside it the cost grows linearly from zero. Why a deadband: with
    a zero-width anchor the cheapest way for the policy to absorb the forward
    CoM shift of extended arms was to keep the pelvis mathematically fixed and
    pitch the torso backward (the orientation term is quadratic and nearly free
    at small angles). A few centimetres of pelvis travel over stationary feet
    is the natural ankle/hip compensation; only sustained drift beyond the band
    is a problem for manipulation, and that is still penalised.
    """
        asset: Entity = env.scene[asset_cfg.name]
        pos_xy = asset.data.root_link_pos_w[:, :2]
        rot = matrix_from_quat(asset.data.root_link_quat_w)
        yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
        command = env.command_manager.get_command(command_name)
        assert command is not None
        lin_idle = torch.norm(command[:, :2], dim=1) <= command_threshold
        yaw_idle = torch.abs(command[:, 2]) <= command_threshold
        newly_lin = lin_idle & ~self.was_lin_idle
        self.anchor_xy[newly_lin] = pos_xy[newly_lin]
        newly_yaw = yaw_idle & ~self.was_yaw_idle
        self.anchor_yaw[newly_yaw] = yaw[newly_yaw]
        self.was_lin_idle = lin_idle
        self.was_yaw_idle = yaw_idle
        idle = lin_idle
        raw_dist = torch.norm(pos_xy - self.anchor_xy, dim=1)
        raw_dyaw = wrap_to_pi(yaw - self.anchor_yaw).abs()
        dist = (raw_dist - xy_deadband).clamp(min=0.0, max=max_error)
        dyaw = (raw_dyaw - yaw_deadband).clamp(min=0.0, max=max_error)
        idle_f = idle.float()
        log = env.extras.setdefault('log', {})
        log['Metrics/idle_drift_m'] = (raw_dist * idle_f).sum() / idle_f.sum().clamp(min=1)
        log['Metrics/idle_drift_max_m'] = (raw_dist * idle_f).max()
        return dist * idle_f + yaw_weight * dyaw * yaw_idle.float()

def idle_feet_still(env: ManagerBasedRlEnv, sensor_name: str, command_name: str, command_threshold: float=0.1, asset_cfg: SceneEntityCfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Keep both feet planted and still whenever no locomotion is commanded.

  Complements :class:`idle_position_anchor`, which only sees foot motion that
  actually moves the base. A shuffle step that lifts a foot and puts it back
  down in the same place leaves the anchor error unchanged, but it still breaks
  the fixed base frame that precise manipulation depends on, and it is what
  reads as the robot "shifting" while standing.

  Cost = per-foot horizontal speed + a unit cost for leaving the ground, gated
  to idle. Acting on foot MOTION rather than on joint deviation from a default
  pose means it does not fight a commanded squat.
  """
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = feet_sensor(env)
    foot_vel_xy = site_vel(env)[:, slice(None), :2]
    horiz_speed = torch.norm(foot_vel_xy, dim=-1)
    in_air = (sensor.data.current_contact_time <= 0.0).float()
    cost = torch.sum(horiz_speed + in_air, dim=1)
    command = env.command_manager.get_command(command_name)
    if command is not None:
        total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        cost = cost * (total_command <= command_threshold).float()
    return cost

def feet_lateral_distance(env: ManagerBasedRlEnv, min_distance: float=0.2, max_distance: float=0.35, asset_cfg: SceneEntityCfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Penalize a stance that is too narrow or too wide (HOMIE's feet_distance).

    Lateral (base-frame y) distance between the two foot sites, penalized
    linearly outside [min_distance, max_distance]. A too-narrow stance is what
    makes half-squat + arms-forward poses tippy; a too-wide stance blocks
    walking. Applied at all times (walking and standing).
    """
    asset: Entity = env.scene[asset_cfg.name]
    feet_pos = site_pos(env)[:, slice(None), :]
    diff_w = feet_pos[:, 0, :] - feet_pos[:, 1, :]
    diff_b = quat_apply_inverse(asset.data.root_link_quat_w, diff_w)
    lateral = torch.abs(diff_b[:, 1])
    too_narrow = torch.clamp(min_distance - lateral, min=0.0)
    too_wide = torch.clamp(lateral - max_distance, min=0.0)
    return too_narrow + too_wide

def _com_support_ratio(env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg, foot_half_length: float, foot_half_width: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised distance of the whole-body CoM from the double-support region.

  Returns ``(r, double_support)``: ``r`` is the CoM's horizontal offset from
  the midpoint between the feet, expressed in the base yaw frame and divided by
  the region's half-extents, so ``r < 1`` is inside the region and ``r = 0`` is
  its centre. The region is a conservative ellipse: fore-aft half-extent =
  ``foot_half_length`` + half the fore-aft stagger of the feet, lateral
  half-extent = half the lateral foot separation + ``foot_half_width``. That
  is smaller than the true support polygon (an ellipse inscribed in the
  rectangle spanned by the two feet), which is what we want from a safety
  margin, and needs no polygon API.

  The CoM is MuJoCo's ``subtree_com`` of the root body: the mass-weighted
  centre of every link including the arms, hands and any randomised payload
  mass -- not the pelvis. That is the whole point of the term: when the arms
  extend, this moves and the pelvis does not.
  """
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = feet_sensor(env)
    com_xy = whole_com(env)[:, :2]
    feet = site_pos(env)[:, slice(None), :2]
    assert sensor.data.found is not None
    in_contact = sensor.data.found > 0
    double_support = in_contact.sum(dim=1) == 2
    yaw = asset.data.heading_w
    c, s_ = (torch.cos(yaw), torch.sin(yaw))
    d = com_xy - feet.mean(dim=1)
    dx = c * d[:, 0] + s_ * d[:, 1]
    dy = -s_ * d[:, 0] + c * d[:, 1]
    fd = feet[:, 0] - feet[:, 1]
    stagger = (c * fd[:, 0] + s_ * fd[:, 1]).abs()
    separation = (-s_ * fd[:, 0] + c * fd[:, 1]).abs()
    half_x = foot_half_length + 0.5 * stagger
    half_y = foot_half_width + 0.5 * separation
    r = torch.sqrt(torch.square(dx / half_x) + torch.square(dy / half_y) + 1e-08)
    return (r, double_support)

def com_support_region(env: ManagerBasedRlEnv, sensor_name: str, command_name: str, foot_half_length: float=0.09, foot_half_width: float=0.04, safe_fraction: float=0.5, outside_gain: float=2.0, moving_scale: float=0.25, command_threshold: float=0.1, asset_cfg: SceneEntityCfg=_DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Penalise the whole-body CoM leaving the double-support region.

  Cost as a function of the normalised offset ``r`` (see
  :func:`_com_support_ratio`)::

      r <= safe_fraction : 0                                   (comfortably inside)
      safe_fraction..1   : (r - safe_fraction)^2               (approaching the edge)
      r > 1              : ... + outside_gain * (r - 1)        (outside: strong)

  Evaluated in double support only. In single support the CoM is legitimately
  outside the stance foot during a step, so penalising it there would fight
  walking; foot placement handles balance in that regime. While a locomotion
  command is active the cost is scaled by ``moving_scale`` -- the term exists
  mainly for stationary manipulation, and a walking robot's CoM oscillates
  around the support centre by design. Together with the pelvis deadband in
  ``idle_position_anchor`` this is what lets the policy answer an arm-induced
  CoM shift with a small pelvis/ankle adjustment instead of a torso lean:
  keeping the CoM centred is now rewarded directly, and the cheap way to do it
  (shift the pelvis a couple of centimetres) is no longer penalised.
  """
    r, double_support = _com_support_ratio(env, sensor_name, asset_cfg, foot_half_length, foot_half_width)
    cost = torch.square((r - safe_fraction).clamp(min=0.0)) + outside_gain * (r - 1.0).clamp(min=0.0)
    cost = cost * double_support.float()
    command = env.command_manager.get_command(command_name)
    if command is not None:
        total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        moving = total_command > command_threshold
        cost = cost * torch.where(moving, moving_scale, 1.0)
    return cost

def action_smoothness_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize the SECOND difference of the actions (a jerk proxy).

  ``sum((a_t - 2 a_{t-1} + a_{t-2})^2)`` over the action dimensions.

  Why this and not a larger ``action_rate_l2``. Both suppress high-frequency
  content, but they weight it differently: a first difference has frequency
  response ``4 sin^2(w/2)``, a second difference ``16 sin^4(w/2)``. The second
  is far more concentrated at the top of the band, so it removes visible
  twitch while leaving the smooth, fast action changes a swing leg legitimately
  needs. Raising ``action_rate_l2`` instead damps the whole gait.

  Measured motivation (run arm_robust_v4, iteration 12016): the second
  difference averaged 0.423 per dimension with NO reward term acting on it --
  it was logged as ``mean_action_acc`` and never penalized. Meanwhile 56% of
  the ``action_rate_l2`` cost was exploration noise, which the deployed and
  play-mode policies never emit because inference uses the distribution mean.
  So the term that was paying for smoothness was mostly paying for something
  invisible at deployment, and the quantity that actually reads as jitter was
  unpriced.

  Note the reset behaviour matches ``action_rate_l2``: the action history is
  zeroed on reset, so the first step after a reset sees a full-magnitude
  difference. That is pre-existing, identical for both terms, and washes out
  over an episode.
  """
    acc = env.action_manager.action - 2.0 * env.action_manager.prev_action + env.duet_prev_prev_action
    return torch.sum(torch.square(acc), dim=1)
