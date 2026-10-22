"""Run real MuJoCo physics with the exported policy (not a recorded-pose replay)."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .common import Contract, JOINT_NAMES, Policy, ROOT, pelvis_imu, sha256
from .safety import guarded_target


def build_model(contract, mjcf, dt=.002):
    mjcf = Path(mjcf).resolve()
    if sha256(mjcf) != contract.data['mjcf_sha256']:
        raise ValueError('MJCF differs from the exported contract')
    if dt <= 0 or not np.isclose(contract.dt / dt, round(contract.dt / dt)):
        raise ValueError('Physics dt must divide the 0.02 s policy interval')
    xml = ET.parse(mjcf).getroot()
    compiler = xml.find('compiler')
    compiler.set('meshdir', str(mjcf.parent / compiler.get('meshdir', '.')))
    option = xml.find('option')
    if option is None:
        option = ET.SubElement(xml, 'option')
    option.set('timestep', str(dt))
    option.set('integrator', 'implicitfast')
    option.set('iterations', '100')
    option.set('gravity', '0 0 -9.81')
    # Match the training contact exclusions; retain the pinned Rev1.0 geometry.
    contact = ET.SubElement(xml, 'contact')
    for a, b in [('pelvis', 'left_hip_roll_link'), ('pelvis', 'right_hip_roll_link'),
                 ('left_elbow_link', 'left_wrist_roll_rubber_hand'),
                 ('right_elbow_link', 'right_wrist_roll_rubber_hand')]:
        ET.SubElement(contact, 'exclude', body1=a, body2=b)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    ids = np.array([model.joint(n).id for n in JOINT_NAMES])
    qadr, dadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    motors = np.array([model.actuator(n).id for n in JOINT_NAMES])
    if model.nq != 30 or model.nv != 29 or model.nu != 23:
        raise ValueError('Expected a free-base G1 23DoF model')
    if not np.array_equal(model.actuator_trnid[motors, 0], ids):
        raise ValueError('Actuator/joint mapping mismatch')
    model.dof_armature[dadr] = contract.armature
    model.dof_damping[dadr] = 0  # kd is explicitly applied below.
    model.jnt_range[ids] = np.column_stack([contract.lower, contract.upper])
    model.jnt_actfrcrange[ids] = np.column_stack([-contract.effort, contract.effort])
    model.jnt_actfrclimited[ids] = True
    model.actuator_ctrllimited[motors] = True
    model.actuator_ctrlrange[motors] = np.column_stack([-contract.effort, contract.effort])
    model.geom_friction[:, 0] = 1.
    return model, qadr, dadr, motors


def run(a):
    contract = Contract(a.bundle/'deploy.json')
    policy = Policy(contract)
    model, qadr, dadr, motors = build_model(contract, a.mjcf, a.physics_dt)
    data = mujoco.MjData(model)
    data.qpos[:3] = [0., 0., .8]
    data.qpos[3:7] = [1., 0., 0., 0.]
    data.qpos[qadr] = contract.q_default
    mujoco.mj_forward(model, data)
    last = np.zeros(13)
    previous_target = contract.q_default.copy()
    observations, actions, rows = [], [], []
    survived = True
    failure = None
    measurement_start = a.settle + 3.
    substeps = round(contract.dt / a.physics_dt)
    if not a.headless:
        from mujoco import viewer as mj_viewer
        viewer_context = mj_viewer.launch_passive(model, data)
    else:
        viewer_context = nullcontext(None)
    with viewer_context as viewer:
        for step in range(round(a.duration / contract.dt)):
            wall_start = time.monotonic()
            t = step * contract.dt
            command = np.array([a.vx, a.vy, a.yaw]) if t >= a.settle else np.zeros(3)
            height = a.height if t >= a.settle else .78
            # Freejoint angular qvel is expressed in the root/body frame.
            omega, gravity = pelvis_imu(data.qpos[3:7], data.qvel[3:6])
            obs = contract.observations(data.qpos[qadr], data.qvel[dadr], omega,
                                        gravity, command, height, last, step)
            action = policy(obs)
            arms = contract.q_default[13:].copy()
            if a.arm_motion and t >= a.settle:
                arm_t = t - a.settle
                arms += .15 * np.sin(2*np.pi*.25*arm_t + np.arange(10)*.2) * min(arm_t, 1.)
            target = contract.targets(action, arms)
            if a.hardware_limits:
                target = guarded_target(contract, target, previous_target, contract.dt)
            previous_target = target.copy()
            for _ in range(substeps):
                torque = contract.kp*(target-data.qpos[qadr]) - contract.kd*data.qvel[dadr]
                data.ctrl[motors] = np.clip(torque, -contract.effort, contract.effort)
                mujoco.mj_step(model, data)
            last = action
            _, gravity_after = pelvis_imu(data.qpos[3:7], data.qvel[3:6])
            tilt = float(np.arccos(np.clip(-gravity_after[2], -1., 1.)))
            # Root translational freejoint velocity is in world coordinates.
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            velocity = rotation.reshape(3, 3).T @ data.qvel[:3]
            rows.append([data.time, *velocity, data.qvel[5], data.qpos[2], tilt])
            if a.trace:
                observations.append(obs)
                actions.append(action)
            if not np.isfinite(data.qpos).all() or tilt > np.pi/3 or data.qpos[2] < .18:
                survived = False
                failure = 'fall_or_nonfinite_state'
                break
            if viewer:
                if not viewer.is_running():
                    survived, failure = False, 'viewer_closed_before_completion'
                    break
                viewer.sync()
                time.sleep(max(0., contract.dt - (time.monotonic()-wall_start)))
    rows = np.asarray(rows)
    measured = rows[rows[:, 0] >= measurement_start]
    result = dict(backend='MuJoCo physics (mj_step)', mujoco_version=mujoco.__version__,
                  policy_sha256=contract.data['policy_sha256'], checkpoint_sha256=contract.data['checkpoint_sha256'],
                  duration_requested=a.duration, duration_completed=float(data.time), survived=survived,
                  failure=failure, command=[a.vx, a.vy, a.yaw], height_command=a.height,
                  arm_motion=a.arm_motion, physics_dt=a.physics_dt, control_dt=contract.dt,
                  hardware_position_limits=a.hardware_limits, measurement_start=measurement_start,
                  mean_vx_vy_yaw=measured[:, [1, 2, 4]].mean(0).tolist() if len(measured) else None,
                  mean_root_height=float(measured[:, 5].mean()) if len(measured) else None,
                  max_tilt_rad=float(rows[:, 6].max()), hardware_validated=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    if a.trace:
        np.savez_compressed(a.output.with_suffix('.npz'), observations=observations, actions=actions, states=rows)
    print(json.dumps(result, indent=2))
    return survived


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT/'artifacts/deploy/22000')
    p.add_argument('--mjcf', type=Path, default=ROOT/'assets/g1/g1_23dof_rev_1_0.xml')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--duration', type=float, default=18.)
    p.add_argument('--settle', type=float, default=3.)
    p.add_argument('--physics-dt', type=float, default=.002)
    p.add_argument('--vx', type=float, default=0.)
    p.add_argument('--vy', type=float, default=0.)
    p.add_argument('--yaw', type=float, default=0.)
    p.add_argument('--height', type=float, default=.78)
    p.add_argument('--arm-motion', action='store_true')
    p.add_argument('--hardware-limits', action='store_true', help='Apply hardware position and target-rate guards; changes policy targets')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/sim2sim/result.json')
    a = p.parse_args()
    values = [a.duration, a.settle, a.physics_dt, a.vx, a.vy, a.yaw, a.height]
    if not np.isfinite(values).all() or a.duration <= 0 or not 0 <= a.settle < a.duration:
        p.error('Invalid duration, settling time or non-finite argument')
    if not (-1 <= a.vx <= 1 and abs(a.vy) <= .5 and abs(a.yaw) <= 1 and .24 <= a.height <= .78):
        p.error('Command exceeds the training range')
    raise SystemExit(0 if run(a) else 2)


if __name__ == '__main__':
    main()
