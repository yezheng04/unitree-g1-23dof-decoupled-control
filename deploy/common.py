"""The shared observation/action contract for MuJoCo and Unitree HG DDS."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
JOINT_NAMES = [f'{s}_{j}_joint' for s in ('left', 'right') for j in
               ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
JOINT_NAMES += ['waist_yaw_joint']
JOINT_NAMES += [f'{s}_{j}_joint' for s in ('left', 'right') for j in
                ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll')]
SDK_INDICES = list(range(13)) + list(range(15, 20)) + list(range(22, 27))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_vector(value, size, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected {size} finite values')
    return a


def rotation_wxyz(quaternion):
    q = finite_vector(quaternion, 4, 'quaternion')
    norm = np.linalg.norm(q)
    if not 0.9 < norm < 1.1:
        raise ValueError('IMU quaternion is not normalized')
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def pelvis_imu(quaternion, omega, waist_q=0., waist_dq=0., frame='pelvis'):
    """Quaternion is body->world, wxyz. Gyroscope is in its own body frame."""
    rotation = rotation_wxyz(quaternion)
    omega = finite_vector(omega, 3, 'gyro')
    if frame == 'torso':
        c, s = np.cos(waist_q), np.sin(waist_q)
        rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        rotation = rotation @ rz.T
        omega = rz @ omega - np.array([0., 0., waist_dq])
    elif frame != 'pelvis':
        raise ValueError('IMU frame must be pelvis or torso')
    return omega, rotation.T @ np.array([0., 0., -1.])


class Contract:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.data = json.loads(self.path.read_text())
        d = self.data
        if d.get('schema_version') != 1 or d.get('joint_names') != JOINT_NAMES:
            raise ValueError('Unsupported contract or joint order')
        if d.get('sdk_indices') != SDK_INDICES or (d.get('actor_dim'), d.get('action_dim')) != (71, 13):
            raise ValueError('Expected the G1 EDU 23DoF 71-observation/13-action contract')
        if d.get('control_dt') != .02 or d.get('normalization_eps') != .01:
            raise ValueError('Unexpected inference timing or normalization')
        for key in ('q_default', 'kp', 'kd', 'armature', 'lower', 'upper', 'effort', 'velocity'):
            setattr(self, key, finite_vector(d[key], 23, key))
        self.scale = finite_vector(d['action_scale'], 13, 'action_scale')
        if (np.any(self.lower >= self.upper) or np.any(self.q_default < self.lower)
                or np.any(self.q_default > self.upper)):
            raise ValueError('Invalid position limits or default pose')
        if any(np.any(getattr(self, key) <= 0) for key in ('kp', 'kd', 'effort', 'velocity', 'armature')) or np.any(self.scale <= 0):
            raise ValueError('Control parameters must be positive')
        self.dt = .02

    def observations(self, q, dq, omega, gravity, command, height, last_action, step):
        q = finite_vector(q, 23, 'q')
        dq = finite_vector(dq, 23, 'dq')
        command = finite_vector(command, 3, 'command')
        omega = finite_vector(omega, 3, 'gyro')
        gravity = finite_vector(gravity, 3, 'gravity')
        last_action = finite_vector(last_action, 13, 'last action')
        if not np.isfinite(height) or not .24 <= height <= .78:
            raise ValueError('Height command must lie in [0.24, 0.78] m')
        angle = 2 * np.pi * ((step * self.dt) % .6) / .6
        phase = [np.sin(angle), np.cos(angle)] if np.linalg.norm(command) >= .1 else [0., 0.]
        # Same term order and unit scales as common/scene_cfg.py + PolicyCfg.
        return np.concatenate([omega, gravity, command, phase, q-self.q_default,
                               dq, last_action, [height]]).astype(np.float32)

    def targets(self, action, arms=None):
        target = self.q_default.copy()
        target[:13] += self.scale * finite_vector(action, 13, 'action')
        if arms is not None:
            target[13:] = finite_vector(arms, 10, 'arm targets')
        return target


class Policy:
    def __init__(self, contract: Contract):
        self.contract = contract
        path = contract.path.parent / contract.data['policy_file']
        if sha256(path) != contract.data['policy_sha256']:
            raise ValueError('Policy hash differs from the deployment contract; re-export')
        torch.set_num_threads(1)
        self.module = torch.jit.load(str(path), map_location='cpu').eval()
        # Warm up before any hardware publisher is created.
        for _ in range(3):
            self(np.zeros(71, dtype=np.float32))

    def __call__(self, observation):
        observation = finite_vector(observation, 71, 'observation').astype(np.float32)
        with torch.inference_mode():
            out = self.module(torch.from_numpy(observation).unsqueeze(0))
        return finite_vector(out.squeeze(0).numpy(), 13, 'policy output')
