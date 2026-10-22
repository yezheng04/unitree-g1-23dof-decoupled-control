"""Pure, testable guards for the experimental G1 hardware controller."""
from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np

from .common import SDK_INDICES, finite_vector

L1, START, SELECT, A, B = 1, 2, 3, 8, 9


@dataclass(frozen=True)
class Remote:
    keys: int
    lx: float = 0.
    ly: float = 0.
    rx: float = 0.

    def pressed(self, button):
        return bool(self.keys & (1 << button))

    @classmethod
    def decode(cls, data):
        raw = bytes(data)
        if len(raw) != 40:
            raise ValueError('Expected 40 wireless remote bytes')
        keys = struct.unpack_from('<H', raw, 2)[0]
        lx, rx = struct.unpack_from('<ff', raw, 4)
        ly = struct.unpack_from('<f', raw, 20)[0]
        axes = finite_vector([lx, ly, rx], 3, 'remote axes')
        if np.max(np.abs(axes)) > 1.05:
            raise ValueError('Remote axis outside [-1, 1]')
        return cls(keys, *np.clip(axes, -1., 1.))


@dataclass(frozen=True)
class State:
    q: np.ndarray
    dq: np.ndarray
    omega: np.ndarray
    gravity: np.ndarray
    remote: Remote
    mode_machine: int
    tick: int
    received: float
    temperature: float = 0.


class Watchdog:
    """Faults are latched; a fresh message never re-enables a failed controller."""
    def __init__(self, contract, mode_machine, timeout=.1):
        self.contract = contract
        self.mode_machine = mode_machine
        self.timeout = timeout
        self.reason = None

    def trip(self, reason):
        if self.reason is None:
            self.reason = reason
        return self.reason

    def check(self, state, target_time, now, engaged):
        if self.reason:
            return self.reason
        if state is None or not 0 <= now-state.received <= self.timeout:
            return self.trip('lowstate_timeout')
        if not 0 <= now-target_time <= self.timeout:
            return self.trip('control_timeout')
        if state.mode_machine != self.mode_machine:
            return self.trip('mode_machine_changed')
        if state.remote.pressed(SELECT) or state.remote.pressed(B):
            return self.trip('operator_stop')
        if engaged and not state.remote.pressed(L1):
            return self.trip('deadman_released')
        arrays = (state.q, state.dq, state.omega, state.gravity)
        if not all(np.isfinite(a).all() for a in arrays):
            return self.trip('nonfinite_state')
        if -state.gravity[2] < np.cos(np.deg2rad(45.)):
            return self.trip('excessive_tilt')
        c = self.contract
        if np.any(state.q < c.lower-.05) or np.any(state.q > c.upper+.05):
            return self.trip('measured_joint_limit')
        if np.any(np.abs(state.dq) > c.velocity * 1.2):
            return self.trip('measured_joint_speed')
        if not np.isfinite(state.temperature) or state.temperature >= 70.:
            return self.trip('motor_temperature')
        return None


def guarded_target(contract, target, previous, dt):
    target = finite_vector(target, 23, 'target')
    previous = finite_vector(previous, 23, 'previous target')
    if not 0 < dt <= .05:
        raise ValueError('Control deadline missed')
    # Runtime guard, deliberately narrower than the unconstrained training action.
    target = np.clip(target, contract.lower+.02, contract.upper-.02)
    max_step = np.minimum(contract.velocity, np.r_[np.full(13, 8.), np.full(10, 2.)]) * dt
    return previous + np.clip(target-previous, -max_step, max_step)


def fill_command(message, contract, target, state, damping=False):
    """Only the 23 active HG slots are enabled; other motors remain disabled."""
    target = finite_vector(target, 23, 'target')
    message.mode_pr = 0  # PR mode, never AB motor coordinates.
    message.mode_machine = state.mode_machine
    for motor in message.motor_cmd:
        motor.mode = 0
        motor.q = motor.dq = motor.tau = motor.kp = motor.kd = 0.
    if damping:
        kp = np.zeros(23)
        kd = np.minimum(contract.kd, 2.)
        target = np.zeros(23)
    else:
        kp, kd = contract.kp, contract.kd
        # Bound the PD torque estimate using fresh q/dq. Firmware protection is
        # still required: this is not a guaranteed torque limit between samples.
        target = np.clip(target, state.q + (kd*state.dq-contract.effort)/kp,
                         state.q + (kd*state.dq+contract.effort)/kp)
        if np.any(target < contract.lower) or np.any(target > contract.upper):
            raise ValueError('No admissible position target within PD torque bound')
    for i, slot in enumerate(SDK_INDICES):
        m = message.motor_cmd[slot]
        m.mode, m.q, m.dq, m.tau = 1, float(target[i]), 0., 0.
        m.kp, m.kd = float(kp[i]), float(kd[i])
    return message
