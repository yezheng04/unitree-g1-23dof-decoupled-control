"""Experimental G1 EDU 23DoF HG DDS deployment; offline dry-run is the default."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np

from .common import Contract, Policy, ROOT, SDK_INDICES, pelvis_imu
from .safety import A, B, L1, SELECT, START, Remote, State, Watchdog, fill_command, guarded_target


class Receiver:
    def __init__(self, frame, crc):
        self.frame, self.crc = frame, crc
        self.lock = threading.Lock()
        self.state = None
        self.rejected = 0

    def callback(self, msg):
        try:
            if self.crc.Crc(msg) != msg.crc or msg.mode_pr != 0:
                raise ValueError('Bad CRC or non-PR state')
            q = np.array([msg.motor_state[i].q for i in SDK_INDICES])
            dq = np.array([msg.motor_state[i].dq for i in SDK_INDICES])
            omega, gravity = pelvis_imu(msg.imu_state.quaternion, msg.imu_state.gyroscope,
                                        q[12], dq[12], self.frame)
            state = State(q, dq, omega, gravity, Remote.decode(msg.wireless_remote),
                          int(msg.mode_machine), int(msg.tick), time.monotonic(),
                          max(max(msg.motor_state[i].temperature) for i in SDK_INDICES))
            if not np.isfinite(np.concatenate([q, dq])).all():
                raise ValueError('Non-finite motor state')
            with self.lock:
                if self.state is not None:
                    delta = (state.tick-self.state.tick) & 0xffffffff
                    if delta == 0 or delta >= 0x80000000:
                        return  # Repeated/out-of-order ticks do not refresh the watchdog.
                self.state = state
        except (ValueError, TypeError, IndexError, OverflowError):
            self.rejected += 1

    def snapshot(self):
        with self.lock:
            return self.state


class Publisher:
    """One lowcmd writer at 250 Hz, with a watchdog independent of 50 Hz inference."""
    def __init__(self, contract, receiver, channel, message, crc, state):
        self.contract, self.receiver, self.channel = contract, receiver, channel
        self.message, self.crc = message, crc
        self.guard = Watchdog(contract, state.mode_machine)
        self.lock = threading.Lock()
        self.target = state.q.copy()
        self.target_time = time.monotonic()
        self.engaged = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name='g1-lowcmd', daemon=True)
        self.publish_error = None

    def update(self, target, engaged):
        with self.lock:
            self.target = target.copy()
            self.target_time = time.monotonic()
            self.engaged = engaged

    def run(self):
        deadline = time.monotonic()
        while not self.stop.is_set():
            try:
                now = time.monotonic()
                if now-deadline > .02:
                    self.guard.trip('publisher_deadline')
                state = self.receiver.snapshot()
                with self.lock:
                    target, stamp, engaged = self.target.copy(), self.target_time, self.engaged
                fault = self.guard.check(state, stamp, time.monotonic(), engaged)
                if state is not None:
                    try:
                        fill_command(self.message, self.contract, target, state, damping=bool(fault) or not engaged)
                    except ValueError as exc:
                        self.guard.trip(str(exc))
                        fill_command(self.message, self.contract, np.zeros(23), state, damping=True)
                    self.message.crc = self.crc.Crc(self.message)
                    if self.channel.Write(self.message) is False:
                        raise RuntimeError('DDS write failed')
            except Exception as exc:
                self.guard.trip(f'publisher_error: {exc}')
                self.publish_error = str(exc)
                break
            deadline += .004
            self.stop.wait(max(0., deadline-time.monotonic()))

    def close(self):
        self.guard.trip('controller_exit')
        # Best effort damping while DDS is available. Power loss/process kill
        # cannot be handled in software; the physical stop/harness remain needed.
        time.sleep(.5)
        self.stop.set()
        self.thread.join(timeout=1.)


def dry_run(contract, policy):
    last = np.zeros(13)
    for step in range(100):
        obs = contract.observations(contract.q_default, np.zeros(23), np.zeros(3),
                                   [0., 0., -1.], [0., 0., 0.], .78, last, step)
        last = policy(obs)
        guarded_target(contract, contract.targets(last), contract.q_default, contract.dt)
    print(json.dumps({'mode': 'offline_dry_run', 'inference_steps': 100, 'dds_initialized': False,
                      'sdk_indices': SDK_INDICES, 'policy_sha256': contract.data['policy_sha256'],
                      'hardware_validated': False}, indent=2))


def connect(a):
    # Imported only after explicit --read-only or --execute. Default is network-free.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.utils.crc import CRC
    ChannelFactoryInitialize(0, a.interface)
    receiver = Receiver(a.imu_frame, CRC())
    subscriber = ChannelSubscriber('rt/lowstate', LowState_)
    subscriber.Init(receiver.callback, 10)
    deadline = time.monotonic()+10.
    while receiver.snapshot() is None:
        if time.monotonic() > deadline:
            raise TimeoutError('No valid HG PR lowstate after 10 s; check NIC, mode and firmware')
        time.sleep(.01)
    return receiver, subscriber


def ensure_motion_available(switcher, release_mode):
    code, mode = switcher.CheckMode()
    if code != 0 or not isinstance(mode, dict) or 'name' not in mode:
        raise RuntimeError('Cannot establish motion service state; no lowcmd publisher started')
    if mode['name']:
        if not release_mode:
            raise RuntimeError('Motion service is active. Enter developer low-level mode or use --release-mode explicitly')
        code, _ = switcher.ReleaseMode()
        if code != 0:
            raise RuntimeError(f'ReleaseMode failed: {code}')
        time.sleep(1.)
        code, mode = switcher.CheckMode()
        if code != 0 or mode.get('name'):
            raise RuntimeError('Motion service still active; refusing a competing writer')


def execute(a, contract, policy, receiver):
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    switcher = MotionSwitcherClient()
    switcher.SetTimeout(3.)
    switcher.Init()
    ensure_motion_available(switcher, a.release_mode)
    state = receiver.snapshot()
    guard = Watchdog(contract, state.mode_machine)
    reason = guard.check(state, time.monotonic(), time.monotonic(), False)
    if reason:
        raise RuntimeError(f'Initial state rejected: {reason}')
    channel = ChannelPublisher('rt/lowcmd', LowCmd_)
    channel.Init()
    publisher = Publisher(contract, receiver, channel, unitree_hg_msg_dds__LowCmd_(), CRC(), state)
    phase, previous_keys = 'WAIT', state.remote.keys
    target = state.q.copy()
    last_action, command = np.zeros(13), np.zeros(3)
    start = time.monotonic()
    deadline = start
    policy_step = 0
    publisher.thread.start()
    print('WAIT: hold L1 + press Start to interpolate to default pose; keep L1 held. A starts policy after READY. B/Select stops.', flush=True)
    try:
        while time.monotonic()-start < a.duration:
            now = time.monotonic()
            if now-deadline > .03:
                raise RuntimeError('50 Hz inference deadline missed')
            state = receiver.snapshot()
            fault = publisher.guard.check(state, publisher.target_time, time.monotonic(), phase != 'WAIT')
            if fault:
                raise RuntimeError(f'Latched stop: {fault}')
            remote = state.remote
            rising = remote.keys & ~previous_keys
            previous_keys = remote.keys
            if phase == 'WAIT' and remote.pressed(L1) and rising & (1 << START):
                ramp_start, ramp_q = now, state.q.copy()
                target = ramp_q.copy()
                phase = 'RAMP'
                print('RAMP: interpolating for 3 s', flush=True)
            if phase == 'RAMP':
                alpha = np.clip((now-ramp_start)/3., 0., 1.)
                blend = alpha*alpha*(3.-2.*alpha)
                target = guarded_target(contract, ramp_q*(1-blend)+contract.q_default*blend, target, contract.dt)
                if alpha >= 1.:
                    phase = 'READY'
                    print('READY: feet on floor, hold L1 and press A to enable policy', flush=True)
            elif phase == 'READY':
                target = guarded_target(contract, contract.q_default, target, contract.dt)
                if remote.pressed(L1) and rising & (1 << A):
                    if np.max(np.abs(state.q-contract.q_default)) > .3:
                        raise RuntimeError('Default pose tracking error too large to start policy')
                    phase = 'POLICY'
                    policy_step = 0
                    print('POLICY: left stick forward/lateral, right stick yaw; release L1 or B/Select for latched damping stop', flush=True)
            if phase == 'POLICY':
                axes = np.array([remote.ly, -remote.lx, -remote.rx])
                axes[np.abs(axes) < .1] = 0.
                desired = axes * np.array([.3, .15, .3])
                command += np.clip(desired-command, -contract.dt*np.array([.25, .25, .4]), contract.dt*np.array([.25, .25, .4]))
                obs = contract.observations(state.q, state.dq, state.omega, state.gravity,
                                            command, a.height, last_action, policy_step)
                action = policy(obs)
                target = guarded_target(contract, contract.targets(action), target, contract.dt)
                last_action = action
                policy_step += 1
            publisher.update(target, phase != 'WAIT')
            deadline += contract.dt
            time.sleep(max(0., deadline-time.monotonic()))
    finally:
        publisher.close()
        channel.Close()
        print('Stopped. Damping was requested; support the robot and use its physical stop. The motion service is not re-enabled automatically.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT/'artifacts/deploy/22000')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--read-only', action='store_true', help='Subscribe to state only; no lowcmd publisher')
    modes.add_argument('--execute', action='store_true', help='Enable the lowcmd publisher and operator-controlled state machine')
    p.add_argument('--interface', help='Explicit Ethernet interface connected to the robot')
    p.add_argument('--imu-frame', choices=['pelvis', 'torso'], help='Confirm the frame of lowstate IMU before live use')
    p.add_argument('--release-mode', action='store_true', help='Explicitly release the active motion service before control')
    p.add_argument('--height', type=float, default=.78)
    p.add_argument('--duration', type=float, default=60.)
    a = p.parse_args()
    if not np.isfinite([a.duration, a.height]).all() or a.duration <= 0 or not .6 <= a.height <= .78:
        p.error('Hardware duration must be positive and height must lie in [0.60, 0.78] m')
    if a.release_mode and not a.execute:
        p.error('--release-mode requires --execute')
    if (a.execute or a.read_only) and (not a.interface or not a.imu_frame):
        p.error('Live use requires both --interface and --imu-frame')
    contract = Contract(a.bundle/'deploy.json')
    policy = Policy(contract)
    if not a.execute and not a.read_only:
        dry_run(contract, policy)
        return
    receiver, subscriber = connect(a)
    try:
        if a.read_only:
            deadline = time.monotonic()+a.duration
            while time.monotonic() < deadline:
                state = receiver.snapshot()
                if time.monotonic()-state.received > .1:
                    raise TimeoutError('State stream stopped')
                print(json.dumps({'tick': state.tick, 'mode_machine': state.mode_machine,
                                  'q': state.q.tolist(), 'gravity': state.gravity.tolist(),
                                  'gyro': state.omega.tolist(), 'remote_keys': state.remote.keys,
                                  'temperature_max': state.temperature, 'rejected_packets': receiver.rejected}), flush=True)
                time.sleep(.5)
        else:
            execute(a, contract, policy, receiver)
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        subscriber.Close()


if __name__ == '__main__':
    main()
