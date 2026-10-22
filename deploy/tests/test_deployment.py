"""CPU contract and fault-injection tests; no DDS network or robot required."""
import json
from pathlib import Path
import struct
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from deploy.common import Contract, JOINT_NAMES, SDK_INDICES, pelvis_imu
from deploy.export_policy import Actor
from deploy.safety import L1, SELECT, Remote, State, Watchdog, fill_command, guarded_target
from deploy.sim2real import Publisher, Receiver, ensure_motion_available


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'deploy.json'
        d = dict(schema_version=1, joint_names=JOINT_NAMES, sdk_indices=SDK_INDICES,
                 actor_dim=71, action_dim=13, control_dt=.02, normalization_eps=.01,
                 q_default=[.1]*23, kp=[40.]*23, kd=[2.]*23, armature=[.01]*23,
                 lower=[-2.]*23, upper=[2.]*23, effort=[30.]*23, velocity=[20.]*23,
                 action_scale=[.5]*13)
        self.path.write_text(json.dumps(d))
        self.c = Contract(self.path)

    def state(self, **kw):
        fields = dict(q=self.c.q_default.copy(), dq=np.zeros(23), omega=np.zeros(3),
                      gravity=np.array([0., 0., -1.]), remote=Remote(1 << L1),
                      mode_machine=5, tick=1, received=10., temperature=30.)
        fields.update(kw)
        return State(**fields)

    def test_observation_layout_and_phase(self):
        obs = self.c.observations(np.arange(23)*.01, np.arange(23), [1, 2, 3], [0, 0, -1],
                                  [.5, 0, 0], .6, np.arange(13), 5)
        self.assertEqual(obs.shape, (71,))
        np.testing.assert_allclose(obs[:9], [1, 2, 3, 0, 0, -1, .5, 0, 0])
        np.testing.assert_allclose(obs[9:11], [np.sin(np.pi/3), .5])
        np.testing.assert_allclose(obs[11:34], np.arange(23)*.01-.1, atol=1e-7)
        np.testing.assert_allclose(obs[34:57], np.arange(23))
        np.testing.assert_allclose(obs[57:70], np.arange(13))
        self.assertAlmostEqual(obs[70], .6)
        obs2 = self.c.observations(self.c.q_default, np.zeros(23), [0]*3, [0, 0, -1], [.09, 0, 0], .78, [0]*13, 5)
        np.testing.assert_equal(obs2[9:11], [0, 0])

    def test_imu_torso_yaw_removal(self):
        yaw = .7
        omega, gravity = pelvis_imu([np.cos(yaw/2), 0, 0, np.sin(yaw/2)], [0, 0, .4], yaw, .4, 'torso')
        np.testing.assert_allclose(omega, 0., atol=1e-10)
        np.testing.assert_allclose(gravity, [0, 0, -1])
        # 90 degree roll rotates downward world gravity onto body -y.
        _, gravity = pelvis_imu([2**-.5, 2**-.5, 0, 0], [0]*3)
        np.testing.assert_allclose(gravity, [0, -1, 0], atol=1e-10)
        with self.assertRaises(ValueError):
            pelvis_imu([0]*4, [0]*3)

    def test_contract_rejects_wrong_motor_map(self):
        d = json.loads(self.path.read_text())
        d['sdk_indices'] = list(range(23))
        self.path.write_text(json.dumps(d))
        with self.assertRaises(ValueError):
            Contract(self.path)

    def test_action_scale_and_independent_arms(self):
        target = self.c.targets(np.ones(13), np.full(10, .4))
        np.testing.assert_allclose(target[:13], .6)
        np.testing.assert_allclose(target[13:], .4)
        with self.assertRaises(ValueError):
            self.c.targets(np.full(13, np.nan))

    def test_target_limits_and_rate(self):
        target = guarded_target(self.c, np.full(23, 99.), self.c.q_default, .02)
        np.testing.assert_allclose(target[:13], .26)
        np.testing.assert_allclose(target[13:], .14)
        for _ in range(100):
            target = guarded_target(self.c, np.full(23, 99.), target, .02)
        self.assertTrue(np.all(target <= 1.98+1e-10))
        with self.assertRaises(ValueError):
            guarded_target(self.c, target, target, .2)

    def test_watchdog_fault_injection(self):
        faults = [({'received': 9.}, 'lowstate_timeout'),
                  ({'mode_machine': 4}, 'mode_machine_changed'),
                  ({'remote': Remote(1 << SELECT)}, 'operator_stop'),
                  ({'remote': Remote(0)}, 'deadman_released'),
                  ({'gravity': np.array([1., 0., 0.])}, 'excessive_tilt'),
                  ({'q': np.full(23, 3.)}, 'measured_joint_limit'),
                  ({'dq': np.full(23, 25.)}, 'measured_joint_speed'),
                  ({'temperature': 75.}, 'motor_temperature'),
                  ({'q': np.full(23, np.nan)}, 'nonfinite_state')]
        for fields, expected in faults:
            with self.subTest(expected=expected):
                guard = Watchdog(self.c, 5)
                self.assertEqual(guard.check(self.state(**fields), 10., 10.01, True), expected)
                self.assertEqual(guard.check(self.state(), 10., 10.01, True), expected)
        guard = Watchdog(self.c, 5)
        self.assertEqual(guard.check(self.state(), 9., 10.01, True), 'control_timeout')

    def test_unused_sdk_slots_disabled_and_damping(self):
        msg = SimpleNamespace(motor_cmd=[SimpleNamespace(mode=1, kp=999) for _ in range(35)])
        fill_command(msg, self.c, self.c.q_default, self.state())
        self.assertEqual(msg.mode_pr, 0)
        self.assertEqual(msg.mode_machine, 5)
        for i, m in enumerate(msg.motor_cmd):
            self.assertEqual(m.mode, int(i in SDK_INDICES))
        fill_command(msg, self.c, self.c.q_default, self.state(), damping=True)
        self.assertTrue(all(m.kp == 0 and m.tau == 0 for m in msg.motor_cmd))
        self.assertTrue(all(msg.motor_cmd[i].kd > 0 for i in SDK_INDICES))

    def test_remote_byte_order(self):
        raw = bytearray(40)
        struct.pack_into('<H', raw, 2, (1 << L1) | (1 << SELECT))
        struct.pack_into('<ff', raw, 4, .25, -.5)
        struct.pack_into('<f', raw, 20, .75)
        remote = Remote.decode(raw)
        self.assertTrue(remote.pressed(L1))
        self.assertEqual((remote.lx, remote.ly, remote.rx), (.25, .75, -.5))

    def test_receiver_rejects_crc_duplicates_and_out_of_order(self):
        crc = SimpleNamespace(Crc=lambda msg: 42)
        receiver = Receiver('pelvis', crc)
        msg = SimpleNamespace(crc=42, mode_pr=0, mode_machine=5, tick=100,
                              imu_state=SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0]*3),
                              motor_state=[SimpleNamespace(q=.1, dq=0., temperature=[30, 30]) for _ in range(35)],
                              wireless_remote=bytes(40))
        receiver.callback(msg)
        first = receiver.snapshot()
        receiver.callback(msg)
        self.assertIs(receiver.snapshot(), first)
        msg.tick = 99
        receiver.callback(msg)
        self.assertIs(receiver.snapshot(), first)
        msg.tick, msg.crc = 101, 0
        receiver.callback(msg)
        self.assertIs(receiver.snapshot(), first)
        msg.crc = 42
        receiver.callback(msg)
        self.assertEqual(receiver.snapshot().tick, 101)

    def test_publisher_falls_back_without_policy_updates(self):
        sent = []
        channel = SimpleNamespace(Write=lambda m: sent.append([x.kp for x in m.motor_cmd]))
        receiver = SimpleNamespace(snapshot=lambda: self.state(received=time.monotonic()))
        msg = SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(35)])
        publisher = Publisher(self.c, receiver, channel, msg, SimpleNamespace(Crc=lambda m: 1), self.state())
        publisher.update(self.c.q_default, True)
        publisher.thread.start()
        try:
            time.sleep(.18)
        finally:
            publisher.stop.set()
            publisher.thread.join(timeout=1.)
        self.assertIsNotNone(publisher.guard.reason)
        self.assertGreater(len(sent), 1)
        self.assertTrue(all(v == 0 for v in sent[-1]))

    def test_mujoco_freejoint_angular_velocity_matches_body_gyro(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/><site name="imu"/></body></worldbody><sensor><gyro site="imu"/></sensor></mujoco>')
        data = mujoco.MjData(model)
        data.qpos[3:7] = [.5, .5, .5, .5]
        data.qvel[3:6] = [1., 2., 3.]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.sensordata[:3], data.qvel[3:6], atol=1e-10)

    def test_motion_service_takeover_requires_explicit_flag(self):
        from unittest.mock import Mock, patch
        switcher = Mock()
        switcher.CheckMode.return_value = (0, {'name': 'sport'})
        with self.assertRaises(RuntimeError):
            ensure_motion_available(switcher, False)
        switcher.ReleaseMode.assert_not_called()
        switcher.CheckMode.side_effect = [(0, {'name': 'sport'}), (0, {'name': ''})]
        switcher.ReleaseMode.return_value = (0, None)
        with patch('deploy.sim2real.time.sleep'):
            ensure_motion_available(switcher, True)
        switcher.ReleaseMode.assert_called_once()
        switcher.CheckMode.side_effect = None
        switcher.CheckMode.return_value = (1, None)
        with self.assertRaises(RuntimeError):
            ensure_motion_available(switcher, True)

    def test_normalization_export_uses_checkpoint_statistics(self):
        state = {}
        for index, inp, out in [(0, 71, 512), (2, 512, 256), (4, 256, 128), (6, 128, 13)]:
            state[f'actor.{index}.weight'] = torch.zeros(out, inp)
            state[f'actor.{index}.bias'] = torch.zeros(out)
            state[f'actor.{index}.weight'][0, 0] = 1.
        state['actor_obs_normalizer._mean'] = torch.full((1, 71), 3.)
        state['actor_obs_normalizer._std'] = torch.full((1, 71), 2.)
        actor = Actor(state).eval()
        with torch.inference_mode():
            result = torch.jit.script(actor)(torch.full((1, 71), 7.02))
        self.assertAlmostEqual(result[0, 0].item(), 2., places=6)
        self.assertEqual(torch.count_nonzero(result).item(), 1)


if __name__ == '__main__':
    unittest.main()
