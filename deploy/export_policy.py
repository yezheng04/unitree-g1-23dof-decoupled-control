"""Export the evaluated RSL-RL 3 checkpoint to CPU TorchScript + a portable contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch import nn

from .common import JOINT_NAMES, ROOT, SDK_INDICES, sha256


class Actor(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(71, 512), nn.ELU(), nn.Linear(512, 256),
                                   nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 13))
        self.actor.load_state_dict({k.removeprefix('actor.'): v for k, v in state.items() if k.startswith('actor.')}, strict=True)
        self.register_buffer('mean', state['actor_obs_normalizer._mean'].clone())
        self.register_buffer('std', state['actor_obs_normalizer._std'].clone())
        if self.mean.shape != (1, 71) or self.std.shape != (1, 71) or (self.std < 0).any():
            raise ValueError('Invalid observation normalization')
        if not all(torch.isfinite(x).all() for x in self.state_dict().values()):
            raise ValueError('Non-finite policy weights')

    def forward(self, observation):
        # RSL-RL 3.1.2 EmpiricalNormalization uses std + 1e-2, not sqrt(var + eps).
        return self.actor((observation - self.mean) / (self.std + 0.01))


def export(checkpoint, contract_path, urdf, mjcf, output):
    torch.set_num_threads(1)
    checkpoint, contract_path, urdf, mjcf, output = map(Path, (checkpoint, contract_path, urdf, mjcf, output))
    source = json.loads(contract_path.read_text())
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    info = payload.get('infos', {}).get('duet_port', {})
    if info.get('obs_dims') != [71, 96] or info.get('actions') != 13:
        raise ValueError('Expected a checkpoint from this IsaacLab task')
    if source['joint_names'] != JOINT_NAMES or source['action_joint_names'] != JOINT_NAMES[:13]:
        raise ValueError('Checkpoint contract joint order mismatch')
    if (source['control_dt'] != .02 or source['actor_dim'] != 71 or source['action_dim'] != 13
            or source['trained_stage_step'] != info['common_step_counter']):
        raise ValueError('Checkpoint and sidecar metadata disagree')
    if sha256(urdf) != source['urdf_sha256']:
        raise ValueError('URDF does not match the training contract')
    limits = {j.attrib['name']: j.find('limit').attrib for j in ET.parse(urdf).getroot().findall('joint') if j.attrib['type'] != 'fixed'}
    actor = Actor(payload['model_state_dict']).eval()
    scripted = torch.jit.script(actor)
    generator = torch.Generator().manual_seed(42)
    samples = torch.randn(128, 71, generator=generator) * (actor.std + .01) + actor.mean
    with torch.inference_mode():
        error = (actor(samples) - scripted(samples)).abs().max().item()
    if error > 1e-5:
        raise ValueError(f'Export parity failed: {error}')
    output.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output/'policy.pt'))
    meta = {k: source[k] for k in ('joint_names', 'actor_dim', 'critic_dim', 'action_dim', 'control_dt', 'urdf_sha256')}
    for key in ('q_default', 'kp', 'kd'):
        meta[key] = [source[key][n] for n in JOINT_NAMES]
    meta.update(schema_version=1, sdk_indices=SDK_INDICES, normalization_eps=.01,
                policy_file='policy.pt', policy_sha256=sha256(output/'policy.pt'),
                checkpoint_sha256=sha256(checkpoint), completed_updates=info['completed_updates'],
                mjcf_sha256=sha256(mjcf), action_scale=[source['action_scale'][n] for n in JOINT_NAMES[:13]],
                armature=[.025101925 if ('hip_roll' in n or 'knee' in n) else .010177520
                          if ('hip_pitch' in n or 'hip_yaw' in n or 'waist' in n)
                          else .003609725*(2 if 'ankle' in n else 1) for n in JOINT_NAMES],
                export_max_abs_error=error, hardware_validated=False)
    for key in ('lower', 'upper', 'effort', 'velocity'):
        meta[key] = [float(limits[n][key]) for n in JOINT_NAMES]
    (output/'deploy.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'checkpoint_sha256': meta['checkpoint_sha256'], 'export_max_abs_error': error}))
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/checkpoints/model_22000.pt')
    p.add_argument('--contract', type=Path)
    p.add_argument('--urdf', type=Path, default=ROOT/'assets/g1/g1_23dof_rev_1_0.urdf')
    p.add_argument('--mjcf', type=Path, default=ROOT/'assets/g1/g1_23dof_rev_1_0.xml')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/deploy/22000')
    a = p.parse_args()
    export(a.checkpoint, a.contract or a.checkpoint.with_suffix('.contract.json'), a.urdf, a.mjcf, a.output)


if __name__ == '__main__':
    main()
