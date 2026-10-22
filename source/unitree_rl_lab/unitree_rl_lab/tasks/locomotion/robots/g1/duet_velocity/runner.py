"""RSL 3.1 bridge: DUET symmetry, entropy schedule and resumable curricula."""
import copy,json,hashlib
from pathlib import Path
import torch
from rsl_rl.runners import OnPolicyRunner
from .symmetry import check_symmetry
from ..common.asset_cfg import JOINTS, Q0, ACTION_SCALE, KP, KD, URDF


def entropy_at(iteration):
    return .01 + (.004-.01)*min(1., max(0., (iteration-1500)/1000.))

class DuetRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        super().__init__(env, copy.deepcopy(train_cfg), log_dir, device)
        self.duet_updates = 0
        obs = env.get_observations()
        assert obs['policy'].shape[-1] == 71 and obs['critic'].shape[-1] == 96
        assert env.num_actions == 13
        with torch.inference_mode():
            errs = check_symmetry(env, obs, torch.randn(env.num_envs,13,device=device))
        assert max(errs.values()) < 1e-6, errs
        self.symmetry_check = errs
        update = self.alg.update
        def scheduled_update(*args, **kwargs):
            self.alg.entropy_coef = entropy_at(self.duet_updates)
            result = update(*args, **kwargs)
            result['entropy_coef'] = self.alg.entropy_coef
            self.duet_updates += 1
            return result
        self.alg.update = scheduled_update

    def save(self, path, infos=None):
        info = dict(infos or {})
        info['duet_port'] = {'version':1,'common_step_counter':int(self.env.unwrapped.common_step_counter),
            'completed_updates':self.duet_updates, 'obs_dims':[71,96], 'actions':13}
        super().save(path, info)
        contract={'task':'Unitree-G1-23dof-Duet-Velocity','physics':'IsaacLab/PhysX',
            'joint_names':JOINTS,'action_joint_names':JOINTS[:13],'external_arm_joint_names':JOINTS[13:],
            'actor_dim':71,'critic_dim':96,'action_dim':13,'control_dt':.02,
            'q_default':Q0,'kp':KP,'kd':KD,'action_scale':{n:ACTION_SCALE[n] for n in JOINTS[:13]},
            'urdf':str(URDF),'urdf_sha256':hashlib.sha256(URDF.read_bytes()).hexdigest(),
            'trained_stage_step':info['duet_port']['common_step_counter'],
            'height_range_at_full_curriculum':[.24,.78],
            'deployment_validated':False}
        Path(path).with_suffix('.contract.json').write_text(json.dumps(contract,indent=2))

    def load(self, path, load_optimizer=True, map_location=None):
        # Reject upstream/native weights before touching the model/optimizer.
        payload=torch.load(path,map_location='cpu',weights_only=False)
        info=(payload.get('infos') or {}).get('duet_port')
        if not info or info.get('obs_dims') != [71,96] or info.get('actions') != 13:
            raise ValueError('Resume requires a checkpoint produced by this IsaacLab DUET port.')
        result=super().load(path,load_optimizer,map_location)
        self.duet_updates=info['completed_updates']
        self.current_learning_iteration=self.duet_updates
        if load_optimizer:
            self.env.unwrapped.common_step_counter=info['common_step_counter']
            self.env.reset()  # reapply ranges before sampling fresh episodes
        return result
