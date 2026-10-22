"""DUET environment: action second differences and auxiliary diagnostics."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from .adapters import diagnostics

class DuetEnv(ManagerBasedRLEnv):
    def step(self, action):
        self.duet_prev_prev_action = self.action_manager.prev_action.clone()
        result = super().step(action)
        result[-1].setdefault('log', {}).update({f'Metrics/{k}': v for k, v in diagnostics(self).items()})
        return result

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, 'duet_prev_prev_action'):
            self.duet_prev_prev_action = torch.zeros_like(self.action_manager.action)
        self.duet_prev_prev_action[env_ids] = 0.
