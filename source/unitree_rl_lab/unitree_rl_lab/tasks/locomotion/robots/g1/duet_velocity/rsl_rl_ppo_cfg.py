from isaaclab.utils import configclass
from ..common.ppo_cfg import BasePPORunnerCfg
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class DuetPPORunnerCfg(BasePPORunnerCfg):
    # README's example uses seed 1; source factory default is 42.
    class_name = 'DuetRunner'
    logger = 'wandb'
    empirical_normalization = False
    obs_groups = {'policy': ['policy'], 'critic': ['critic']}
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(init_noise_std=1.,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu',actor_obs_normalization=True,critic_obs_normalization=True,noise_std_type='scalar')
    seed = 1
    max_iterations = 25001
    save_interval = 500
    experiment_name = 'g1_23_duet_isaaclab'
    wandb_project = 'g1-23-duet-isaaclab'
    def __post_init__(self):
        self.algorithm.entropy_coef = .01
        self.algorithm.symmetry_cfg = {
            'use_data_augmentation': True, 'use_mirror_loss': True,
            'mirror_loss_coeff': 1.,
            'data_augmentation_func': 'unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.symmetry:mirror_g1_obs_actions'}
