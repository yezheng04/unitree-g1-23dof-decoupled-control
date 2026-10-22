"""Train the isolated DUET port. Uses its runner, never the generic Unitree exporter."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--num_envs',type=int,default=4096)
p.add_argument('--max_iterations',type=int,default=25001)
p.add_argument('--seed',type=int,default=1)
p.add_argument('--run_name',default='duet')
p.add_argument('--checkpoint',type=Path)
p.add_argument('--logger',choices=['wandb','tensorboard'],default='wandb')
p.add_argument('--offline',action='store_true')
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(args).app
import os,json,torch,gymnasium as gym
from datetime import datetime
import unitree_rl_lab.tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.velocity_env_cfg import RobotEnvCfg
from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.rsl_rl_ppo_cfg import DuetPPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.runner import DuetRunner
try:
    if args.offline:os.environ['WANDB_MODE']='offline'
    cfg=RobotEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
    if args.device:cfg.sim.device=args.device
    agent=DuetPPORunnerCfg();agent.seed=args.seed;agent.logger=args.logger
    agent.run_name=args.run_name;agent.max_iterations=args.max_iterations
    log=Path('logs/rsl_rl')/agent.experiment_name/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+args.run_name)
    log.mkdir(parents=True)
    import shutil,importlib.metadata,hashlib
    task_source=Path(__import__('unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity',fromlist=['']).__file__).parent
    shutil.copytree(task_source,log/'task_source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__,log/'train_duet.py')
    (log/'provenance.json').write_text(json.dumps({'upstream_commit':'439c21bd2817feedd519475e8301195b95cb6fcb','rsl_rl':importlib.metadata.version('rsl-rl-lib'),'torch':torch.__version__,'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in task_source.glob('*.py')}},indent=2))
    dump_yaml(str(log/'env.yaml'),cfg);dump_yaml(str(log/'agent.yaml'),agent)
    env=RslRlVecEnvWrapper(gym.make('Unitree-G1-23dof-Duet-Velocity',cfg=cfg),clip_actions=None)
    runner=DuetRunner(env,agent.to_dict(),log_dir=str(log),device=cfg.sim.device)
    if args.checkpoint:runner.load(str(args.checkpoint))
    remaining=args.max_iterations-runner.duet_updates
    if remaining<=0:raise ValueError('Budget already reached: --max_iterations is total updates, including resume.')
    runner.learn(remaining,init_at_random_ep_len=True)
    assert all(torch.isfinite(t).all() for t in runner.alg.policy.state_dict().values())
    (log/'TRAIN_PASS.json').write_text(json.dumps({'completed_updates':runner.duet_updates,'global_step':env.unwrapped.common_step_counter,'symmetry':runner.symmetry_check,'dimensions':[71,96,13],'finite_weights':True},indent=2))
    print('DUET_TRAIN_PASS',log,flush=True);env.close()
except BaseException:
    import traceback
    error=traceback.format_exc();print(error,flush=True)
    if 'log' in globals():(log/'ERROR.txt').write_text(error)
    raise
finally:app.close()
