"""Integration and source-equation parity checks; no long training."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('outputs/validation')); AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);app=AppLauncher(args).app
import ast,json,copy,torch,types
import gymnasium as gym
from isaaclab.utils import math as math_utils,string as string_utils
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import unitree_rl_lab.tasks
from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity import adapters as a,rewards as rw
from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.velocity_env_cfg import RobotEnvCfg,SOURCE,LOWER,ARMS
from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.rsl_rl_ppo_cfg import DuetPPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.runner import DuetRunner,entropy_at
from unitree_rl_lab.tasks.locomotion.robots.g1.common import mdp as port
report={};env=None
try:
 cfg=RobotEnvCfg();cfg.scene.num_envs=64;cfg.seed=1;cfg.observations.policy.enable_corruption=False
 env=gym.make('Unitree-G1-23dof-Duet-Velocity',cfg=cfg).unwrapped
 obs,_=env.reset();assert obs['policy'].shape==(64,71) and obs['critic'].shape==(64,96)
 ids=a.ordered_ids(env);leg=env.action_manager.get_term('joint_pos');arm=env.action_manager.get_term('upper_body_pose')
 assert [env.scene['robot'].joint_names[i] for i in leg._joint_ids]==LOWER
 assert arm.joint_names==ARMS and len(set(leg._joint_ids)&set(arm.joint_ids.tolist()))==0
 report['mapping']={'actor':71,'critic':96,'rl_actions':13,'external_arm':10,'joint_names': [env.scene['robot'].joint_names[i] for i in ids]}
 stage_results={}
 for it in [0,1500,1750,2000,3000,3500,5000,6000,7000]:
  env.common_step_counter=it*24;obs,_=env.reset();height=env.command_manager.get_term('base_height');twist=env.command_manager.get_term('twist')
  assert height.command.min()>=height.current_floor-1e-5 and height.command.max()<=.78001, (it,float(height.command.min()),float(height.command.max()),height.current_floor)
  if it==1750:assert abs(twist.cfg.ranges.ang_vel_z[1]-.6)<1e-6
  if it==7000:assert abs(height.current_floor-.24)<1e-6 and abs(arm.curriculum_ratio-1.)<1e-6
  stage_results[it]={'height_floor':height.current_floor,'walk_floor':height.cfg.walk_min_height,'arm_ratio':arm.curriculum_ratio,'arm_clean_fraction':arm.clean_fraction,'yaw_max':twist.cfg.ranges.ang_vel_z[1],'entropy':entropy_at(it)}
 report['curriculum']=stage_results
 # Fixed per-env commands exercise movement, pure turn, idle, and crouch gates.
 cmd=env.command_manager.get_term('twist');hc=env.command_manager.get_term('base_height')
 def fixed(self):
  self.vel_command_b[:]=torch.tensor([[.5,0,0],[0,0,.3],[0,.2,0],[0,0,0]],device=env.device).repeat(16,1)
  self.is_standing_env[:]=self.vel_command_b.norm(dim=1)<1e-8
 cmd._update_command=types.MethodType(fixed,cmd)
 # Late-curriculum arms and repeated episode resets, all terms finite.
 max_arm_delta=0.;max_arm_accel=0.
 for i in range(180):
  obs,reward,terminated,truncated,info=env.step(torch.zeros(64,13,device=env.device))
  assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(reward).all()
  assert torch.isfinite(env.reward_manager._step_reward).all()
  max_arm_delta=max(max_arm_delta,float((arm.traj_pos-arm.default_pose).abs().max()))
  max_arm_accel=max(max_arm_accel,float(arm.traj_acc.abs().max()))
 assert max_arm_delta>.05
 report['simulation']={'steps':180,'all_rewards_finite':True,'reward_terms':len(env.reward_manager.active_terms),'max_arm_target_delta_rad':max_arm_delta,'max_arm_target_accel_rad_s2':max_arm_accel}
 # Compare original reward expressions on the SAME adapted physical state.
 src=(args.upstream/'src/tasks/common/mdp/rewards.py').read_text();tree=ast.parse(src)
 ns={'torch':torch,'_DEFAULT_ASSET_CFG':types.SimpleNamespace(name='robot',body_ids=[],site_ids=slice(None),joint_ids=slice(None)),
     'matrix_from_quat':math_utils.matrix_from_quat,'quat_apply_inverse':math_utils.quat_apply_inverse,'wrap_to_pi':math_utils.wrap_to_pi,
     'resolve_matching_names_values':string_utils.resolve_matching_names_values,'ManagerTermBase':object}
 code='from __future__ import annotations\n'+'\n\n'.join(ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef))
 exec(compile(code,'upstream_reward_expressions','exec'),ns)
 d=env.scene['robot'].data
 class DataProxy:
  def __getattr__(self,n):return getattr(d,n)
 dp=DataProxy();dp.site_pos_w=a.site_pos(env);dp.site_lin_vel_w=a.site_vel(env);dp.gravity_vec_w=d.GRAVITY_VEC_W
 dp.data=types.SimpleNamespace(subtree_com=a.whole_com(env)[:,None,:]);dp.indexing=types.SimpleNamespace(root_body_id=0)
 robot=types.SimpleNamespace(data=dp)
 class EnvProxy:
  def __getattr__(self,n):return getattr(env,n)
 ep=EnvProxy();ep.scene={'robot':robot,'feet_ground_contact':a.feet_sensor(env)}
 ep.action_manager=types.SimpleNamespace(action=env.action_manager.action,prev_action=env.action_manager.prev_action,prev_prev_action=env.duet_prev_prev_action)
 errors={}
 for name in env.reward_manager.active_terms:
  term=env.reward_manager.get_term_cfg(name);fn=SOURCE['rewards'][name]['func'].split(':')[-1]
  if fn not in ns or fn == 'self_collision_cost':continue
  params=copy.deepcopy(term.params)
  if 'asset_cfg' in params:
   c=params['asset_cfg'];params['asset_cfg']=types.SimpleNamespace(name=c.name,body_ids=c.body_ids,joint_ids=c.joint_ids,site_ids=slice(None))
  original=ns[fn](ep,**params);ported=term.func(env,**term.params)
  error=float((original-ported).abs().max());assert error<1e-5,(name,error);errors[name]=error
 report['source_formula_max_errors']=errors
 # PPO checkpoint loads; schedules follow saved environment counter after reset.
 wrapper=RslRlVecEnvWrapper(env);agent=DuetPPORunnerCfg();agent.logger='tensorboard'
 runner=DuetRunner(wrapper,agent.to_dict(),log_dir=None,device=env.device)
 runner.load(str(args.checkpoint));ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
 state=ckpt['infos']['duet_port'];assert env.common_step_counter==state['common_step_counter'] and runner.duet_updates==state['completed_updates']
 from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.curriculums import piecewise_linear
 expected_arm=min(1., max(0.,arm.cfg.init_ratio+piecewise_linear(state['common_step_counter'],arm.cfg.ratio_stages)))
 assert abs(arm.curriculum_ratio-expected_arm)<1e-6
 with torch.inference_mode():
  act=runner.get_inference_policy()(wrapper.get_observations());assert act.shape==(64,13) and torch.isfinite(act).all()
 report['resume']={'checkpoint':str(args.checkpoint),'state':state,'restored_stage':True,'inference_finite':True,'symmetry':runner.symmetry_check}
 report['passed']=True
 out=(args.output/'VALIDATION_PASS.json');out.write_text(json.dumps(report,indent=2));print('DUET_VALIDATION_PASS',out,flush=True)
except BaseException:
 import traceback
 error=traceback.format_exc();print(error,flush=True)
 (args.output/'VALIDATION_ERROR.txt').write_text(error)
 raise
finally:
 if env is not None:env.close()
 app.close()
