"""Fixed-command DUET policy evaluation and actual PhysX trajectory capture."""
import argparse, json, traceback, types, hashlib
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--per_case',type=int,default=16)
p.add_argument('--seconds',type=float,default=18.)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
app=AppLauncher(args).app;env=None
try:
 import numpy as np, torch, gymnasium as gym
 import unitree_rl_lab.tasks
 from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
 from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.velocity_env_cfg import RobotEnvCfg
 from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.rsl_rl_ppo_cfg import DuetPPORunnerCfg
 from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity.runner import DuetRunner
 from unitree_rl_lab.tasks.locomotion.robots.g1.duet_velocity import adapters
 from unitree_rl_lab.tasks.locomotion.robots.g1.common import mdp
 cases=[]
 def add(name,cmd,height=.78,arms=False):cases.append(dict(name=name,command=cmd,height=height,arms=arms))
 add('stand',[0,0,0]);add('forward05',[.5,0,0]);add('backward03',[-.3,0,0])
 add('left02',[0,.2,0]);add('right02',[0,-.2,0]);add('yaw_left03',[0,0,.3]);add('yaw_right03',[0,0,-.3])
 add('curve_left',[.5,0,.3]);add('curve_right',[.5,0,-.3])
 for h in [.60,.38,.24]:add('squat'+str(round(h*100)),[0,0,0],h)
 add('walk_low60',[.5,0,0],.6)
 for name,cmd,h in [('stand_arms',[0,0,0],.78),('forward_arms',[.5,0,0],.78),('yaw_left_arms',[0,0,.3],.78),('yaw_right_arms',[0,0,-.3],.78),('squat38_arms',[0,0,0],.38),('squat24_arms',[0,0,0],.24)]:add(name,cmd,h,True)
 cfg=RobotEnvCfg();cfg.scene.num_envs=len(cases)*args.per_case;cfg.seed=42;cfg.curriculum={}
 cfg.episode_length_s=args.seconds+5
 cfg.observations.policy.enable_corruption=False
 cfg.events.push_robot=None;cfg.events.encoder_bias=None;cfg.events.base_com=None;cfg.events.hand_com=None
 cfg.events.reset_base.params['pose_range'].update(x=(0.,0.),y=(0.,0.),yaw=(0.,0.))
 cfg.actions.upper_body_pose.ratio_stages=((0,1.),);cfg.actions.upper_body_pose.init_ratio=0.
 cfg.actions.upper_body_pose.clean_fraction_stages=((0,0.),)
 cfg.commands.base_height.floor_curriculum_start=.24
 cfg.commands.base_height.walk_min_height=.6;cfg.commands.base_height.nominal_env_fraction=0.
 env=gym.make('Unitree-G1-23dof-Duet-Velocity',cfg=cfg).unwrapped
 w=RslRlVecEnvWrapper(env);robot=env.scene['robot'];device=env.device
 runner=DuetRunner(w,DuetPPORunnerCfg().to_dict(),log_dir=None,device=device)
 runner.load(str(args.checkpoint),load_optimizer=False)
 policy=runner.get_inference_policy(device=device)
 target=torch.tensor([c['command'] for c in cases],device=device).repeat_interleave(args.per_case,0)
 heights=torch.tensor([c['height'] for c in cases],device=device).repeat_interleave(args.per_case,0)
 dynamic=torch.tensor([c['arms'] for c in cases],device=device).repeat_interleave(args.per_case)
 clock={'t':0.}
 def command(self):
  self.vel_command_b[:]=target if clock['t']>=3. else 0.
  self.is_standing_env[:]=self.vel_command_b.norm(dim=1)<1e-8
 def height(self):self._height[:,0]=heights if clock['t']>=3. else .78
 twist=env.command_manager.get_term('twist');hc=env.command_manager.get_term('base_height')
 twist._update_command=types.MethodType(command,twist);hc._update_command=types.MethodType(height,hc)
 arm=env.action_manager.get_term('upper_body_pose');original=arm.process_actions
 def arm_process(self,actions):
  original(actions)
  self._traj_pos[~dynamic]=self._default[~dynamic]
  self._traj_vel[~dynamic]=0.;self._traj_acc[~dynamic]=0.
 arm.process_actions=types.MethodType(arm_process,arm)
 samples={k:[] for k in ['velocity','valid','height','tilt','contact','foot_speed','foot_height','xy','arm_error']}
 poses=[];bodies=[];frame_metrics=[]
 selected=torch.arange(len(cases),device=device)*args.per_case
 with torch.inference_mode():
  w.reset();env.common_step_counter=0;env.episode_length_buf.zero_();command(twist);height(hc)
  obs=w.get_observations();alive=torch.ones(env.num_envs,device=device,dtype=torch.bool)
  for step in range(round(args.seconds/env.step_dt)):
   clock['t']=step*env.step_dt;command(twist);height(hc)
   # Commands must be included in the observation used for this action.
   obs=w.get_observations()
   obs,_,done,_=w.step(policy(obs));alive &= ~done.bool()
   d=robot.data
   v=torch.cat((d.root_link_lin_vel_b[:,:2],d.root_link_ang_vel_b[:,2:3]),1)
   measured=d.root_link_pos_w[:,2]-adapters.site_pos(env)[:,:,2].min(1).values+.02
   values={'velocity':v,'valid':alive,'height':measured,'tilt':torch.acos((-d.projected_gravity_b[:,2]).clamp(-1,1)),
     'contact':mdp.foot_contact(env),'foot_speed':mdp.feet_state(env)[1][:,:,:2].norm(dim=-1),'foot_height':mdp.foot_height(env),
     'xy':d.root_link_pos_w[:,:2]-env.scene.env_origins[:,:2],
     'arm_error':(d.joint_pos[:,arm.joint_ids]-arm.traj_pos).abs().mean(1)}
   for key,value in values.items():samples[key].append(value.cpu().numpy().copy())
   rp=d.root_link_pos_w[selected]-env.scene.env_origins[selected]
   poses.append(torch.cat((rp,d.root_link_quat_w[selected],d.joint_pos[selected]),1).cpu().numpy())
   bp=d.body_link_pos_w[selected]-env.scene.env_origins[selected,None,:]
   bodies.append(torch.cat((bp,d.body_link_quat_w[selected]),2).cpu().numpy())
   frame_metrics.append(torch.cat((v[selected],measured[selected,None],alive[selected,None].float()),1).cpu().numpy())
   if step%200==0:print('EVAL_STEP',step,'alive',int(alive.sum()),flush=True)
 data={k:np.stack(v) for k,v in samples.items()};time=(np.arange(len(poses))+1)*env.step_dt
 report={}
 for i,c in enumerate(cases):
  sl=slice(i*args.per_case,(i+1)*args.per_case);mask=data['valid'][:,sl]&(time[:,None]>=6.)
  row=dict(c,survived=int(data['valid'][-1,sl].sum()),total=args.per_case,valid_samples=int(mask.sum()),video_env_index=i*args.per_case,
    video_survived=bool(data['valid'][-1,i*args.per_case]))
  if mask.any():
   v=data['velocity'][:,sl][mask];h=data['height'][:,sl][mask];contact=data['contact'][:,sl][mask].astype(bool)
   row.update(mean_velocity=v.mean(0).tolist(),mae=np.abs(v-np.array(c['command'])).mean(0).tolist(),
    height_mean=float(h.mean()),height_mae=float(np.abs(h-c['height']).mean()),
    tilt_mean_deg=float(np.rad2deg(data['tilt'][:,sl][mask]).mean()),
    xy_speed_rms=float(np.sqrt((v[:,:2]**2).sum(1).mean())),contact_duty=contact.mean(0).tolist(),
    stance_foot_speed=float(data['foot_speed'][:,sl][mask][contact].mean()) if contact.any() else None,
    foot_height_p95=np.percentile(data['foot_height'][:,sl][mask],95,axis=0).tolist(),
    arm_position_mae=float(data['arm_error'][:,sl][mask].mean()))
  survivors=data['valid'][-1,sl]
  if survivors.any():row['survivor_xy_displacement_after6s']=float(np.linalg.norm(data['xy'][-1,sl][survivors]-data['xy'][299,sl][survivors],axis=1).mean())
  report[c['name']]=row
 manifest={'task':'Unitree-G1-23dof-Duet-Velocity','checkpoint':str(args.checkpoint),'sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
   'seconds':args.seconds,'per_case':args.per_case,'seed':42,'dt':env.step_dt,
   'protocol':'Deterministic mean actions; 0-3s standing at .78m, step to test command at 3s; statistics from 6s to end, first-episode-only until first fall. Arms active from start in arms cases. Full arm ratio=1, no clean slice. Height targets are not smoothed.',
   'randomization':'Retain training foot friction and reset hand/torso mass randomization. Disable observation noise, pushes, COM randomization and encoder bias. Initial xy/yaw zero. This is not full training-distribution robustness.',
   'physics':'IsaacLab PhysX','rendering':'Recorded PhysX body/joint poses, offline MuJoCo graphics ONLY, no mj_step.',
   'joint_names':robot.joint_names,'body_names':robot.body_names,'results':report,'cases':cases,
   'checkpoint_updates':runner.duet_updates}
 (args.output/'evaluation.json').write_text(json.dumps(manifest,indent=2))
 np.savez_compressed(args.output/'traces.npz',time=time,**data)
 np.savez_compressed(args.output/'replay.npz',poses=np.stack(poses).transpose(1,0,2),body_poses=np.stack(bodies).transpose(1,0,2,3),metrics=np.stack(frame_metrics).transpose(1,0,2))
 (args.output/'PASS.json').write_text(json.dumps({'evaluation_completed':True,'cases':len(cases),'episodes':len(cases)*args.per_case,'learning':False}))
 print('DUET_EVALUATION_PASS',json.dumps(report),flush=True)
except BaseException:
 error=traceback.format_exc();(args.output/'ERROR.txt').write_text(error);print(error,flush=True);raise
finally:
 if env is not None:env.close()
 app.close()
