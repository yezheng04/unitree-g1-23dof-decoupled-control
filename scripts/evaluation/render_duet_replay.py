"""Render recorded PhysX poses with MuJoCo graphics only (never mj_step)."""
import os
os.environ['MUJOCO_GL']='egl'
import argparse,json,xml.etree.ElementTree as ET,hashlib,re
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
manifest=json.loads((a.input/'evaluation.json').read_text())
checkpoint_tag=Path(manifest['checkpoint']).stem
if not re.fullmatch(r'[A-Za-z0-9_-]+', checkpoint_tag): raise ValueError('Invalid checkpoint label')
from unitree_rl_lab.paths import MJCF
asset=MJCF;root=ET.parse(asset).getroot();root.find('compiler').set('meshdir',str(asset.parent/'meshes'))
visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='960',offheight='640');ET.SubElement(visual,'headlight',ambient='.45 .45 .45',diffuse='.7 .7 .7',specular='.15 .15 .15')
wb=root.find('worldbody');ET.SubElement(wb,'light',pos='0 -3 5',dir='0 0 -1',diffuse='.7 .7 .7')
ET.SubElement(wb,'geom',name='review_floor',type='plane',size='40 40 .1',rgba='.82 .84 .86 1',group='2',contype='0',conaffinity='0')
for i in range(-20,41):
 for axis in (0,1):
  ET.SubElement(wb,'geom',name=f'grid_{axis}_{i+20}',type='box',size='20 .003 .001' if axis==0 else '.003 20 .001',pos=f'5 {i*.5} .001' if axis==0 else f'{i*.5} 5 .001',rgba='.47 .50 .53 1',group='2',contype='0',conaffinity='0')
xml=a.input/'render_scene.xml';ET.ElementTree(root).write(xml)
m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);renderer=mujoco.Renderer(m,640,960)
qids=[m.joint(n).qposadr[0] for n in manifest['joint_names']]
bodyids=[m.body(n).id for n in manifest['body_names']]
assert len(qids)==23 and m.nq==30
opt=mujoco.MjvOption();opt.geomgroup[:]=0;opt.geomgroup[1:3]=1
cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.distance=3.2;cam.azimuth=135;cam.elevation=-16
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
maxpos=maxrot=0.;outjobs=[]
arr=np.load(a.input/'replay.npz')
groups={'locomotion':['forward05','left02','right02','yaw_left03','yaw_right03','curve_left','curve_right'],
        'height':['squat60','squat38','squat24','walk_low60'],
        'arms':['forward_arms','yaw_left_arms','squat24_arms']}
lookup={c['name']:i for i,c in enumerate(manifest['cases'])}
for group,names in groups.items():
 output=a.input/f'duet_{checkpoint_tag}_{group}.mp4'
 with imageio.get_writer(str(output),fps=25,codec='libx264',quality=7,macro_block_size=16,ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart']) as writer:
  for name in names:
   ci=lookup[name];case=manifest['cases'][ci];positions=arr['poses'][ci];origin=positions[0,:2].copy();smooth=np.zeros(3);maxcase=0.
   for tick in range(1,len(positions),2):
    v=positions[tick];d.qpos[:7]=v[:7];d.qpos[qids]=v[7:];mujoco.mj_forward(m,d)
    ref=arr['body_poses'][ci,tick];err=float(np.linalg.norm(d.xpos[bodyids]-ref[:,:3],axis=1).max());dot=np.abs((d.xquat[bodyids]*ref[:,3:]).sum(1));rot=float((1-dot).max())
    maxpos=max(maxpos,err);maxrot=max(maxrot,rot);maxcase=max(maxcase,err)
    assert err<.0002 and rot<.00001,(name,tick,err,rot)
    cam.lookat[:]=[*v[:2],.75]
    renderer.update_scene(d,camera=cam,scene_option=opt);frame=renderer.render();im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
    met=arr['metrics'][ci,tick];smooth=.9*smooth+.1*met[:3];t=(tick+1)*.02;cmd=case['command'] if t>=3 else [0,0,0];hcmd=case['height'] if t>=3 else .78
    draw.rectangle((0,0,960,134),fill=(15,20,28))
    msgs=[(f'DUET IsaacLab | {checkpoint_tag} | {name} | t={t:.2f}s | 1x','white'),
          (f'Cmd vx/vy/yaw: {cmd[0]:+.2f} / {cmd[1]:+.2f} / {cmd[2]:+.2f}  | height: {hcmd:.2f} m',(130,220,255)),
          (f'Actual: {smooth[0]:+.3f} / {smooth[1]:+.3f} / {smooth[2]:+.3f} | height: {met[3]:.3f} m',(240,220,130)),
          (f'Arms: {"FULL RANDOM TRAJECTORY" if case["arms"] else "DEFAULT PD TARGET"} | '+('FIRST EPISODE ALIVE' if met[4]>.5 else 'FELL: subsequent auto-reset excluded from statistics'),(130,230,170) if met[4]>.5 else (255,100,100)),
          ('Recorded PhysX trajectory | graphics replay only | grid 0.5 m',(190,190,190))]
    for row,(msg,color) in enumerate(msgs):draw.text((12,5+row*25),msg,font=font,fill=color)
    writer.append_data(np.asarray(im))
    if tick in (249,599,799):im.save(a.input/f'{checkpoint_tag}_{name}_{tick}.jpg')
    if a.smoke and tick>=149:break
   print('RENDERED',group,name,'max_link_error_m',maxcase,flush=True)
   if a.smoke:break
 outjobs.append({'video':str(output),'cases':names,'seconds_per_case':manifest['seconds']})
 if a.smoke:break
renderer.close()
(a.input/'RENDER_PASS.json').write_text(json.dumps({'offline_pose_rendering':True,'physics':'IsaacLab PhysX','mujoco_simulation_steps':0,'max_link_position_error_m':maxpos,'max_quaternion_1_absdot':maxrot,'visual_asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),'fps':25,'smoke':a.smoke,'videos':outjobs},indent=2))
print('RENDER_COMPLETE',flush=True)
