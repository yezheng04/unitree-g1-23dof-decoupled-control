"""Own Rev1.0 body; controller recipe from Unitree mjlab 1425b15 (Apache-2.0)."""
from pathlib import Path
import xml.etree.ElementTree as ET
import isaaclab.sim as sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg, UnitreeUrdfFileCfg
from unitree_rl_lab.paths import PROJECT_ROOT as ROOT, URDF, CACHE_DIR
if not URDF.is_file():
    raise FileNotFoundError(f"Robot asset missing: {URDF}. Run python scripts/setup/fetch_robot_assets.py or set G1_23DOF_URDF.")
_tree = ET.parse(URDF).getroot()
JOINTS = [j.attrib['name'] for j in _tree.findall('joint') if j.attrib['type']!='fixed']
assert len(JOINTS)==23
# Source MJCF depth-first joint/observation order, explicitly resolved in Lab.
JOINTS = [f'{s}_{n}_joint' for s in ('left','right') for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')] + ['waist_yaw_joint'] + [f'{s}_{n}_joint' for s in ('left','right') for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll')]
BODY_NAMES = ['pelvis'] + [j.find('child').attrib['link'] for j in _tree.findall('joint') if j.attrib['type']!='fixed']
FOOT_NAMES = ['left_ankle_roll_link','right_ankle_roll_link']
FOOT_OFFSET = (0.04,0.,-0.037)
IMU_OFFSET = (0.04525,0.,-0.08339)
_limits={j.attrib['name']:j.find('limit').attrib for j in _tree.findall('joint') if j.attrib['type']!='fixed'}
EFFORT={n:float(_limits[n]['effort']) for n in JOINTS}
VELOCITY={n:float(_limits[n]['velocity']) for n in JOINTS}
ARMATURE={n: (0.025101925 if 'hip_roll' in n or 'knee' in n else 0.010177520 if 'hip_pitch' in n or 'hip_yaw' in n or 'waist' in n else 0.003609725*(2 if 'ankle' in n else 1)) for n in JOINTS}
OMEGA=10*2*3.1415926535
KP={n:a*OMEGA**2 for n,a in ARMATURE.items()}
KD={n:4*a*OMEGA for n,a in ARMATURE.items()}
# Keep source action scaling, while enforcing own hardware torque/velocity limits.
SOURCE_EFFORT={n:(50. if 'ankle' in n else EFFORT[n]) for n in JOINTS}
ACTION_SCALE={n:.25*SOURCE_EFFORT[n]/KP[n] for n in JOINTS}
Q0={n: (-.1 if 'hip_pitch' in n else .3 if 'knee' in n else -.2 if 'ankle_pitch' in n else .35 if 'shoulder_pitch' in n else .87 if 'elbow' in n else (.18 if n.startswith('left') else -.18) if 'shoulder_roll' in n else 0.) for n in JOINTS}
def spawn_robot(prim_path,cfg,translation=None,orientation=None):
    # Source MJCF explicitly excludes pelvis/hip-roll and elbow/wrist pairs.
    from pxr import UsdPhysics
    import omni.usd
    prim=sim.spawn_from_urdf(prim_path,cfg,translation=translation,orientation=orientation)
    stage=omni.usd.get_context().get_stage()
    for path in sim.find_matching_prim_paths(prim_path):
        for a,b in [('pelvis','left_hip_roll_link'),('pelvis','right_hip_roll_link'),('left_elbow_link','left_wrist_roll_link'),('right_elbow_link','right_wrist_roll_link')]:
            node=stage.GetPrimAtPath(path+'/'+a)
            target=stage.GetPrimAtPath(path+'/'+b)
            if node.IsValid() and target.IsValid():
                UsdPhysics.FilteredPairsAPI.Apply(node).CreateFilteredPairsRel().AddTarget(target.GetPath())
    return prim

ROBOT_CFG=UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(func=spawn_robot,asset_path=str(URDF),fix_base=False,merge_fixed_joints=True,
        usd_dir=str(CACHE_DIR/'g1_23dof_mjlab_port_v1'),usd_file_name='g1_23dof_mjlab_port.usd',
        articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=True,solver_position_iteration_count=8,solver_velocity_iteration_count=4)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,.8),rot=(1,0,0,0),joint_pos=Q0,joint_vel={'.*':0.}),
    actuators={'body':ImplicitActuatorCfg(joint_names_expr=JOINTS,stiffness=KP,damping=KD,armature=ARMATURE,effort_limit_sim=EFFORT,velocity_limit_sim=VELOCITY)},
    soft_joint_pos_limit_factor=.9,
    joint_sdk_names=JOINTS[:13]+['','']+JOINTS[13:18]+['','']+JOINTS[18:],
)
