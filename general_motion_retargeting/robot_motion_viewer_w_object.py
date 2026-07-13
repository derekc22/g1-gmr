import os
import re
import time
import tempfile
from pathlib import Path
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print


def _xml_escape_attr(value):
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _center_obj_mesh(mesh_path, output_path, scale=1.0):
    """Write a copy of an OBJ whose vertex centroid is at the local origin."""
    mesh_path = Path(mesh_path)
    output_path = Path(output_path)
    lines = mesh_path.read_text().splitlines()

    vertices = []
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not vertices:
        raise ValueError(f"No vertices found in object mesh: {mesh_path}")

    center = np.asarray(vertices, dtype=np.float64).mean(axis=0)
    vertex_idx = 0
    with output_path.open("w") as f:
        for line in lines:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    xyz = (np.asarray(vertices[vertex_idx], dtype=np.float64) - center) * scale
                    rest = parts[4:]
                    f.write(f"v {xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}")
                    if rest:
                        f.write(" " + " ".join(rest))
                    f.write("\n")
                    vertex_idx += 1
                    continue
            f.write(line + "\n")

    return center


def _build_object_mesh_xml(base_xml_path, object_mesh_path, object_mesh_scale=1.0):
    """Create a temporary MuJoCo XML replacing the placeholder object box with a mesh."""
    base_xml_path = Path(base_xml_path)
    object_mesh_path = Path(object_mesh_path)
    if not object_mesh_path.exists():
        raise FileNotFoundError(f"Object mesh not found: {object_mesh_path}")

    tmpdir = tempfile.TemporaryDirectory(prefix="gmr_object_mesh_")
    tmp_path = Path(tmpdir.name)
    centered_mesh_path = tmp_path / f"{object_mesh_path.stem}_centered.obj"
    _center_obj_mesh(object_mesh_path, centered_mesh_path, scale=object_mesh_scale)

    xml = base_xml_path.read_text()
    def absolutize_meshdir(match):
        meshdir = Path(match.group(2))
        if not meshdir.is_absolute():
            meshdir = base_xml_path.parent / meshdir
        return f'{match.group(1)}{_xml_escape_attr(meshdir)}{match.group(3)}'

    xml = re.sub(
        r'(<compiler\b[^>]*\bmeshdir=")([^"]*)(")',
        absolutize_meshdir,
        xml,
        count=1,
    )

    mesh_asset = (
        f'    <mesh name="render_object_mesh" '
        f'file="{_xml_escape_attr(centered_mesh_path)}"/>\n'
    )
    if "</asset>" not in xml:
        tmpdir.cleanup()
        raise ValueError(f"Could not find </asset> in {base_xml_path}")
    xml = xml.replace("  </asset>", mesh_asset + "  </asset>", 1)

    object_body = """    <body name="object">
      <joint name="object" type="free"/>
      <geom name="object" type="mesh" mesh="render_object_mesh" mass="0.01" contype="0" conaffinity="0" rgba="0.72 0.58 0.42 1"/>
    </body>"""

    pattern = re.compile(
        r"\s*<body name=\"object\">\s*"
        r"<joint name=\"object\" type=\"free\"\s*/>\s*"
        r"<geom name=\"object\" type=\"box\"[^>]*(?:></geom>|/>)\s*"
        r"</body>",
        re.DOTALL,
    )
    xml, num_replaced = pattern.subn("\n" + object_body, xml, count=1)
    if num_replaced != 1:
        tmpdir.cleanup()
        raise ValueError(f"Could not replace placeholder object body in {base_xml_path}")

    xml_path = tmp_path / f"{base_xml_path.stem}_{object_mesh_path.stem}.xml"
    xml_path.write_text(xml)
    return tmpdir, xml_path


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

class RobotMotionViewerWithObject:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                # video recording
                record_video=False,
                video_path=None,
                video_width=640,
                video_height=480,
                keyboard_callback=None,
                object_mesh_path=None,
                object_mesh_scale=1.0,
                ):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self._object_mesh_tmpdir = None
        if object_mesh_path is not None:
            self._object_mesh_tmpdir, self.xml_path = _build_object_mesh_xml(
                self.xml_path,
                object_mesh_path,
                object_mesh_scale=object_mesh_scale,
            )
            print(f"Rendering object mesh: {object_mesh_path} (scale={object_mesh_scale:.6g})")
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video


        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False, 
            key_callback=keyboard_callback
            )      

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot
        
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
        
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos,
            # object data
            obj_pos, obj_rot,
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True, 
            follow_camera=True,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
        self.data.qpos[7:36] = dof_pos
        self.data.qpos[36:39] = obj_pos
        self.data.qpos[39:] = obj_rot
        
        mj.mj_forward(self.model, self.data)
        
        if follow_camera:
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.viewer.cam.distance = self.viewer_cam_distance
            self.viewer.cam.elevation = -10  # 正面视角，轻微向下看
            # self.viewer.cam.azimuth = 180    # 正面朝向机器人
        
        if human_motion_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0
            # Draw the task targets for reference
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None
                    )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)
    
    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
        if self._object_mesh_tmpdir is not None:
            self._object_mesh_tmpdir.cleanup()
