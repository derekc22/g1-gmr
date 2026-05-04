from general_motion_retargeting import RobotMotionViewerWithObject, load_robot_motion_w_object
import argparse
import os
import pickle
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

paused = False
motion_num = 0
motion_id = 0
current_motion_id = -1
terminate = False
DEFAULT_OBJECTS_DIR = "/home/learning/Documents/omomo_release/data/captured_objects"

def keyboard_callback(keycode):
    global paused, motion_id, motion_num, terminate
    if chr(keycode) == ' ':
        paused = not paused
    if chr(keycode) == '[':
        motion_id = (motion_id - 1) % motion_num
    if chr(keycode) == ']':
        motion_id = (motion_id + 1) % motion_num
    if chr(keycode) == '.':
        terminate = True


def infer_object_name(motion_data, motion_file, reference_data=None):
    """Infer OMOMO object name from metadata or filenames like sub1_clothesstand_000.pkl."""
    if reference_data is not None:
        object_name = reference_data.get("object_name")
        if object_name:
            return str(object_name)

    object_name = motion_data.get("object_name")
    if object_name:
        return str(object_name)

    mesh_file = motion_data.get("mesh_file")
    if mesh_file:
        stem = os.path.splitext(os.path.basename(str(mesh_file)))[0]
        return stem.replace("_cleaned_simplified", "")

    seq_name = motion_data.get("seq_name") or os.path.splitext(motion_file)[0]
    seq_name = os.path.splitext(os.path.basename(str(seq_name)))[0]
    match = re.match(r"^sub\d+_(.+)_\d+$", seq_name)
    if match:
        return match.group(1)

    parts = seq_name.split("_")
    if len(parts) >= 3 and parts[0].startswith("sub") and parts[-1].isdigit():
        return "_".join(parts[1:-1])
    return None


def load_obj_vertices(mesh_path):
    vertices = []
    with open(mesh_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        return None
    return np.asarray(vertices, dtype=np.float64)


def find_object_mesh(object_name, objects_dir):
    if not object_name:
        return None

    candidates = [
        f"{object_name}_cleaned_simplified.obj",
        f"{object_name}.obj",
        f"{object_name.lower()}_cleaned_simplified.obj",
        f"{object_name.lower()}.obj",
    ]
    for candidate in candidates:
        path = os.path.join(objects_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def load_reference_motion(reference_motion_folder, motion_data, motion_file):
    if not reference_motion_folder:
        return None

    seq_name = motion_data.get("seq_name") or os.path.splitext(motion_file)[0]
    seq_name = os.path.splitext(os.path.basename(str(seq_name)))[0]
    path = os.path.join(reference_motion_folder, f"{seq_name}.pkl")
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


def infer_object_mesh_scale(motion_data, mesh_path):
    if motion_data is None:
        return None

    if "object_mesh_scale" in motion_data:
        return float(motion_data["object_mesh_scale"])

    if "object_verts" not in motion_data:
        return None

    mesh_vertices = load_obj_vertices(mesh_path)
    if mesh_vertices is None:
        return None

    object_verts = np.asarray(motion_data["object_verts"], dtype=np.float64)
    if object_verts.ndim != 3 or object_verts.shape[1] != mesh_vertices.shape[0]:
        return None

    if "object_rotation" in motion_data:
        object_rotation = np.asarray(motion_data["object_rotation"], dtype=np.float64)[0]
    elif "object_rot" in motion_data:
        object_rotation = R.from_quat(np.asarray(motion_data["object_rot"], dtype=np.float64)[0]).as_matrix()
    else:
        object_rotation = np.eye(3)

    mesh_centered = mesh_vertices - mesh_vertices.mean(axis=0)
    fitted = (object_rotation @ mesh_centered.T).T
    target = object_verts[0] - object_verts[0].mean(axis=0)
    denom = float(np.sum(fitted * fitted))
    if denom <= 0.0:
        return None

    scale = float(np.sum(fitted * target) / denom)
    if scale <= 0.0:
        return None
    return scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_folder", type=str, required=True)
    parser.add_argument("--objects_dir", type=str, default=DEFAULT_OBJECTS_DIR)
    parser.add_argument("--object_name", type=str, default=None)
    parser.add_argument("--object_mesh_file", type=str, default=None)
    parser.add_argument("--object_scale", type=float, default=None)
    parser.add_argument("--reference_motion_folder", type=str, default=None)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--save_dir", type=str, 
                        default="videos")
    parser.add_argument("--auto", action="store_true")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_folder = args.robot_motion_folder
    
    if not os.path.exists(robot_motion_folder):
        raise FileNotFoundError(f"Motion data dir {robot_motion_folder} does not exist.")
    
    # Parse robot_motion_folder to extract log_id and sample_id for video naming
    save_dir = args.save_dir
    pattern = r'logs/(.*?)/samples/(.*)'
    match = re.search(pattern, robot_motion_folder)
    if match:
        log_id = match.group(1)
        sample_id = match.group(2)
        video_dir = os.path.join(save_dir, log_id, sample_id)
        video_path = os.path.join(video_dir, "render.mp4")
    else:
        # Fallback: use folder name as video name
        folder_name = os.path.basename(robot_motion_folder.rstrip('/'))
        video_dir = save_dir
        video_path = os.path.join(video_dir, f"{folder_name}.mp4")
    
    os.makedirs(video_dir, exist_ok=True)
    print(f"Video will be saved to: {video_path}")
    
    motion_files = [f for f in os.listdir(robot_motion_folder) if f.endswith('.pkl')]
    motion_files = sorted(motion_files)
    motion_num = len(motion_files)
    print(f"Found {motion_num} motion files in {robot_motion_folder}, loading...")
    motion_dataset = []
    for motion_file in tqdm(motion_files):
        motion_path = os.path.join(robot_motion_folder, motion_file)
        motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_object_pos, motion_object_rot, motion_local_body_pos, motion_link_body_list, motion_hand_positions = load_robot_motion_w_object(motion_path)
        reference_data = load_reference_motion(args.reference_motion_folder, motion_data, motion_file)
        motion_dataset.append({
            "motion_file": motion_file,
            "motion_data": motion_data,
            "motion_fps": motion_fps,
            "motion_root_pos": motion_root_pos,
            "motion_root_rot": motion_root_rot,
            "motion_dof_pos": motion_dof_pos,
            "motion_object_pos": motion_object_pos,
            "motion_object_rot": motion_object_rot,
            "motion_local_body_pos": motion_local_body_pos,
            "motion_link_body_list": motion_link_body_list,
            "motion_hand_positions": motion_hand_positions,
            "reference_data": reference_data,
            "object_name": infer_object_name(motion_data, motion_file, reference_data=reference_data),
        })
    print("Loading done.")

    object_mesh_path = args.object_mesh_file
    if object_mesh_path is None:
        object_name = args.object_name
        if object_name is None:
            object_names = sorted({
                item["object_name"]
                for item in motion_dataset
                if item["object_name"] is not None
            })
            if len(object_names) > 1:
                print(f"Multiple object types found {object_names}; using {object_names[0]} for this viewer.")
            object_name = object_names[0] if object_names else None
        object_mesh_path = find_object_mesh(object_name, args.objects_dir)

    object_mesh_scale = args.object_scale
    if object_mesh_scale is None and object_mesh_path is not None:
        for item in motion_dataset:
            object_mesh_scale = infer_object_mesh_scale(item["motion_data"], object_mesh_path)
            if object_mesh_scale is None:
                object_mesh_scale = infer_object_mesh_scale(item["reference_data"], object_mesh_path)
            if object_mesh_scale is not None:
                break
    if object_mesh_scale is None:
        object_mesh_scale = 1.0

    if object_mesh_path is None:
        print("No object mesh found; falling back to placeholder box.")
    else:
        print(f"Object mesh: {object_mesh_path}")
        print(f"Object mesh scale: {object_mesh_scale:.6g}")

    env = RobotMotionViewerWithObject(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=video_path, 
                            keyboard_callback=keyboard_callback,
                            object_mesh_path=object_mesh_path,
                            object_mesh_scale=object_mesh_scale)
    
    frame_idx = 0
    while True and not terminate:
        # get current motion
        if current_motion_id != motion_id:
            current_motion_id = motion_id
            frame_idx = 0
            motion_data = motion_dataset[motion_id]
            motion_file = motion_data["motion_file"]
            motion_fps = motion_data["motion_fps"]
            motion_root_pos = motion_data["motion_root_pos"]
            motion_root_rot = motion_data["motion_root_rot"]
            motion_dof_pos = motion_data["motion_dof_pos"]
            motion_object_pos = motion_data["motion_object_pos"]
            motion_object_rot = motion_data["motion_object_rot"]
            print(f"Switched to motion {motion_id}: {motion_file}, fps: {motion_fps}, num_frames: {len(motion_root_pos)}")
        
        min_len = min([len(motion_object_pos), len(motion_root_pos)])
        if not paused:
            env.step(motion_root_pos[frame_idx], 
                    motion_root_rot[frame_idx], 
                    motion_dof_pos[frame_idx],
                    motion_object_pos[frame_idx],
                    motion_object_rot[frame_idx], 
                    rate_limit=True)
            frame_idx += 1
            if frame_idx >= min_len:
                if args.auto:
                    motion_id += 1
                    if motion_id == motion_num:
                        terminate = True
                else:
                    frame_idx = 0
    env.close()
