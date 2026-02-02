from general_motion_retargeting import RobotMotionViewerWithObject, load_robot_motion_w_object
import argparse
import os
import re
from tqdm import tqdm

paused = False
motion_num = 0
motion_id = 0
current_motion_id = -1
terminate = False

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_folder", type=str, required=True)

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
        })
    print("Loading done.")

    env = RobotMotionViewerWithObject(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=video_path, 
                            keyboard_callback=keyboard_callback)
    
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