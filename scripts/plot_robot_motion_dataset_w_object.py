from general_motion_retargeting import load_robot_motion_w_object
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from rotation import quat_wxyz_to_euler
import re

def plot_root_obj_hand_pos_rot(motion_data, save_dir):
    motion_file = motion_data["motion_file"]
    motion_fps = motion_data["motion_fps"]
    motion_root_pos = motion_data["motion_root_pos"]
    motion_root_rot = motion_data["motion_root_rot"]
    motion_object_pos = motion_data["motion_object_pos"]
    motion_object_rot = motion_data["motion_object_rot"]
    motion_hand_positions = motion_data.get("motion_hand_positions", None)

    min_len = min([len(motion_object_pos), len(motion_root_pos)])
    if motion_hand_positions is not None:
        min_len = min(min_len, len(motion_hand_positions))

    dt = 1/motion_fps
    tf = dt * min_len
    t = np.linspace(0, (min_len-1)*dt, min_len)

    # Determine figure size and layout based on whether hand data is available
    if motion_hand_positions is not None:
        plt.figure(figsize=(17, 15))  # Taller figure for 6 rows
        nrows = 6
    else:
        plt.figure(figsize=(17, 9))
        nrows = 4
    
    # Row 1: Root position
    plt.subplot(nrows, 3, 1)
    plt.plot(t, motion_root_pos[:min_len, 0], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('x_root [m]')
    plt.grid()

    plt.subplot(nrows, 3, 2)
    plt.plot(t, motion_root_pos[:min_len, 1], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('y_root [m]')    
    plt.grid()

    plt.subplot(nrows, 3, 3)
    plt.plot(t, motion_root_pos[:min_len, 2], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('z_root [m]')    
    plt.grid()

    # Row 2: Object position
    plt.subplot(nrows, 3, 4)
    plt.plot(t, motion_object_pos[:min_len, 0], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('x_obj [m]')
    plt.grid()

    plt.subplot(nrows, 3, 5)
    plt.plot(t, motion_object_pos[:min_len, 1], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('y_obj [m]')    
    plt.grid()

    plt.subplot(nrows, 3, 6)
    plt.plot(t, motion_object_pos[:min_len, 2], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('z_obj [m]')    
    plt.grid()

    # Row 3: Root rotation
    motion_root_rot_eul = quat_wxyz_to_euler(motion_root_rot)

    plt.subplot(nrows, 3, 7)
    plt.plot(t, motion_root_rot_eul[:min_len, 0], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('roll_root [rad]')
    plt.grid()

    plt.subplot(nrows, 3, 8)
    plt.plot(t, motion_root_rot_eul[:min_len, 1], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('pitch_root [rad]')    
    plt.grid()

    plt.subplot(nrows, 3, 9)
    plt.plot(t, motion_root_rot_eul[:min_len, 2], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('yaw_root [rad]')    
    plt.grid()

    # Row 4: Object rotation
    motion_object_rot_eul = quat_wxyz_to_euler(motion_object_rot)

    plt.subplot(nrows, 3, 10)
    plt.plot(t, motion_object_rot_eul[:min_len, 0], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('roll_obj [rad]')
    plt.grid()

    plt.subplot(nrows, 3, 11)
    plt.plot(t, motion_object_rot_eul[:min_len, 1], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('pitch_obj [rad]')    
    plt.grid()

    plt.subplot(nrows, 3, 12)
    plt.plot(t, motion_object_rot_eul[:min_len, 2], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('yaw_obj [rad]')    
    plt.grid()

    # Row 5: Left hand position (if available)
    if motion_hand_positions is not None:
        plt.subplot(nrows, 3, 13)
        plt.plot(t, motion_hand_positions[:min_len, 0], linewidth=2)
        plt.xlabel('t [s]')
        plt.ylabel('x_left_hand [m]')
        plt.grid()

        plt.subplot(nrows, 3, 14)
        plt.plot(t, motion_hand_positions[:min_len, 1], linewidth=2)
        plt.xlabel('t [s]')
        plt.ylabel('y_left_hand [m]')    
        plt.grid()

        plt.subplot(nrows, 3, 15)
        plt.plot(t, motion_hand_positions[:min_len, 2], linewidth=2)
        plt.xlabel('t [s]')
        plt.ylabel('z_left_hand [m]')    
        plt.grid()

        # Row 6: Right hand position
        plt.subplot(nrows, 3, 16)
        plt.plot(t, motion_hand_positions[:min_len, 3], linewidth=2)
        plt.xlabel('t [s]')
        plt.ylabel('x_right_hand [m]')
        plt.grid()

        plt.subplot(nrows, 3, 17)
        plt.plot(t, motion_hand_positions[:min_len, 4], linewidth=2)
        plt.xlabel('t [s]')
        plt.ylabel('y_right_hand [m]')    
        plt.grid()

        plt.subplot(nrows, 3, 18)
        plt.plot(t, motion_hand_positions[:min_len, 5], linewidth=2)
        plt.xlabel('t [s]')
        plt.ylabel('z_right_hand [m]')    
        plt.grid()
    
    f_name, _ = os.path.splitext(motion_file)
    plt.suptitle(f_name)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{f_name}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_folder", type=str, required=True)
    parser.add_argument("--num_motions", type=int)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_folder = args.robot_motion_folder

    save_dir = args.save_dir
    retarget_anem = os.path.basename(robot_motion_folder)
    save_dir = os.path.join(save_dir, retarget_anem)
    os.makedirs(save_dir, exist_ok=True)

    
    if not os.path.exists(robot_motion_folder):
        raise FileNotFoundError(f"Motion data dir {robot_motion_folder} does not exist.")
    
    motion_files = [f for f in os.listdir(robot_motion_folder) if f.endswith('.pkl')]
    motion_files = sorted(motion_files)
    if args.num_motions: 
        motion_files = motion_files[:args.num_motions]
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

    for motion_data in motion_dataset:
        plot_root_obj_hand_pos_rot(motion_data, save_dir)
