from general_motion_retargeting import load_robot_motion_w_object, load_robot_motion_model_w_object
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from rotation import quat_wxyz_to_euler
import re

def plot_root_obj_pos_rot(model_data, retargeted_data, save_dir):
    motion_file = model_data["motion_file"]
    motion_fps_model = model_data["motion_fps"]
    model_root_pos = model_data["motion_root_pos"]
    model_root_rot = model_data["motion_root_rot"]
    model_object_pos = model_data["motion_object_pos"]
    model_object_rot = model_data["motion_object_rot"]
    source_start = model_data["source_start"]

    min_len_model = min([len(model_object_pos), len(model_root_pos)])

    # dt_model = 1/motion_fps_model
    # start_t = source_start * dt_model
    # tf_model = (dt_model * min_len_model) + start_t
    # t_model = np.arange(start_t, tf_model, dt_model)

    dt_model = 1/motion_fps_model
    start_t = source_start * dt_model
    tf_model = (min_len_model-1)*dt_model + start_t
    t_model = np.linspace(start_t, tf_model, min_len_model)



    motion_fps_retargeted = retargeted_data["motion_fps"]
    retargeted_root_pos = retargeted_data["motion_root_pos"]
    retargeted_root_rot = retargeted_data["motion_root_rot"]
    retargeted_object_pos = retargeted_data["motion_object_pos"]
    retargeted_object_rot = retargeted_data["motion_object_rot"]

    min_len_retargeted = min([len(retargeted_object_pos), len(retargeted_root_pos)])

    # dt_retargeted = 1/motion_fps_retargeted
    # tf_retargeted = dt_retargeted * min_len_retargeted
    # t_retargeted = np.arange(0, tf_retargeted, dt_retargeted)

    dt_retargeted = 1/motion_fps_retargeted
    start_t = source_start * dt_retargeted
    tf_retargeted = (min_len_retargeted-1)*dt_retargeted
    t_retargeted = np.linspace(0, tf_retargeted, min_len_retargeted)



    plt.figure(figsize=(17, 9))

    # RETARGETED 

    plt.subplot(4,3,1)
    plt.plot(t_retargeted, retargeted_root_pos[:, 0], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('x_root [m]')
    plt.grid()

    plt.subplot(4,3,2)
    plt.plot(t_retargeted, retargeted_root_pos[:, 1], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('y_root [m]')    
    plt.grid()

    plt.subplot(4,3,3)
    plt.plot(t_retargeted, retargeted_root_pos[:, 2], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('z_root [m]')    
    plt.grid()

    plt.subplot(4,3,4)
    plt.plot(t_retargeted, retargeted_object_pos[:, 0], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('x_obj [m]')
    plt.grid()

    plt.subplot(4,3,5)
    plt.plot(t_retargeted, retargeted_object_pos[:, 1], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('y_obj [m]')    
    plt.grid()

    plt.subplot(4,3,6)
    plt.plot(t_retargeted, retargeted_object_pos[:, 2], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('z_obj [m]')    
    plt.grid()

    retargeted_root_rot_eul = quat_wxyz_to_euler(retargeted_root_rot)

    plt.subplot(4,3,7)
    plt.plot(t_retargeted, retargeted_root_rot_eul[:, 0], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('roll_root [rad]')
    plt.grid()

    plt.subplot(4,3,8)
    plt.plot(t_retargeted, retargeted_root_rot_eul[:, 1], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('pitch_root [rad]')    
    plt.grid()

    plt.subplot(4,3,9)
    plt.plot(t_retargeted, retargeted_root_rot_eul[:, 2], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('yaw_root [rad]')    
    plt.grid()

    retargeted_object_rot_eul = quat_wxyz_to_euler(retargeted_object_rot)

    plt.subplot(4,3,10)
    plt.plot(t_retargeted, retargeted_object_rot_eul[:, 0], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('roll_obj [rad]')
    plt.grid()

    plt.subplot(4,3,11)
    plt.plot(t_retargeted, retargeted_object_rot_eul[:, 1], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('pitch_obj [rad]')    
    plt.grid()

    plt.subplot(4,3,12)
    plt.plot(t_retargeted, retargeted_object_rot_eul[:, 2], linewidth=2, label="retargeted")
    plt.xlabel('t [s]')
    plt.ylabel('yaw_obj [rad]')    
    plt.grid()

    # MODEL

    plt.subplot(4,3,1)
    plt.plot(t_model, model_root_pos[:, 0], linewidth=2, label="model")
    plt.xlabel('t [s]')
    plt.ylabel('x_root [m]')
    plt.legend()


    plt.subplot(4,3,2)
    plt.plot(t_model, model_root_pos[:, 1], linewidth=2, label="model")
    plt.xlabel('t [s]')
    plt.ylabel('y_root [m]')    

    plt.subplot(4,3,3)
    plt.plot(t_model, model_root_pos[:, 2], linewidth=2, label="model")
    plt.xlabel('t [s]')
    plt.ylabel('z_root [m]')    

    plt.subplot(4,3,4)
    plt.plot(t_model, model_object_pos[:, 0], linewidth=2, label="model")
    plt.xlabel('t [s]')
    plt.ylabel('x_obj [m]')

    plt.subplot(4,3,5)
    plt.plot(t_model, model_object_pos[:, 1], linewidth=2, label="model")
    plt.xlabel('t [s]')
    plt.ylabel('y_obj [m]')    

    plt.subplot(4,3,6)
    plt.plot(t_model, model_object_pos[:, 2], linewidth=2, label="model")
    plt.xlabel('t [s]')
    plt.ylabel('z_obj [m]')    

    model_root_rot_eul = quat_wxyz_to_euler(model_root_rot)

    plt.subplot(4,3,7)
    plt.plot(t_model, model_root_rot_eul[:, 0], linewidth=2, label="model")
    plt.xlabel('t [s]')
    plt.ylabel('roll_root [rad]')

    plt.subplot(4,3,8)
    plt.plot(t_model, model_root_rot_eul[:, 1], linewidth=2, label="model")
    plt.xlabel('t [s]')
    plt.ylabel('pitch_root [rad]')    

    plt.subplot(4,3,9)
    plt.plot(t_model, model_root_rot_eul[:, 2], linewidth=2, label="model")
    plt.xlabel('t [s]')
    plt.ylabel('yaw_root [rad]')    

    model_object_rot_eul = quat_wxyz_to_euler(model_object_rot)

    plt.subplot(4,3,10)
    plt.plot(t_model, model_object_rot_eul[:, 0], linewidth=2, label="model")
    plt.xlabel('t [s]')
    plt.ylabel('roll_obj [rad]')

    plt.subplot(4,3,11)
    plt.plot(t_model, model_object_rot_eul[:, 1], linewidth=2, label="model")
    plt.xlabel('t [s]')
    plt.ylabel('pitch_obj [rad]')    

    plt.subplot(4,3,12)
    plt.plot(t_model, model_object_rot_eul[:, 2], linewidth=2, label="model")
    plt.xlabel('t [s]')
    plt.ylabel('yaw_obj [rad]')    

    

    # TITLE
    
    f_name, _ = os.path.splitext(motion_file)
    plt.suptitle(f_name)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{f_name}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_model_folder", type=str, required=True)
    parser.add_argument("--robot_motion_retargeted_folder", type=str, required=True)
    parser.add_argument("--num_motions", type=int)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_model_folder = args.robot_motion_model_folder
    robot_motion_retargeted_folder = args.robot_motion_retargeted_folder

    save_dir = args.save_dir
    pattern = r'logs/(.*?)/samples/(.*)'
    match = re.search(pattern, robot_motion_model_folder)
    if match:
        log_id = match.group(1)
        sample_id = match.group(2)
        save_dir = os.path.join(save_dir, log_id, sample_id)
    else:
        raise FileNotFoundError("IDs not found.")
        
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(robot_motion_model_folder):
        raise FileNotFoundError(f"Motion data dir {robot_motion_model_folder} does not exist.")
    
    if not os.path.exists(robot_motion_retargeted_folder):
        raise FileNotFoundError(f"Motion data dir {robot_motion_retargeted_folder} does not exist.")
    
    motion_model_files = [f for f in os.listdir(robot_motion_model_folder) if f.endswith('.pkl')]
    motion_model_files = sorted(motion_model_files)

    motion_num = len(motion_model_files)
    print(f"Found {motion_num} motion files in {robot_motion_model_folder}, loading...")
    motion_model_dataset = []
    motion_retargeted_dataset = []

    for motion_model_file in tqdm(motion_model_files):
        # print(motion_model_file)
        # exit()
        motion_model_file_base = re.sub(r'^.*?sub(\w*)_sample.*?(\..+?)$', r'sub\1\2', motion_model_file)
        motion_retargeted_path = os.path.join(robot_motion_retargeted_folder, motion_model_file_base)
        # Check if the path exists AND if it is a file
        if os.path.exists(motion_retargeted_path) and os.path.isfile(motion_retargeted_path):

            motion_model_path = os.path.join(robot_motion_model_folder, motion_model_file)
            source_start, motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_object_pos, motion_object_rot, motion_local_body_pos, motion_link_body_list = load_robot_motion_model_w_object(motion_model_path)
            motion_model_dataset.append({
                "motion_file": motion_model_file,
                "motion_data": motion_data,
                "motion_fps": motion_fps,
                "motion_root_pos": motion_root_pos,
                "motion_root_rot": motion_root_rot,
                "motion_dof_pos": motion_dof_pos,
                "motion_object_pos": motion_object_pos,
                "motion_object_rot": motion_object_rot,
                "motion_local_body_pos": motion_local_body_pos,
                "motion_link_body_list": motion_link_body_list,
                "source_start": source_start
            })

            motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_object_pos, motion_object_rot, motion_local_body_pos, motion_link_body_list = load_robot_motion_w_object(motion_retargeted_path)
            motion_retargeted_dataset.append({
                "motion_file": motion_model_file,
                "motion_data": motion_data,
                "motion_fps": motion_fps,
                "motion_root_pos": motion_root_pos,
                "motion_root_rot": motion_root_rot,
                "motion_dof_pos": motion_dof_pos,
                "motion_object_pos": motion_object_pos,
                "motion_object_rot": motion_object_rot,
                "motion_local_body_pos": motion_local_body_pos,
                "motion_link_body_list": motion_link_body_list,
            })
    print("Loading done.")


    for model_data, retargeted_data in zip(motion_model_dataset[:args.num_motions], motion_retargeted_dataset[:args.num_motions]):
        plot_root_obj_pos_rot(model_data, retargeted_data, save_dir)