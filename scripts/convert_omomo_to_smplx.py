import os
import joblib
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target_dir", type=str, required=True)
parser.add_argument("--num_motions", type=int)

args = parser.parse_args()


# these paths are from the original OMOMO dataset
motion_path1 = "/home/learning/Documents/g1-gmr/OMOMO_DATA/OMOMO_p_files/train_diffusion_manip_seq_joints24.p"
motion_path2 = "/home/learning/Documents/g1-gmr/OMOMO_DATA/OMOMO_p_files/test_diffusion_manip_seq_joints24.p"
all_motion_data1 = joblib.load(motion_path1)
all_motion_data2 = joblib.load(motion_path2)

# save as individual files
target_dir = args.target_dir
os.makedirs(target_dir, exist_ok=True)
for motion_data in [all_motion_data1, all_motion_data2]:
    keys = list(motion_data.keys())
    if args.num_motions:
        keys = keys[:args.num_motions]
    for data_name in keys:

        smpl_data = motion_data[data_name]
        seq_name = smpl_data['seq_name']
        # save as npz
        num_frames = smpl_data["pose_body"].shape[0]
        mocap_frame_rate = 30
        poses = np.concatenate([smpl_data["pose_body"], 
                                np.zeros((num_frames, 102))],
                                axis=1)
        smpl_data["poses"] = poses
        smpl_data["mocap_frame_rate"] = np.array(mocap_frame_rate)

        obj_trans = smpl_data.get("obj_trans", None)
        obj_com_pos = smpl_data.get("obj_com_pos", None)
        if obj_trans is not None:
            object_pos = np.asarray(obj_trans, dtype=np.float32).reshape(num_frames, 3)
        elif obj_com_pos is not None:
            object_pos = np.asarray(obj_com_pos, dtype=np.float32).reshape(num_frames, 3)
        else:
            object_pos = np.zeros((num_frames, 3), dtype=np.float32)

        obj_rot = smpl_data.get("obj_rot", None)
        if obj_rot is not None:
            object_rot = np.asarray(obj_rot, dtype=np.float32).reshape(num_frames, 3, 3)
        else:
            object_rot = np.zeros((num_frames, 3, 3), dtype=np.float32)

        smpl_data["object_pos"] = object_pos.astype(np.float32)
        smpl_data["object_rot"] = object_rot.astype(np.float32)
        
        # use pickle to save
        with open(f"{target_dir}/{seq_name}.pkl", "wb") as f:
            pickle.dump(smpl_data, f)
        print(f"saved {seq_name}")