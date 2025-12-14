import pickle

def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
        motion_fps = motion_data["fps"]
        motion_root_pos = motion_data["root_pos"]
        motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]] # from xyzw to wxyz
        motion_dof_pos = motion_data["dof_pos"]
        motion_local_body_pos = motion_data["local_body_pos"]
        motion_link_body_list = motion_data["link_body_list"]
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list

def load_robot_motion_w_object(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
        motion_fps = motion_data["fps"]
        motion_root_pos = motion_data["root_pos"]
        motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]] # from xyzw to wxyz
        motion_dof_pos = motion_data["dof_pos"]
        motion_object_pos = motion_data["object_pos"]
        motion_object_rot = motion_data["object_rot"][:, [3, 0, 1, 2]] # from xyzw to wxyz
        motion_local_body_pos = motion_data["local_body_pos"]
        motion_link_body_list = motion_data["link_body_list"]
        motion_hand_positions = motion_data.get("hand_positions", None)
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_object_pos, motion_object_rot, motion_local_body_pos, motion_link_body_list, motion_hand_positions

def load_robot_motion_model_w_object(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
        motion_fps = motion_data["fps"]
        motion_root_pos = motion_data["root_pos"]
        motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]] # from xyzw to wxyz
        motion_dof_pos = motion_data["dof_pos"]
        motion_object_pos = motion_data["object_pos"]
        motion_object_rot = motion_data["object_rot"][:, [3, 0, 1, 2]] # from xyzw to wxyz
        motion_local_body_pos = motion_data["local_body_pos"]
        motion_link_body_list = motion_data["link_body_list"]
        source_start = motion_data["source_start"]
        motion_hand_positions = motion_data.get("hand_positions", None)

    return source_start, motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_object_pos, motion_object_rot, motion_local_body_pos, motion_link_body_list, motion_hand_positions
