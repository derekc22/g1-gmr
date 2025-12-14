#!/usr/bin/env python3
"""
Unified OMOMO to Robot Dataset Converter

Converts OMOMO .p files directly to robot motion datasets with object geometry.
Combines the functionality of:
  - convert_omomo_to_smplx.py (OMOMO -> SMPL-X)
  - smplx_to_robot_dataset.py (SMPL-X -> Robot)
  - extract_obj_geom.py (Object geometry extraction)

Output per sequence (.pkl):
  Robot motion:
    - fps, root_pos, root_rot, dof_pos, local_body_pos, link_body_list, seq_name
  Object representation (OMOMO-faithful):
    - object_verts: (T, K, 3) - reconstructed mesh vertices
    - object_centroid: (T, 3) - g_t = mean(V_t)
    - object_rotation: (T, 3, 3) - R_t rotation matrices
    - bps_encoding: (T, 1024, 3) - BPS d(B_t, V_t)
  Object representation (visualization-compatible):
    - object_pos: (T, 3) - alias for object_centroid
    - object_rot: (T, 4) - quaternion (xyzw) from object_rotation
  Hand positions (OMOMO Stage 1):
    - hand_positions: (T, 6) - [left_x, left_y, left_z, right_x, right_y, right_z]
  For articulated objects (mop, vacuum):
    - vertex_part_id: (K,) - 0=top, 1=bottom
    - part_rotations: (T, 2, 3, 3) - [top_rot, bottom_rot]

Usage:
    python convert_omomo_to_robot_dataset.py \\
        --omomo_files train_diffusion_manip_seq_joints24.p test_diffusion_manip_seq_joints24.p \\
        --objects_dir /path/to/captured_objects \\
        --smplx_model_dir /path/to/body_models \\
        --output_dir output/ \\
        --robot unitree_g1
"""

import argparse
import gc
import json
import os
import pathlib
import pickle
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

try:
    import joblib
except ImportError:
    sys.exit("Error: pip install joblib")

try:
    import smplx
except ImportError:
    sys.exit("Error: pip install smplx")

# Import GMR modules
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel
from rotation import mat_to_quat_xyzw


# =============================================================================
# MESH LOADING (from extract_obj_geom.py)
# =============================================================================

def load_obj_mesh(path: str) -> np.ndarray:
    """Load vertices from .obj file."""
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"No vertices in {path}")
    return np.array(verts, dtype=np.float32)


def build_mesh_map(objects_dir: str) -> Dict[str, Tuple[str, np.ndarray]]:
    """Load object meshes. Returns {object_name: (path, vertices)}."""
    if not os.path.isdir(objects_dir):
        sys.exit(f"Error: {objects_dir} not found")
    
    mesh_map = {}
    for fname in sorted(os.listdir(objects_dir)):
        if not fname.endswith('.obj'):
            continue
        name = fname.replace('_cleaned_simplified', '').replace('_cleaned', '').replace('.obj', '')
        if name in mesh_map and 'cleaned_simplified' not in fname:
            continue
        path = os.path.join(objects_dir, fname)
        mesh_map[name] = (path, load_obj_mesh(path))
    
    return mesh_map


ARTICULATED_OBJECTS = {'mop', 'vacuum'}


# =============================================================================
# BPS ENCODING (from extract_obj_geom.py)
# =============================================================================

def generate_bps_points(n: int = 1024, seed: int = 42) -> np.ndarray:
    """Sample n points uniformly in unit ball."""
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < n:
        p = rng.uniform(-1, 1, (n * 2, 3))
        points.extend(p[np.linalg.norm(p, axis=1) <= 1].tolist())
    return np.array(points[:n], dtype=np.float32)


def compute_bps(verts: np.ndarray, centroid: np.ndarray, 
                basis: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Compute BPS encoding d(B_t, V_t) for each frame."""
    T = verts.shape[0]
    bps = np.zeros((T, len(basis), 3), dtype=np.float32)
    for t in range(T):
        B_t = basis * radius + centroid[t]
        tree = cKDTree(verts[t])
        _, idx = tree.query(B_t)
        bps[t] = B_t - verts[t, idx]
    return bps


# =============================================================================
# VERTEX RECONSTRUCTION (from extract_obj_geom.py)
# =============================================================================

def reconstruct_verts(V0: np.ndarray, R: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Reconstruct per-frame vertices: V_t = R_t @ (s_t * V0) + t_t"""
    V0_64 = V0.astype(np.float64)
    V0_scaled = s[:, None, None] * V0_64[None, :, :]
    rotated = np.einsum('tij,tkj->tki', R, V0_scaled)
    return (rotated + t[:, None, :]).astype(np.float32)


def get_object_name(seq_name: str) -> str:
    """Extract object name from OMOMO seq_name: 'sub10_clothesstand_000' -> 'clothesstand'"""
    parts = seq_name.split('_')
    obj_parts = [p for i, p in enumerate(parts) 
                 if not (i == 0 and p.startswith('sub')) and not (i == len(parts)-1 and p.isdigit())]
    return '_'.join(obj_parts) if obj_parts else seq_name


# =============================================================================
# SMPL-X LOADING (adapted from load_smplx_file)
# =============================================================================

def load_smplx_from_omomo(seq_data: dict, smplx_model_path: str):
    """
    Load SMPL-X body model and compute output from OMOMO sequence data.
    Returns: smplx_data dict, body_model, smplx_output, human_height
    """
    num_frames = seq_data["pose_body"].shape[0]
    
    # Prepare SMPL-X data dict (mimic convert_omomo_to_smplx.py format)
    poses = np.concatenate([seq_data["pose_body"], np.zeros((num_frames, 102))], axis=1)
    
    smplx_data = {
        'poses': poses,
        'pose_body': seq_data["pose_body"],
        'betas': seq_data["betas"],
        'root_orient': seq_data["root_orient"],
        'trans': seq_data["trans"],
        'gender': seq_data.get("gender", "neutral"),
        'mocap_frame_rate': np.array(30),
        'seq_name': seq_data.get('seq_name', 'unknown'),
    }
    
    # Get number of betas
    betas_raw = smplx_data["betas"]
    if betas_raw.ndim == 1:
        num_betas = betas_raw.shape[0]
    else:
        num_betas = betas_raw.shape[-1]
    
    # Handle gender
    gender = str(smplx_data["gender"])
    if isinstance(gender, bytes):
        gender = gender.decode('utf-8')
    
    # Create body model
    body_model = smplx.create(
        smplx_model_path,
        "smplx",
        gender=gender,
        use_pca=False,
        num_betas=num_betas,
    )
    
    # Prepare betas
    betas_array = betas_raw
    if betas_array.ndim == 2 and betas_array.shape[0] == 1:
        betas_array = np.repeat(betas_array, num_frames, axis=0)
    elif betas_array.ndim == 1:
        betas_array = np.repeat(betas_array.reshape(1, -1), num_frames, axis=0)
    
    # Run body model
    smplx_output = body_model(
        betas=torch.tensor(betas_array).float(),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    
    # Compute human height
    if len(smplx_data["betas"].shape) == 1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]
    
    return smplx_data, body_model, smplx_output, human_height


# =============================================================================
# OBJECT GEOMETRY EXTRACTION
# =============================================================================

def extract_object_geometry(
    seq_data: dict, 
    mesh_map: dict, 
    bps_basis: np.ndarray,
    z_offset: float,
    xy_offset: np.ndarray,
    bps_radius: float = 1.0,
    centroid_tol: float = 0.05,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Extract object geometry and apply coordinate offsets.
    
    Args:
        seq_data: OMOMO sequence data
        mesh_map: {object_name: (path, vertices)}
        bps_basis: (1024, 3) BPS basis points
        z_offset: Height offset (from robot adjustment)
        xy_offset: XY offset (from robot root origin)
        bps_radius: BPS ball radius
        centroid_tol: Max allowed centroid validation error
    
    Returns:
        (result_dict, None) on success, (None, error_msg) on failure
    """
    seq_name = seq_data.get('seq_name', 'unknown')
    obj_rot = seq_data.get('obj_rot')
    obj_trans = seq_data.get('obj_trans')
    obj_scale = seq_data.get('obj_scale')
    obj_com = seq_data.get('obj_com_pos')
    
    if obj_rot is None or obj_trans is None:
        return None, "missing obj_rot or obj_trans"
    
    # Reshape
    obj_trans = np.asarray(obj_trans).squeeze(-1)
    obj_rot = np.asarray(obj_rot)
    T = obj_rot.shape[0]
    
    # Per-frame scale
    if obj_scale is not None:
        scale = np.asarray(obj_scale, dtype=np.float64)
        if scale.ndim == 0:
            scale = np.full(T, float(scale))
    else:
        scale = np.ones(T, dtype=np.float64)
    
    # Find mesh
    obj_name = get_object_name(seq_name)
    if obj_name not in mesh_map:
        obj_name_lower = obj_name.lower()
        if obj_name_lower in mesh_map:
            obj_name = obj_name_lower
        else:
            return None, f"mesh not found: {obj_name}"
    
    # Check articulated
    is_articulated = obj_name in ARTICULATED_OBJECTS
    has_bottom_transforms = (seq_data.get('obj_bottom_rot') is not None and 
                            seq_data.get('obj_bottom_trans') is not None)
    
    articulated_info = None
    
    if is_articulated and has_bottom_transforms:
        top_key = f'{obj_name}_top'
        bottom_key = f'{obj_name}_bottom'
        
        if top_key not in mesh_map or bottom_key not in mesh_map:
            return None, f"articulated mesh parts not found: {top_key}, {bottom_key}"
        
        _, V0_top = mesh_map[top_key]
        _, V0_bottom = mesh_map[bottom_key]
        mesh_path = mesh_map[obj_name][0]
        n_top = V0_top.shape[0]
        n_bottom = V0_bottom.shape[0]
        
        obj_bottom_rot = np.asarray(seq_data['obj_bottom_rot'])
        obj_bottom_trans = np.asarray(seq_data['obj_bottom_trans']).squeeze(-1)
        obj_bottom_scale = seq_data.get('obj_bottom_scale')
        
        if obj_bottom_scale is not None:
            bottom_scale = np.asarray(obj_bottom_scale, dtype=np.float64)
            if bottom_scale.ndim == 0:
                bottom_scale = np.full(T, float(bottom_scale))
        else:
            bottom_scale = scale.copy()
        
        verts_top = reconstruct_verts(V0_top, obj_rot, obj_trans, scale)
        verts_bottom = reconstruct_verts(V0_bottom, obj_bottom_rot, obj_bottom_trans, bottom_scale)
        verts = np.concatenate([verts_top, verts_bottom], axis=1)
        num_verts = n_top + n_bottom
        
        vertex_part_id = np.concatenate([np.zeros(n_top, dtype=np.int8), 
                                         np.ones(n_bottom, dtype=np.int8)])
        part_rotations = np.stack([obj_rot, obj_bottom_rot], axis=1)
        articulated_info = {
            'vertex_part_id': vertex_part_id,
            'part_rotations': part_rotations.astype(np.float32),
            'num_top_verts': n_top,
            'num_bottom_verts': n_bottom,
        }
    else:
        mesh_path, V0 = mesh_map[obj_name]
        verts = reconstruct_verts(V0, obj_rot, obj_trans, scale)
        num_verts = V0.shape[0]
    
    # Validate centroid BEFORE applying offsets (compare to OMOMO's obj_com_pos)
    centroid_raw = np.mean(verts, axis=1)
    if obj_com is not None:
        err = np.linalg.norm(centroid_raw - obj_com, axis=1)
        max_err = float(np.max(err))
        if max_err > centroid_tol:
            return None, f"centroid error {max_err:.4f}m > {centroid_tol}m"
    else:
        max_err = None
    
    # =========================================================================
    # APPLY COORDINATE OFFSETS (same as robot motion)
    # =========================================================================
    # z offset (height adjustment)
    verts[:, :, 2] += z_offset
    # xy offset (root origin)
    verts[:, :, 0] += xy_offset[0]
    verts[:, :, 1] += xy_offset[1]
    
    # Compute centroid AFTER offsets
    centroid = np.mean(verts, axis=1)
    
    # Compute BPS encoding AFTER offsets (so BPS is in aligned coordinate frame)
    bps = compute_bps(verts, centroid, bps_basis, bps_radius)
    
    result = {
        'object_verts': verts,
        'object_centroid': centroid.astype(np.float32),
        'object_rotation': obj_rot.astype(np.float32),
        'bps_encoding': bps,
        'object_name': obj_name,
        'num_verts': num_verts,
        'mesh_file': mesh_path,
        'is_articulated': is_articulated and has_bottom_transforms,
        'centroid_max_err': max_err,
    }
    
    if articulated_info is not None:
        result.update(articulated_info)
    
    return result, None


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_sequence(
    seq_data: dict,
    smplx_model_path: str,
    mesh_map: dict,
    bps_basis: np.ndarray,
    robot: str,
    device: str,
    bps_radius: float = 1.0,
    centroid_tol: float = 0.05,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Process one OMOMO sequence: retarget to robot + extract object geometry.
    
    Returns:
        (result_dict, None) on success, (None, error_msg) on failure
    """
    seq_name = seq_data.get('seq_name', 'unknown')
    
    # =========================================================================
    # STEP 1: SMPL-X to Robot Retargeting
    # =========================================================================
    try:
        smplx_data, body_model, smplx_output, human_height = load_smplx_from_omomo(
            seq_data, smplx_model_path
        )
    except Exception as e:
        return None, f"SMPL-X loading failed: {e}"
    
    try:
        smplx_frame_data_list, aligned_fps = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=30
        )
    except Exception as e:
        return None, f"SMPL-X processing failed: {e}"
    
    # Retarget
    retargeter = GMR(
        src_human="smplx",
        tgt_robot=robot,
        actual_human_height=human_height,
    )
    
    qpos_list = []
    for smplx_frame_data in smplx_frame_data_list:
        qpos = retargeter.retarget(smplx_frame_data)
        qpos_list.append(qpos.copy())
    qpos_list = np.array(qpos_list)
    
    # Kinematics
    kinematics_model = KinematicsModel(retargeter.xml_file, device=device)
    
    root_pos = qpos_list[:, :3]
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    dof_pos = qpos_list[:, 7:]
    num_frames = root_pos.shape[0]
    
    # Forward kinematics for local body positions
    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0
    
    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos, fk_root_rot, 
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )
    body_names = kinematics_model.body_names
    
    # =========================================================================
    # STEP 2: Compute Height and XY Offsets
    # =========================================================================
    z_offset = 0.0
    xy_offset = np.zeros(2, dtype=np.float32)
    
    # Height adjustment
    body_pos, _ = kinematics_model.forward_kinematics(
        torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
        torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )
    lowest_height = torch.min(body_pos[..., 2]).item()
    z_offset = -lowest_height
    root_pos[:, 2] += z_offset
    
    # XY offset to start at origin
    xy_offset = -root_pos[0, :2].copy()
    root_pos[:, :2] += xy_offset
    
    # =========================================================================
    # STEP 2.5: Extract Global Hand Positions (OMOMO Stage 1 target)
    # =========================================================================
    # Recompute FK with adjusted root to get global hand positions
    global_body_pos, _ = kinematics_model.forward_kinematics(
        torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
        torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )
    global_body_pos_np = global_body_pos.detach().cpu().numpy()
    
    # Find hand link indices (left_rubber_hand=29, right_rubber_hand=37 for G1)
    left_hand_idx = body_names.index('left_rubber_hand') if 'left_rubber_hand' in body_names else None
    right_hand_idx = body_names.index('right_rubber_hand') if 'right_rubber_hand' in body_names else None
    
    # Extract hand positions (T, 6) = [left_x, left_y, left_z, right_x, right_y, right_z]
    if left_hand_idx is not None and right_hand_idx is not None:
        left_hand_pos = global_body_pos_np[:, left_hand_idx, :]   # (T, 3)
        right_hand_pos = global_body_pos_np[:, right_hand_idx, :] # (T, 3)
        hand_positions = np.concatenate([left_hand_pos, right_hand_pos], axis=-1)  # (T, 6)
    else:
        # Fallback: use wrist links
        left_wrist_idx = body_names.index('left_wrist_yaw_link') if 'left_wrist_yaw_link' in body_names else 28
        right_wrist_idx = body_names.index('right_wrist_yaw_link') if 'right_wrist_yaw_link' in body_names else 36
        left_hand_pos = global_body_pos_np[:, left_wrist_idx, :]
        right_hand_pos = global_body_pos_np[:, right_wrist_idx, :]
        hand_positions = np.concatenate([left_hand_pos, right_hand_pos], axis=-1)
    
    # =========================================================================
    # STEP 3: Extract Object Geometry (with same offsets)
    # =========================================================================
    obj_result, obj_err = extract_object_geometry(
        seq_data, mesh_map, bps_basis, z_offset, xy_offset,
        bps_radius=bps_radius, centroid_tol=centroid_tol
    )
    
    if obj_result is None:
        return None, f"object extraction failed: {obj_err}"
    
    # =========================================================================
    # STEP 4: Build unified output
    # =========================================================================
    motion_data = {
        # Robot motion
        "fps": aligned_fps,
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "dof_pos": dof_pos.astype(np.float32),
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": body_names,
        "seq_name": seq_name,
        
        # Object geometry
        "object_verts": obj_result['object_verts'],
        "object_centroid": obj_result['object_centroid'],
        "object_rotation": obj_result['object_rotation'],
        "bps_encoding": obj_result['bps_encoding'],
        "object_name": obj_result['object_name'],
        "num_verts": obj_result['num_verts'],
        "is_articulated": obj_result['is_articulated'],
        
        # Hand positions for OMOMO Stage 1 (T, 6)
        "hand_positions": hand_positions.astype(np.float32),
        
        # Visualization-compatible object fields (aliases/conversions)
        # object_pos: alias for object_centroid
        "object_pos": obj_result['object_centroid'],
        # object_rot: (T, 4) quaternion xyzw from (T, 3, 3) rotation matrix
        "object_rot": mat_to_quat_xyzw(
            torch.from_numpy(obj_result['object_rotation']).float()
        ).numpy().astype(np.float32),
        
        # Offsets (for reference/debugging)
        "z_offset": z_offset,
        "xy_offset": xy_offset,
    }
    
    # Add articulated fields if present
    if obj_result['is_articulated']:
        motion_data['vertex_part_id'] = obj_result['vertex_part_id']
        motion_data['part_rotations'] = obj_result['part_rotations']
        motion_data['num_top_verts'] = obj_result['num_top_verts']
        motion_data['num_bottom_verts'] = obj_result['num_bottom_verts']
    
    return motion_data, None


def main():
    parser = argparse.ArgumentParser(
        description="Convert OMOMO to robot motion dataset with object geometry"
    )
    parser.add_argument("--objects_dir", required=True,
                        help="Directory with captured_objects/*.obj meshes")
    parser.add_argument("--smplx_model_dir", required=True,
                        help="Path to SMPL-X body models")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for .pkl files")
    parser.add_argument("--robot", default="unitree_g1",
                        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", 
                                "unitree_h1_2", "booster_t1", "fourier_n1"],
                        help="Target robot")
    parser.add_argument("--bps_radius", type=float, default=1.0)
    parser.add_argument("--centroid_tol", type=float, default=0.05,
                        help="Max centroid validation error (m)")
    parser.add_argument("--num_motions", type=int, help="Limit sequences (for testing)")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip sequences with existing output files")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load meshes
    print(f"Loading meshes from {args.objects_dir}")
    mesh_map = build_mesh_map(args.objects_dir)
    print(f"  Loaded {len(mesh_map)} objects: {list(mesh_map.keys())}")
    
    # Generate and save BPS basis
    bps_basis = generate_bps_points(1024)
    bps_path = os.path.join(args.output_dir, "bps_basis_points.npy")
    np.save(bps_path, bps_basis)
    print(f"  Saved BPS basis to {bps_path}")
    
    # Process files
    total_success, total_failed = 0, 0
    index_entries = []
    failures = []

    omomo_files = [
        "/home/learning/Documents/g1-gmr/OMOMO_DATA/OMOMO_p_files/train_diffusion_manip_seq_joints24.p",
        "/home/learning/Documents/g1-gmr/OMOMO_DATA/OMOMO_p_files/test_diffusion_manip_seq_joints24.p" 
    ]
    
    for omomo_file in omomo_files:
        print(f"\nProcessing {omomo_file}")
        data = joblib.load(omomo_file)
        
        seqs = list(data.items())
        if args.num_motions:
            seqs = seqs[:args.num_motions]
        
        for seq_key, seq_data in tqdm(seqs, desc=pathlib.Path(omomo_file).stem):
            seq_name = seq_data.get('seq_name', str(seq_key))
            out_path = os.path.join(args.output_dir, f"{seq_name}.pkl")
            
            if args.skip_existing and os.path.exists(out_path):
                if args.verbose:
                    print(f"  SKIP {seq_name} (exists)")
                continue
            
            result, err = process_sequence(
                seq_data,
                args.smplx_model_dir,
                mesh_map,
                bps_basis,
                args.robot,
                args.device,
                bps_radius=args.bps_radius,
                centroid_tol=args.centroid_tol,
            )
            
            if result is None:
                total_failed += 1
                failures.append(f"{seq_name}: {err}")
                if args.verbose:
                    print(f"  FAIL {seq_name}: {err}")
                continue
            
            # Save
            with open(out_path, "wb") as f:
                pickle.dump(result, f)
            
            total_success += 1
            index_entries.append({
                'seq_name': seq_name,
                'file': f"{seq_name}.pkl",
                'T': int(result['root_pos'].shape[0]),
                'K': int(result['num_verts']),
                'object': result['object_name'],
                'is_articulated': result['is_articulated'],
            })
            
            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Success: {total_success}, Failed: {total_failed}")
    
    # Save index
    index_path = os.path.join(args.output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index_entries, f, indent=2)
    print(f"Saved index to {index_path}")
    
    # Save failures
    if failures:
        fail_path = os.path.join(args.output_dir, "failures.txt")
        with open(fail_path, "w") as f:
            f.write("\n".join(failures))
        print(f"Saved {len(failures)} failures to {fail_path}")


if __name__ == "__main__":
    main()
