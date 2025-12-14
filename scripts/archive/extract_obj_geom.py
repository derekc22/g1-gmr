#!/usr/bin/env python3
"""
Extract OMOMO object geometry from .p files.

Reconstructs per-frame mesh vertices V ∈ R^{T×K×3} and computes BPS encoding
as defined in OMOMO paper Section 3.1.

OMOMO .p file format (fixed, no guessing needed):
    obj_rot: (T, 3, 3) - rotation matrices
    obj_trans: (T, 3, 1) - translation
    obj_scale: (T,) - per-frame scale
    obj_com_pos: (T, 3) - center of mass (for validation)
    seq_name: str - e.g. "sub10_clothesstand_000"

Output per sequence (.npz):
    object_verts: (T, K, 3) - reconstructed mesh vertices
    object_centroid: (T, 3) - centroid g_t = mean(V_t)
    object_rotation: (T, 3, 3) - rotation R_t (for contact constraints)
    bps_encoding: (T, 1024, 3) - BPS d(B_t, V_t)

Usage:
    python extract_obj_geom.py \\
        --inputs OMOMO_DATA/OMOMO_p_files/train_diffusion_manip_seq_joints24.p \\
        --objects_dir /path/to/captured_objects \\
        --out_dir output/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import joblib
except ImportError:
    sys.exit("Error: pip install joblib")

try:
    from scipy.spatial import cKDTree
except ImportError:
    sys.exit("Error: pip install scipy")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x


# =============================================================================
# MESH LOADING
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
    """
    Load object meshes. Prefer *_cleaned_simplified.obj files.
    Returns {object_name: (path, vertices)}.
    
    For articulated objects (mop, vacuum), also loads _top and _bottom parts.
    """
    if not os.path.isdir(objects_dir):
        sys.exit(f"Error: {objects_dir} not found")
    
    mesh_map = {}
    for fname in sorted(os.listdir(objects_dir)):
        if not fname.endswith('.obj'):
            continue
        # Extract object name: "clothesstand_cleaned_simplified.obj" -> "clothesstand"
        name = fname.replace('_cleaned_simplified', '').replace('_cleaned', '').replace('.obj', '')
        # Prefer cleaned_simplified over others
        if name in mesh_map and 'cleaned_simplified' not in fname:
            continue
        path = os.path.join(objects_dir, fname)
        mesh_map[name] = (path, load_obj_mesh(path))
    
    return mesh_map


# Articulated objects that have separate top/bottom parts with independent transforms
ARTICULATED_OBJECTS = {'mop', 'vacuum'}


# =============================================================================
# BPS ENCODING (OMOMO Section 3.1)
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
    """
    Compute BPS encoding d(B_t, V_t) for each frame.
    
    Args:
        verts: (T, K, 3) mesh vertices
        centroid: (T, 3) per-frame centroid
        basis: (1024, 3) fixed basis points in unit ball
        radius: ball radius (OMOMO uses 1.0)
    
    Returns:
        (T, 1024, 3) difference vectors
    """
    T = verts.shape[0]
    bps = np.zeros((T, len(basis), 3), dtype=np.float32)
    for t in range(T):
        B_t = basis * radius + centroid[t]  # translate to centroid
        tree = cKDTree(verts[t])
        _, idx = tree.query(B_t)
        bps[t] = B_t - verts[t, idx]
    return bps


# =============================================================================
# VERTEX RECONSTRUCTION
# =============================================================================

def reconstruct_verts(V0: np.ndarray, R: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Reconstruct per-frame vertices: V_t = R_t @ (s_t * V0) + t_t
    
    Args:
        V0: (K, 3) canonical mesh
        R: (T, 3, 3) rotation matrices
        t: (T, 3) translation
        s: (T,) per-frame scale
    """
    V0_64 = V0.astype(np.float64)
    # Apply per-frame scale: (T, K, 3) = (T, 1, 1) * (K, 3)
    V0_scaled = s[:, None, None] * V0_64[None, :, :]  # (T, K, 3)
    # Rotate: einsum 'tij,tkj->tki' but V0_scaled is already (T,K,3)
    rotated = np.einsum('tij,tkj->tki', R, V0_scaled)
    return (rotated + t[:, None, :]).astype(np.float32)


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def get_object_name(seq_name: str) -> str:
    """Extract object name from OMOMO seq_name: 'sub10_clothesstand_000' -> 'clothesstand'"""
    parts = seq_name.split('_')
    # Skip 'subN' prefix and numeric suffix
    obj_parts = [p for i, p in enumerate(parts) 
                 if not (i == 0 and p.startswith('sub')) and not (i == len(parts)-1 and p.isdigit())]
    return '_'.join(obj_parts) if obj_parts else seq_name


def extract_sequence(seq_data: dict, mesh_map: dict, bps_basis: np.ndarray,
                     bps_radius: float = 1.0, tol: float = 0.01) -> Tuple[Optional[dict], Optional[str]]:
    """Extract geometry from one sequence."""
    
    # Get required fields (OMOMO format is fixed)
    seq_name = seq_data.get('seq_name', 'unknown')
    obj_rot = seq_data.get('obj_rot')  # (T, 3, 3)
    obj_trans = seq_data.get('obj_trans')  # (T, 3, 1)
    obj_scale = seq_data.get('obj_scale')  # (T,)
    obj_com = seq_data.get('obj_com_pos')  # (T, 3)
    
    if obj_rot is None or obj_trans is None:
        return None, "missing obj_rot or obj_trans"
    
    # Reshape translation: (T, 3, 1) -> (T, 3)
    obj_trans = np.asarray(obj_trans).squeeze(-1)
    obj_rot = np.asarray(obj_rot)
    T = obj_rot.shape[0]
    
    # Get per-frame scale
    if obj_scale is not None:
        scale = np.asarray(obj_scale, dtype=np.float64)
        if scale.ndim == 0:
            scale = np.full(T, float(scale))
    else:
        scale = np.ones(T, dtype=np.float64)
    
    # Find mesh
    obj_name = get_object_name(seq_name)
    if obj_name not in mesh_map:
        # Try lowercase
        obj_name_lower = obj_name.lower()
        if obj_name_lower in mesh_map:
            obj_name = obj_name_lower
        else:
            return None, f"mesh not found: {obj_name}"
    
    # Check if articulated object (mop, vacuum have top + bottom parts)
    is_articulated = obj_name in ARTICULATED_OBJECTS
    has_bottom_transforms = (seq_data.get('obj_bottom_rot') is not None and 
                            seq_data.get('obj_bottom_trans') is not None)
    
    if is_articulated and has_bottom_transforms:
        # Load top and bottom meshes
        top_key = f'{obj_name}_top'
        bottom_key = f'{obj_name}_bottom'
        
        if top_key not in mesh_map or bottom_key not in mesh_map:
            return None, f"articulated mesh parts not found: {top_key}, {bottom_key}"
        
        _, V0_top = mesh_map[top_key]
        _, V0_bottom = mesh_map[bottom_key]
        mesh_path = mesh_map[obj_name][0]  # Report the main mesh path
        n_top = V0_top.shape[0]
        n_bottom = V0_bottom.shape[0]
        
        # Get bottom transforms
        obj_bottom_rot = np.asarray(seq_data['obj_bottom_rot'])
        obj_bottom_trans = np.asarray(seq_data['obj_bottom_trans']).squeeze(-1)
        obj_bottom_scale = seq_data.get('obj_bottom_scale')
        
        if obj_bottom_scale is not None:
            bottom_scale = np.asarray(obj_bottom_scale, dtype=np.float64)
            if bottom_scale.ndim == 0:
                bottom_scale = np.full(T, float(bottom_scale))
        else:
            bottom_scale = scale.copy()
        
        # Reconstruct both parts
        verts_top = reconstruct_verts(V0_top, obj_rot, obj_trans, scale)
        verts_bottom = reconstruct_verts(V0_bottom, obj_bottom_rot, obj_bottom_trans, bottom_scale)
        
        # Merge: concatenate vertices
        verts = np.concatenate([verts_top, verts_bottom], axis=1)  # (T, K_top + K_bottom, 3)
        num_verts = n_top + n_bottom
        
        # For articulated objects, store per-vertex rotation index and both rotations
        # vertex_part_id[i] = 0 for top vertices, 1 for bottom vertices
        vertex_part_id = np.concatenate([np.zeros(n_top, dtype=np.int8), 
                                         np.ones(n_bottom, dtype=np.int8)])
        part_rotations = np.stack([obj_rot, obj_bottom_rot], axis=1)  # (T, 2, 3, 3)
        articulated_info = {
            'vertex_part_id': vertex_part_id,      # (K,) - which part each vertex belongs to
            'part_rotations': part_rotations.astype(np.float32),  # (T, 2, 3, 3) - rotations for [top, bottom]
            'num_top_verts': n_top,
            'num_bottom_verts': n_bottom,
        }
    else:
        # Standard rigid object
        mesh_path, V0 = mesh_map[obj_name]
        verts = reconstruct_verts(V0, obj_rot, obj_trans, scale)
        num_verts = V0.shape[0]
        articulated_info = None
    
    centroid = np.mean(verts, axis=1)
    
    # Validate against obj_com_pos
    if obj_com is not None:
        err = np.linalg.norm(centroid - obj_com, axis=1)
        max_err = float(np.max(err))
        if max_err > tol:
            return None, f"centroid error {max_err:.4f}m > {tol}m"
    else:
        max_err = None
    
    # Compute BPS encoding
    bps = compute_bps(verts, centroid, bps_basis, bps_radius)
    
    result = {
        'object_verts': verts,
        'object_centroid': centroid.astype(np.float32),
        'object_rotation': obj_rot.astype(np.float32),  # Primary rotation (top part for articulated)
        'bps_encoding': bps,
        'num_frames': T,
        'num_verts': num_verts,
        'object_name': obj_name,
        'mesh_file': mesh_path,
        'scale': scale,
        'centroid_max_err': max_err,
        'is_articulated': is_articulated and has_bottom_transforms,
    }
    
    # Add articulated object info for contact constraint postprocessing
    if articulated_info is not None:
        result.update(articulated_info)
    
    return result, None


def main():
    parser = argparse.ArgumentParser(description="Extract OMOMO object geometry with BPS")
    parser.add_argument("--inputs", nargs="+", required=True, help=".p files")
    parser.add_argument("--objects_dir", required=True, help="Directory with .obj meshes")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--bps_radius", type=float, default=1.0)
    parser.add_argument("--centroid_tol", type=float, default=0.01, help="Max centroid error (m)")
    parser.add_argument("--max_seqs", type=int, help="Limit sequences (for testing)")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    # Load meshes
    print(f"Loading meshes from {args.objects_dir}")
    mesh_map = build_mesh_map(args.objects_dir)
    print(f"  Loaded {len(mesh_map)} objects: {list(mesh_map.keys())}")
    
    # Generate BPS basis points
    bps_basis = generate_bps_points(1024)
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        np.save(os.path.join(args.out_dir, "bps_basis_points.npy"), bps_basis)
    
    # Process files
    total_extracted, total_failed = 0, 0
    index_entries, failures = [], []
    
    for input_path in args.inputs:
        print(f"\nProcessing {input_path}")
        data = joblib.load(input_path)
        
        seqs = list(data.items())
        if args.max_seqs:
            seqs = seqs[:args.max_seqs]
        
        for seq_key, seq_data in tqdm(seqs, desc=Path(input_path).stem):
            seq_name = seq_data.get('seq_name', str(seq_key))
            
            result, err = extract_sequence(
                seq_data, mesh_map, bps_basis, args.bps_radius, args.centroid_tol
            )
            
            if result is None:
                total_failed += 1
                failures.append(f"{seq_name}: {err}")
                if args.verbose:
                    print(f"  FAIL {seq_name}: {err}")
                continue
            
            total_extracted += 1
            out_path = os.path.join(args.out_dir, f"{seq_name}.npz")
            
            if not args.dry_run:
                # Build save dict with all fields
                save_dict = {
                    'object_verts': result['object_verts'],
                    'object_centroid': result['object_centroid'],
                    'object_rotation': result['object_rotation'],
                    'bps_encoding': result['bps_encoding'],
                    'num_frames': result['num_frames'],
                    'num_verts': result['num_verts'],
                    'object_name': result['object_name'],
                    'mesh_file': result['mesh_file'],
                    'scale': result['scale'],
                    'is_articulated': result['is_articulated'],
                    'centroid_max_err': result['centroid_max_err'] if result['centroid_max_err'] is not None else -1.0,
                }
                
                # Add articulated object fields if present
                if result['is_articulated']:
                    save_dict['vertex_part_id'] = result['vertex_part_id']
                    save_dict['part_rotations'] = result['part_rotations']
                    save_dict['num_top_verts'] = result['num_top_verts']
                    save_dict['num_bottom_verts'] = result['num_bottom_verts']
                
                np.savez_compressed(out_path, **save_dict)
            
            index_entries.append({
                'seq_name': seq_name,
                'file': f"{seq_name}.npz",
                'T': int(result['num_frames']),
                'K': int(result['num_verts']),
                'object': result['object_name'],
                'is_articulated': result['is_articulated'],
            })
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Extracted: {total_extracted}, Failed: {total_failed}")
    
    if not args.dry_run:
        with open(os.path.join(args.out_dir, "index.json"), "w") as f:
            json.dump(index_entries, f, indent=2)
        if failures:
            with open(os.path.join(args.out_dir, "failures.txt"), "w") as f:
                f.write("\n".join(failures))
    
    if args.dry_run:
        print("[DRY RUN] No files written")


if __name__ == "__main__":
    main()
