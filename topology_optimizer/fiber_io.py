"""
fiber_io.py
=============
Exports segmented 3D fiber centerlines into GeoDict .gad format (JSON-based).
Fully compatible with GeoDict, the Altendorf-Jeulin Model, and evaluate_against_gt.py.
"""

import os
import json
import time
import platform
from datetime import datetime
import numpy as np
from scipy.spatial import cKDTree

def save_fiber_length_histogram(
    pred_lengths: list[int],
    gt_lengths: list[int] | None,
    out_path: str,
):
    """Bar histogram of fiber lengths (predicted vs GT if available)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(np.log10(max(1, min(pred_lengths))),
                       np.log10(max(pred_lengths) + 1), 40)

    ax.hist(pred_lengths, bins=bins, color='steelblue', alpha=0.75, label='Predicted')
    if gt_lengths:
        ax.hist(gt_lengths, bins=bins, color='tomato', alpha=0.60, label='Ground Truth')

    ax.set_xscale('log')
    ax.set_xlabel('Fiber length (voxels)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Fiber Length Distribution  (N_pred={len(pred_lengths)}'
                 + (f', N_gt={len(gt_lengths)}' if gt_lengths else '') + ')')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [viz] Saved fiber length histogram to {out_path}", flush=True)



def export_uint16_tiff(arr: np.ndarray, out_path: str, verbose: bool = True):
    """
    Save 3D array as uint16 TIFF with compression (supports memory-mapped and large volumes).
    Uses chunked / slice-by-slice writing for volumes > 1GB to ensure low peak RAM usage.
    """
    import tifffile
    import time
    t0 = time.time()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    D = arr.shape[0]
    total_bytes = arr.size * 2
    is_big = total_bytes > (2 * 1024 * 1024 * 1024)

    axes = 'ZYX' if arr.ndim == 3 else ('YX' if arr.ndim == 2 else None)
    meta = {'axes': axes} if axes else None

    if total_bytes < 1024 * 1024 * 1024 and not isinstance(arr, np.memmap):
        # Fast in-memory write for arrays < 1 GB (and not memory-mapped)
        u16_arr = np.asarray(arr, dtype=np.uint16)
        tifffile.imwrite(
            out_path,
            u16_arr,
            compression='zlib',
            metadata=meta,
            bigtiff=is_big,
        )
    elif arr.ndim >= 3:
        # Stream slice-by-slice for huge or memory-mapped volumes to keep resident RAM near zero
        def _slice_gen():
            for z in range(D):
                yield np.asarray(arr[z], dtype=np.uint16)

        tifffile.imwrite(
            out_path,
            _slice_gen(),
            shape=arr.shape,
            dtype=np.uint16,
            compression='zlib',
            metadata=meta,
            bigtiff=True,
        )
    else:
        u16_arr = np.asarray(arr, dtype=np.uint16)
        tifffile.imwrite(
            out_path,
            u16_arr,
            compression='zlib',
            metadata=meta,
            bigtiff=True,
        )

    if verbose:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  [export] Saved uint16 TIFF: {out_path} ({size_mb:.1f} MB in {time.time()-t0:.1f}s)", flush=True)


def load_gt_centerlines_from_gad(
    gad_path: str,
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Load ground-truth centerline instance labels from a GAD file.

    Returns
    -------
    gt_centerline_vol : (D, H, W) int32 -- each fiber has a unique integer label
    n_gt_fibers       : int
    """
    with open(gad_path, 'r', encoding='utf-8') as f:
        gad = json.load(f)

    D, H, W = volume_shape
    voxel_length = gad['Domain']['VoxelLength'][0]
    domain_length = np.array([
        gad['Domain']['LengthZ'][0] / voxel_length,
        gad['Domain']['LengthY'][0] / voxel_length,
        gad['Domain']['LengthX'][0] / voxel_length,
    ], dtype=np.float32)

    num_objects = gad['NumberOfObjects']
    gt_vol = np.zeros((D, H, W), dtype=np.int32)

    for obj_idx in range(1, num_objects + 1):
        obj_key = f'Object{obj_idx}'
        obj = gad.get(obj_key, {})
        p_keys = sorted([k for k in obj.keys() if k.startswith('Point')],
                        key=lambda x: int(x[5:]))
        pts_list = []
        for pk in p_keys:
            p_val = obj[pk]
            if isinstance(p_val, dict) and 'Coord' in p_val:
                pts_list.append(p_val['Coord'][0])
            elif isinstance(p_val, (list, tuple)):
                pts_list.append(p_val[0])

        if len(pts_list) < 2:
            continue

        pts = np.array(pts_list, dtype=np.float32) / voxel_length
        pts_zyx = pts[:, [2, 1, 0]] % domain_length

        # Dense interpolation
        n_pts = len(pts_zyx)
        t_dense = np.linspace(0, 1, n_pts * 12)
        t_sparse = np.linspace(0, 1, n_pts)
        dense_z = np.interp(t_dense, t_sparse, pts_zyx[:, 0])
        dense_y = np.interp(t_dense, t_sparse, pts_zyx[:, 1])
        dense_x = np.interp(t_dense, t_sparse, pts_zyx[:, 2])

        iz = np.clip(np.round(dense_z).astype(int), 0, D - 1)
        iy = np.clip(np.round(dense_y).astype(int), 0, H - 1)
        ix = np.clip(np.round(dense_x).astype(int), 0, W - 1)
        gt_vol[iz, iy, ix] = obj_idx

    return gt_vol, num_objects

def order_centerline_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 3D skeleton points along the continuous curve from one tip to the other.
    
    Parameters
    ----------
    pts : (N, 3) float or int [z, y, x]

    Returns
    -------
    ordered_pts : (N, 3) float64 [z, y, x]
    """
    n = len(pts)
    if n <= 2:
        return pts.astype(np.float64)

    pts = pts.astype(np.float64)
    tree = cKDTree(pts)
    centroid = pts.mean(axis=0)
    start_idx = int(np.argmax(np.linalg.norm(pts - centroid, axis=1)))

    unvisited = set(range(n))
    unvisited.remove(start_idx)
    ordered_indices = [start_idx]
    curr = start_idx

    while unvisited:
        k_query = min(16, len(pts))
        dists, neighbors = tree.query(pts[curr], k=k_query)
        found = False
        for d, nbr in zip(dists, neighbors):
            if nbr in unvisited:
                if d > 5.0:  # voxel gap tolerance in skeleton
                    break
                curr = nbr
                unvisited.remove(curr)
                ordered_indices.append(curr)
                found = True
                break
        if not found:
            sub = list(unvisited)
            dists_sub = np.linalg.norm(pts[sub] - pts[curr], axis=1)
            nearest_sub = np.argmin(dists_sub)
            if dists_sub[nearest_sub] > 8.0:
                break
            curr = sub[nearest_sub]
            unvisited.remove(curr)
            ordered_indices.append(curr)

    return pts[ordered_indices]


def export_fibers_as_gad(
    inst_skel: np.ndarray,
    inst_vol: np.ndarray | None = None,
    out_path: str = 'outputs/instance_fibers.gad',
    voxel_size: float = 1e-06,
    is_periodic: bool = False,
    min_points: int = 2,
    default_diameter_vx: float = 7.0,
    verbose: bool = True
) -> str:
    """
    Export resolved 3D fibers from labeled skeleton volume into GeoDict .gad format.

    Parameters
    ----------
    inst_skel           : (D, H, W) int32 labeled skeleton volume (0 = background)
    inst_vol            : optional (D, H, W) int32 instance volume (used to estimate fiber diameter)
    out_path            : path to output .gad file
    voxel_size          : physical voxel size in meters (default: 1e-06 m = 1 um)
    is_periodic         : whether domain is periodic (default: False)
    min_points          : minimum points along centerline to export as a fiber (default: 2)
    default_diameter_vx : fallback fiber diameter in voxels (default: 7.0)
    verbose             : bool

    Returns
    -------
    out_path : str
    """
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    D, H, W = inst_skel.shape

    unique_fids = np.unique(inst_skel[inst_skel > 0])
    now = datetime.now()

    system_dict = {
        "Header": {
            "Release": now.strftime("%Y"),
            "Revision": 0,
            "BuildDate": now.strftime("%d %b %Y"),
            "CreationDate": now.strftime("%d %b %Y"),
            "CreationTime": now.strftime("%H:%M:%S"),
            "Creator": "FiberResolutionPipeline",
            "Platform": platform.system()
        },
        "Description": "Segmented fibers resolved by Dual Neural Field & Topology Optimization Pipeline",
        "Domain": {
            "PeriodicX": is_periodic,
            "PeriodicY": is_periodic,
            "PeriodicZ": is_periodic,
            "DomainMode": "Length",
            "OriginX": [0.0, "m"],
            "OriginY": [0.0, "m"],
            "OriginZ": [0.0, "m"],
            "VoxelLength": [voxel_size, "m"],
            "LengthX": [float(W * voxel_size), "m"],
            "LengthY": [float(H * voxel_size), "m"],
            "LengthZ": [float(D * voxel_size), "m"],
            "OverlapMode": "NewMaterial",
            "HollowMaterialID": 0,
            "PostProcessing": {
                "ResolveOverlap": True,
                "MarkContactVoxels": False,
                "ContactMaterialID": 3
            }
        }
    }

    object_count = 0
    for fid in unique_fids:
        pts_zyx = np.argwhere(inst_skel == fid)
        if len(pts_zyx) < min_points:
            continue

        ordered_zyx = order_centerline_points(pts_zyx)
        if len(ordered_zyx) < min_points:
            continue

        # Estimate diameter in meters
        if inst_vol is not None:
            vol_count = int(np.sum(inst_vol == fid))
            cross_sec_area = vol_count / max(len(ordered_zyx), 1)
            radius_vx = np.sqrt(max(cross_sec_area, 0.5) / np.pi)
            diameter_m = float(2.0 * radius_vx * voxel_size)
        else:
            diameter_m = float(default_diameter_vx * voxel_size)

        object_count += 1
        obj_key = f"Object{object_count}"

        # GeoDict coordinates are in meters (X, Y, Z)
        fiber_dict = {
            "MaterialIDModel": "Constant",
            "MaterialID": 1,
            "Type": "CurvedCircularFiber",
            "NumberOfSegments": len(ordered_zyx) - 1,
            "FiberEndType1": "Rounded",
            "FiberEndType2": "Rounded",
            "CuttingPlane1": [0.0, 0.0, 0.0],
            "CuttingPlane2": [0.0, 0.0, 0.0],
            "Diameter": diameter_m
        }

        for j, pt_zyx in enumerate(ordered_zyx, start=1):
            z_vx, y_vx, x_vx = pt_zyx
            x_m = float(x_vx * voxel_size)
            y_m = float(y_vx * voxel_size)
            z_m = float(z_vx * voxel_size)
            fiber_dict[f"Point{j}"] = [[x_m, y_m, z_m], "m"]

        system_dict[obj_key] = fiber_dict

    system_dict["NumberOfObjects"] = object_count

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(system_dict, f, indent=2)

    if verbose:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  [export] Saved {object_count} fibers to GAD file: {out_path} ({size_mb:.2f} MB in {time.time()-t0:.2f}s)", flush=True)

    return out_path

