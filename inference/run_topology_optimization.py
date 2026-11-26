import os
import sys
import time
import argparse
import gc
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize
import tifffile

# Ensure package root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from topology_optimizer.skeleton_graph import sever_h_junctions, build_segment_graph
from topology_optimizer.link_fibers import find_bridge_candidates, optimize_topology
from topology_optimizer.label_fibers import build_labeled_skeleton, label_fibers
from topology_optimizer.fiber_io import save_fiber_length_histogram, export_uint16_tiff, export_fibers_as_gad

# TODO make sure results don't differ from expectation when input is wrong
# TODO keep records of runs
# TODO make data loading consistent and ensure that orientation data is appropriate


def get_default_volume(target_dir='data/fibers_to_segment'):
    """Finds and returns the first .tif, .tiff, or .npy volume file in the specified directory."""
    if os.path.isdir(target_dir):
        valid_exts = ('.tif', '.tiff', '.npy')
        files = [
            os.path.join(target_dir, f).replace('\\', '/')
            for f in sorted(os.listdir(target_dir))
            if f.lower().endswith(valid_exts)
            and not f.lower().endswith(('_intensity.npy', '_orientation.npy', '_skel.npy', '_skeleton.npy', '_instance.npy'))
            and os.path.isfile(os.path.join(target_dir, f))
        ]
        if files:
            return files[0]
    return 'data/fibers_to_segment/COLLAGENCROP_003_0000.tif'


def get_default_intensity(target_dir='outputs', volume_path=None):
    """Finds default intensity file from outputs folder first, falling back to data/fibers_to_segment."""
    stem = os.path.splitext(os.path.basename(volume_path))[0] if volume_path else ''

    # 1. Search in outputs folder
    if os.path.isdir('outputs'):
        if stem:
            for ext in ('.npy', '.tif', '.tiff'):
                p = os.path.join('outputs', f"{stem}_intensity{ext}").replace('\\', '/')
                if os.path.exists(p):
                    return p
        for name in ['fiber_intensity.npy']:
            p = os.path.join('outputs', name).replace('\\', '/')
            if os.path.exists(p):
                return p
        for f in sorted(os.listdir('outputs')):
            if 'intensity' in f.lower() and f.lower().endswith(('.npy', '.tif', '.tiff')):
                return os.path.join('outputs', f).replace('\\', '/')

    # 2. Search in target_dir / data/fibers_to_segment folder
    if os.path.isdir(target_dir):
        if stem:
            for ext in ('.npy', '.tif', '.tiff'):
                p = os.path.join(target_dir, f"{stem}_intensity{ext}").replace('\\', '/')
                if os.path.exists(p):
                    return p
        for name in ['fiber_intensity.npy', 'intensity.npy']:
            p = os.path.join(target_dir, name).replace('\\', '/')
            if os.path.exists(p):
                return p
        for f in sorted(os.listdir(target_dir)):
            if 'intensity' in f.lower() and f.lower().endswith(('.npy', '.tif', '.tiff')):
                return os.path.join(target_dir, f).replace('\\', '/')

    return 'outputs/fiber_intensity.npy'


def get_default_orientation(target_dir='outputs', volume_path=None):
    """Finds default orientation file from outputs folder first, falling back to data/fibers_to_segment."""
    stem = os.path.splitext(os.path.basename(volume_path))[0] if volume_path else ''

    # 1. Search in outputs folder
    if os.path.isdir('outputs'):
        if stem:
            for ext in ('.npy', '.tif', '.tiff'):
                p = os.path.join('outputs', f"{stem}_orientation{ext}").replace('\\', '/')
                if os.path.exists(p):
                    return p
        for name in ['fiber_orientation.npy', 'dual_collagen_64_orientation.npy', 'dual_collagen_orientation.npy', 'dual_collagen_morpho_orientation.npy']:
            p = os.path.join('outputs', name).replace('\\', '/')
            if os.path.exists(p):
                return p
        for f in sorted(os.listdir('outputs')):
            if 'orientation' in f.lower() and f.lower().endswith(('.npy', '.tif', '.tiff')):
                return os.path.join('outputs', f).replace('\\', '/')

    # 2. Search in target_dir / data/fibers_to_segment folder
    if os.path.isdir(target_dir):
        if stem:
            for ext in ('.npy', '.tif', '.tiff'):
                p = os.path.join(target_dir, f"{stem}_orientation{ext}").replace('\\', '/')
                if os.path.exists(p):
                    return p
        for name in ['fiber_orientation.npy', 'orientation.npy']:
            p = os.path.join(target_dir, name).replace('\\', '/')
            if os.path.exists(p):
                return p
        for f in sorted(os.listdir(target_dir)):
            if 'orientation' in f.lower() and f.lower().endswith(('.npy', '.tif', '.tiff')):
                return os.path.join(target_dir, f).replace('\\', '/')

    return 'outputs/fiber_orientation.npy'


def load_volume_array(path: str, mmap_mode: str = 'r'):
    """Loads a 3D volume from .npy, .tif, or .tiff format."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Volume file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path, mmap_mode=mmap_mode)
    elif ext in ('.tif', '.tiff'):
        vol = tifffile.imread(path)
        if vol.dtype == bool:
            return vol.astype(np.float32)
        elif vol.max() > 1.0:
            return (vol > 127).astype(np.float32)
        else:
            return (vol > 0.5).astype(np.float32)
    else:
        raise ValueError(f"Unsupported volume file format '{ext}' for '{path}'. Supported formats: .npy, .tif, .tiff")


def load_neural_field_array(path: str, mmap_mode: str = 'r'):
    """Loads a neural field array (intensity or orientation) from .npy, .tif, or .tiff format."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Neural field file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path, mmap_mode=mmap_mode)
    elif ext in ('.tif', '.tiff'):
        arr = tifffile.imread(path)
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) / 65535.0
        elif arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        return arr.astype(np.float32)
    else:
        return np.load(path, mmap_mode=mmap_mode)


DEFAULT_PARAMS = dict(
    # Fiber??? TODO
    radius = 2,
    # Skeleton graph
    core_threshold    = 0.28, #TODO
    sigma_prefilter   = 0.60, #TODO scale by fiber diameter

    max_rung_length      = 14, #TODO scale by fiber diameter
    min_trunk_length  = 30, #TODO scale by fiber diameter
    perp_thresh          = 0.50, # sever if angle is above 60°
    rung_thresh          = 0.60, # rule A, sever if angle above 53°

    endpoint_avg_k = 5, # how far inside a segment the end direction is measured

    # Step 2: Gap bridging
    # TODO justify/test/evaluate
    max_gap_distance  = 24.0,
    mutual_thresh     = 0.50,
    forward_thresh    = -0.30,
    durable_thresh    = 0.40,
    durable_min_frac  = 0.40,
    n_gap_samples     = 7,
    max_commit_cost = 35.0,

    # Labeling fibers TODO
    min_fiber_length  = 5, #TODO scale by fiber diameter
    fg_threshold      = 0.10, # TODO
    vol_threshold = 0.05,
    max_watershed_voxels = 2_000_000,

    # TODO further hidden default parameters
    cut_border = 32

)


def run_direct_topology_optimization(
    intensity_path: str,
    orientation_path: str,
    volume_path: str,
    out_dir: str,
    params: dict = None,
    verbose: bool = True,
):
    """TODO DOC
    Direct whole-volume topology optimization with global degree-1 linear assignment matching."""

    if params is None:
        params = DEFAULT_PARAMS.copy()

    os.makedirs(out_dir, exist_ok=True)
    t_total = time.time()

    print("\n" + "=" * 85, flush=True)
    print(" DIRECT GLOBAL FIBER TOPOLOGY OPTIMIZER (Memory-Mapped, Degree <= 1)", flush=True)
    print("=" * 85, flush=True)

    # 1. Load memory-mapped input volumes (0 GB resident RAM for large .npy)
    intensity = load_neural_field_array(intensity_path, mmap_mode='r')
    orientation = load_neural_field_array(orientation_path, mmap_mode='r')
    if orientation.shape[0] == 3:  # (3,D,H,W) -> (D,H,W,3), components in ZYX order
        orientation = np.moveaxis(orientation, 0, -1)
    volume = load_volume_array(volume_path, mmap_mode='r')
    # -------------------------------------------------------------------------
    # Medial Axis Thinning
    # smoothing and thinning; sigma = 0.6, I = 0.28 TODO why
    # -------------------------------------------------------------------------
    skel = extract_skeleton(intensity, params['sigma_prefilter'], params['core_threshold']) # Reference: Lee, Kashyap & Chu (1994) - TODO verify reference
    if not skel.any():
        raise ValueError('Empty skeleton: core_threshold too high for this potential field?')
    skel, intensity, volume, orientation = crop_border([skel, intensity, volume, orientation], params['cut_border'])
    D, H, W = skel.shape
    fg = (np.asarray(volume) >= params['vol_threshold'] | np.asarray(intensity) > params['fg_thresh'])
    orientation = np.where(fg[..., None], orientation, 0.0).astype(np.float32)
    del fg
    ori_z, ori_y, ori_x = orientation[..., 0], orientation[..., 1], orientation[..., 2]

    # -------------------------------------------------------------------------
    # Orientation-Guided Transverse H-Severing
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85, flush=True)
    print(" Orientation-Guided Transverse H-Severing", flush=True)
    print("=" * 85, flush=True)
    t2 = time.time()

    clean_skel = sever_h_junctions(
        skeleton=skel,
        ori_z=ori_z, ori_y=ori_y, ori_x=ori_x,
        max_rung_length=params['max_rung_length'],
        min_trunk_length=params['min_trunk_length'],
        perp_thresh=params['perp_thresh'],
        rung_thresh=params['rung_thresh'],
        verbose=verbose,
    )
    del skel
    gc.collect()
    print(f"  H-severing complete in {time.time()-t2:.1f}s", flush=True)

    t3 = time.time()

    segments = build_segment_graph(
        skeleton=clean_skel,
        ori_z=ori_z, ori_y=ori_y, ori_x=ori_x,
        min_segment_length=params['min_fiber_length'],
        endpoint_avg_k=params['endpoint_avg_k'],
        radius=params['radius'],
        verbose=verbose,
    )
    del clean_skel
    gc.collect()
    print(f"  Segment graph built in {time.time()-t3:.1f}s -- {len(segments)} segments", flush=True)

    if not segments:
        print("  [Warning] No valid segments found.", flush=True)
        return {'N_final_fibers': 0, 'Total_pipeline_runtime_sec': float(time.time() - t_total)}

    t4 = time.time()

    candidates = find_bridge_candidates(
        segments=segments,
        ori_z=ori_z, ori_y=ori_y, ori_x=ori_x,
        max_gap_distance=params['max_gap_distance'],
        mutual_thresh=params['mutual_thresh'],
        forward_thresh=params['forward_thresh'],
        durable_thresh=params['durable_thresh'],
        durable_min_fraction=params['durable_min_frac'],
        n_gap_samples=params['n_gap_samples'],
        verbose=verbose,
    )
    print(f"  Gap candidate search complete in {time.time()-t4:.1f}s -- {len(candidates)} candidate bridges", flush=True)

    t5 = time.time()

    chains = optimize_topology(
        segments=segments,
        candidates=candidates,
        max_commit_cost = params["max_commit_cost"],
        verbose=verbose,
    )
    del candidates, ori_z, ori_y, ori_x, orientation
    gc.collect()
    print(f"  Topology optimization complete in {time.time()-t5:.1f}s -- {len(chains)} continuous fibers resolved", flush=True)

    print("\n" + "=" * 85, flush=True)
    print(" Label fibers", flush=True)
    print("=" * 85, flush=True)
    t6 = time.time()

    inst_skel_path = os.path.join(out_dir, 'instance_skeleton.npy')
    inst_vol_path = os.path.join(out_dir, 'instance_volume.npy')

    inst_skel = np.lib.format.open_memmap(inst_skel_path, mode='w+', dtype=np.int32, shape=(D, H, W))
    min_flen = params['min_fiber_length']
    inst_skel, n_kept_chains = build_labeled_skeleton(
        segments=segments,
        chains=chains,
        volume_shape=(D, H, W),
        min_fiber_length=min_flen,
        out_skel_mmap=inst_skel,
    )
    inst_skel.flush()
    del segments, chains
    gc.collect()

    inst_vol = np.lib.format.open_memmap(inst_vol_path, mode='w+', dtype=np.int32, shape=(D, H, W))
    label_fibers(
        inst_skel=inst_skel,
        fg_threshold=params['fg_threshold'],
        vol_threshold=params['vol_threshold'],
        max_watershed_voxels=params['max_watershed_voxels'],
        volume=volume,
        intensity=intensity,
        out_vol_mmap=inst_vol,
        verbose=verbose
    )
    inst_vol.flush()
    print(f"  Labelling complete in {time.time()-t6:.1f}s -- {n_kept_chains} continuous fibers resolved in final volume", flush=True)

    # -------------------------------------------------------------------------
    # Exporting Volumes & Centerlines
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85, flush=True)
    print(" Exporting uint16 TIFF & Centerlines", flush=True)
    print("=" * 85, flush=True)

    # Export final outputs as uint16 TIFF
    inst_vol_tif_path = os.path.join(out_dir, 'instance_volume.tif')
    inst_skel_tif_path = os.path.join(out_dir, 'instance_skeleton.tif')
    export_uint16_tiff(inst_vol, inst_vol_tif_path, verbose=verbose)
    export_uint16_tiff(inst_skel, inst_skel_tif_path, verbose=verbose)

    # Optional GeoDict GAD export
    inst_gad_path = None
    if params['export_gad']:
        inst_gad_path = os.path.join(out_dir, 'instance_fibers.gad')
        v_size = float(params['gad_voxel_size'])
        export_fibers_as_gad(
            inst_skel=inst_skel,
            inst_vol=inst_vol,
            out_path=inst_gad_path,
            voxel_size=v_size,
            is_periodic=False,
            verbose=verbose
        )

    counts_map = defaultdict(int)
    for z in range(D):
        s_sl = inst_skel[z]
        nz = s_sl[s_sl > 0]
        if len(nz) > 0:
            u_ids, u_cnts = np.unique(nz, return_counts=True)
            for uid, ucnt in zip(u_ids, u_cnts):
                counts_map[int(uid)] += int(ucnt)
    chain_lengths = list(counts_map.values())
    save_fiber_length_histogram(chain_lengths, None, os.path.join(out_dir, 'fiber_length_histogram.png'))

    export_npy = bool(params['export_npy'])
    if not export_npy:
        if hasattr(inst_vol, '_mmap') and inst_vol._mmap is not None:
            try:
                inst_vol._mmap.close()
            except Exception:
                pass
        if hasattr(inst_skel, '_mmap') and inst_skel._mmap is not None:
            try:
                inst_skel._mmap.close()
            except Exception:
                pass
        del inst_vol, inst_skel
        if 's_sl' in locals():
            del s_sl
        gc.collect()
        if os.path.exists(inst_vol_path):
            try:
                os.remove(inst_vol_path)
            except Exception:
                pass
        if os.path.exists(inst_skel_path):
            try:
                os.remove(inst_skel_path)
            except Exception:
                pass
        inst_vol_path = None
        inst_skel_path = None

    total_runtime = float(time.time() - t_total)
    metrics = {
        'N_final_fibers': len(chain_lengths),
        'Mean_fiber_length_voxels': float(np.mean(chain_lengths)) if chain_lengths else 0,
        'Max_fiber_length_voxels': float(max(chain_lengths)) if chain_lengths else 0,
        'Total_pipeline_runtime_sec': total_runtime,
        'instance_volume_tif': inst_vol_tif_path,
        'instance_skeleton_tif': inst_skel_tif_path,
    }
    if export_npy and inst_vol_path:
        metrics['instance_volume_npy'] = inst_vol_path
    if export_npy and inst_skel_path:
        metrics['instance_skeleton_npy'] = inst_skel_path
    if inst_gad_path:
        metrics['instance_fibers_gad'] = inst_gad_path

    print("\n" + "=" * 85, flush=True)
    print(f" TOPOLOGY OPTIMIZATION COMPLETE in {total_runtime:.1f}s")
    print(f"  - Final Continuous Fibers Resolved:  {len(chain_lengths)}")
    print(f"  - Mean Fiber Length:                 {metrics['Mean_fiber_length_voxels']:.1f} voxels")
    print(f"  - Max Fiber Length:                  {metrics['Max_fiber_length_voxels']:.1f} voxels")
    if export_npy and inst_vol_path:
        print(f"  - Instance Volume Array (npy):       {inst_vol_path}")
    print(f"  - Instance Volume TIFF (uint16):     {inst_vol_tif_path}")
    if export_npy and inst_skel_path:
        print(f"  - Labeled Skeleton Array (npy):      {inst_skel_path}")
    print(f"  - Labeled Skeleton TIFF (uint16):    {inst_skel_tif_path}")
    if inst_gad_path:
        print(f"  - GeoDict GAD File (.gad):           {inst_gad_path}")
    print("=" * 85 + "\n", flush=True)

    return metrics

def extract_skeleton(intensity, sigma=0.6, threshold=0.28):
    """One-voxel-wide centerline skeleton from the predicted centerline potential.

    The network regresses I(x) = exp(-d(x)^2 / 2), with d the distance (voxels) to the
    nearest fiber centerline (core/gt_generator.py), i.e. a distance-based centerline
    target as in Sironi et al. (2014). We
      1. smooth I with a Gaussian (std `sigma` voxels) to suppress voxel-level noise,
      2. keep the core I >= threshold, a tube of radius sqrt(-2 ln threshold) ~ 1.6 voxels
         around each predicted centerline (smoothing lowers the peak to ~0.75 but leaves
         this radius almost unchanged),
      3. thin the core to curves with the 3D thinning of Lee, Kashyap & Chu (1994),
         as implemented in skimage.morphology.skeletonize.
    """
    core = gaussian_filter(np.asarray(intensity, np.float32), sigma) >= threshold
    return skeletonize(core)


def crop_border(arrays, b):
    """Drops `b` voxels on every side (after skeletonization, so thinning sees the full volume)."""
    inner = (slice(b, -b if b else None),) * 3
    return [a[inner] for a in arrays]


if __name__ == '__main__':
    default_vol = get_default_volume('data/fibers_to_segment')
    default_int = get_default_intensity('outputs', volume_path=default_vol)
    default_ori = get_default_orientation('outputs', volume_path=default_vol)

    parser = argparse.ArgumentParser(description="Global Fiber Topology Optimization Runner")
    parser.add_argument('--volume', type=str, default=default_vol,
                        help=f"Path to raw/binary volume (.tif, .tiff, .npy) (default: {default_vol})")
    parser.add_argument('--intensity', type=str, default=default_int,
                        help=f"Path to predicted intensity .npy/.tif (default: {default_int})")
    parser.add_argument('--orientation', type=str, default=default_ori,
                        help=f"Path to predicted orientation .npy/.tif (default: {default_ori})")
    parser.add_argument('--out', type=str, default='outputs/topology_resolved',
                        help="Output directory (default: outputs/topology_resolved)")
    parser.add_argument('--sigma-prefilter', type=float, default=0.60, help="Gaussian smoothing sigma before skeletonization (default: 0.60)")
    parser.add_argument('--core-threshold', type=float, default=0.28, help="Potential field threshold for centerline core extraction (default: 0.28)")
    parser.add_argument('--gad', default=None, help="Optional path to GAD ground truth file")
    parser.add_argument('--cut-border', type=int, default=32, help="Margin in voxels to cut from boundary to eliminate loop artifacts (default: 32)")
    parser.add_argument('--min-fiber-length', type=int, default=5,
                        help="Minimum fiber centerline length in voxels (default: 5). Shorter fibers are filtered before labelling.")
    parser.add_argument('--export-gad', action='store_true', default=False,
                        help="Export resolved fibers to GeoDict .gad format (default: False, non-active)")
    parser.add_argument('--gad-voxel-size', type=float, default=1e-06,
                        help="Physical voxel size in meters for .gad export (default: 1e-06)")
    parser.add_argument('--export-npy', action='store_true', default=False,
                        help="Export resolved volumes and centerlines as .npy arrays (default: False)")
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    params['sigma_prefilter'] = args.sigma_prefilter
    params['core_threshold'] = args.core_threshold
    params['cut_border'] = args.cut_border
    params['min_fiber_length'] = args.min_fiber_length
    params['min_chain_length'] = args.min_fiber_length
    params['export_gad'] = args.export_gad
    params['gad_voxel_size'] = args.gad_voxel_size
    params['export_npy'] = args.export_npy

    run_direct_topology_optimization(
        intensity_path=args.intensity,
        orientation_path=args.orientation,
        volume_path=args.volume,
        out_dir=args.out,
        params=params
    )
