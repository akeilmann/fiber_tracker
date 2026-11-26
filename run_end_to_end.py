"""
run_end_to_end.py
=================
Master 1-Click End-to-End Pipeline:
  1. Sliding-Window Neural Field Inference (Intensity & Orientation U-Nets)
  2. 3D Medial Axis Extraction & Spur Pruning
  3. Orientation-Guided Transverse H-Severing
  4. Fragment Graph Construction with Robust Durable Endpoint Averaging
  5. Post-H-Sever Direction-Durable Bridge Candidate Search
  6. Global Minimum-Cost Topology Optimization
  7. Short Fiber Filtering (< 5 vx)
  8. Connected-Component Bounded Voronoi Instance Diffusion

Usage:
  python fiber_resolution_pipeline/run_end_to_end.py
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from inference.run_inference import run_inference, get_default_input_volume
from inference.run_topology_optimization import run_direct_topology_optimization, DEFAULT_PARAMS
from orientation.structure_tensor import compute_structure_tensor_orientation


def main():
    _default_input = get_default_input_volume()
    parser = argparse.ArgumentParser(description="Master End-to-End Fiber Resolution Pipeline")
    parser.add_argument('--input', type=str, default=_default_input, help=f"Input microscopy volume (.tif) (default: {_default_input})")
    parser.add_argument('--intensity-ckpt', type=str, default='checkpoints/best_intensity_unet.pth')
    parser.add_argument('--orientation-ckpt', type=str, default='checkpoints/best_orientation_unet.pth')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--cut-border', type=int, default=32, help="Margin in voxels (e.g. 32) to sever outer boundary loops")
    parser.add_argument('--out', type=str, default='outputs/fiber_resolution_final')
    parser.add_argument('--chunk-size', type=int, default=32, help="Slice chunk size for memory-safe streaming")
    parser.add_argument('--min-fiber-length', type=int, default=5, help="Minimum fiber centerline length in voxels (default: 5)")
    parser.add_argument('--export-gad', action='store_true', default=False,
                        help="Export resolved fibers to GeoDict .gad format (default: False, non-active)")
    parser.add_argument('--gad-voxel-size', type=float, default=1e-06,
                        help="Physical voxel size in meters for .gad export (default: 1e-06)")
    parser.add_argument('--export-npy', action='store_true', default=False,
                        help="Export resolved volumes and centerlines as .npy arrays (default: False)")
    parser.add_argument('--orientation-method', choices=['unet', 'structure_tensor', 'precomputed'],
                        default='structure_tensor')
    parser.add_argument('--orientation-file', type=str, default=None,
                        help="For 'precomputed': float32 .npy, (3,D,H,W), components (vz,vy,vx)")
    parser.add_argument('--st-sigma', type=float, default=1.0)
    parser.add_argument('--st-rho', type=float, default=6.0)
    parser.add_argument('--spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        metavar=('DZ', 'DY', 'DX'))
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    temp_prefix = os.path.join(args.out, 'neural_fields')

    print("=" * 90)
    print(" EXECUTING END-TO-END FIBER RESOLUTION & TOPOLOGY OPTIMIZATION PIPELINE ")
    print("=" * 90)
    mode = 'both' if args.orientation_method == 'unet' else 'intensity'
    print(args.orientation_method)

    # 1. Neural Field Inference
    int_path, ori_path, vol_path = run_inference(
        input_path=args.input,
        intensity_ckpt=args.intensity_ckpt,
        orientation_ckpt=args.orientation_ckpt,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        output_prefix=temp_prefix,
        chunk_size=args.chunk_size,
        export_npy=args.export_npy,
        mode=mode,
    )
    if args.orientation_method == 'structure_tensor':
        ori_path = f"{temp_prefix}_orientation_st.npy"
        compute_structure_tensor_orientation(
            args.input, ori_path,
            sigma=args.st_sigma, rho=args.st_rho
        )
    elif args.orientation_method == 'precomputed':
        if not args.orientation_file or not os.path.isfile(args.orientation_file):
            raise FileNotFoundError("--orientation-method precomputed requires --orientation-file")
        ori_path = args.orientation_file

    if not ori_path or not os.path.isfile(ori_path):
        raise FileNotFoundError(f"Orientation field not found: {ori_path}")

    # 2. Global Topology Optimization
    opt_params = DEFAULT_PARAMS.copy()
    opt_params['cut_border'] = args.cut_border
    opt_params['min_fiber_length'] = args.min_fiber_length
    opt_params['min_chain_length'] = args.min_fiber_length
    opt_params['export_gad'] = args.export_gad
    opt_params['gad_voxel_size'] = args.gad_voxel_size
    opt_params['export_npy'] = args.export_npy

    metrics = run_direct_topology_optimization(
        intensity_path=int_path,
        orientation_path=ori_path,
        volume_path=vol_path,
        out_dir=args.out,
        params=opt_params
    )

    # Clean up intermediate neural field .npy files if export_npy is False
    if not args.export_npy:
        cleanup = (int_path, vol_path) if args.orientation_method == 'precomputed' else (int_path, ori_path, vol_path)
        for p in cleanup:
            if p and os.path.isfile(p) and p.endswith('.npy') and os.path.abspath(p).startswith(os.path.abspath(args.out)):
                try:
                    os.remove(p)
                except Exception:
                    pass

    print("=" * 90)
    print(f" ALL PIPELINE STAGES COMPLETED IN {time.time() - t_start:.2f}s!")
    print(f" Total Continuous Fibers Resolved: {metrics['N_final_fibers']}")
    print(f" Output Directory:                  {args.out}")
    print("=" * 90)


if __name__ == '__main__':
    main()
