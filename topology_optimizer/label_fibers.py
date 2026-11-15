"""
step5_diffuse_labels.py
========================
Voronoi label diffusion: expand skeleton chain labels to fill the full fiber volume.

After topology optimization we have a labeled centerline skeleton. This step
diffuses instance labels outward into the foreground binary mask using a
fast, memory-efficient nearest-neighbor (Voronoi / cKDTree) approach.

Bridges are drawn into the skeleton before diffusion so the gap regions are
also labeled.
TODO reference Griessner
TODO watershed or euclidean
"""

import gc
from collections import defaultdict
import numpy as np
from scipy.ndimage import label as nd_label, find_objects
from scipy.spatial import cKDTree
from skimage.draw import line_nd
from skimage.segmentation import watershed

try:
    from .skeleton_graph import FiberFragment
    from .link_fibers import FiberChain
except ImportError:
    from skeleton_graph import FiberFragment
    from link_fibers import FiberChain


def build_labeled_skeleton(
    fragments: list[FiberFragment],
    chains: list[FiberChain],
    seg_labels: np.ndarray | None,
    volume_shape: tuple[int, int, int],
    min_chain_length: int = 5,
    out_skel_mmap: np.ndarray | None = None,
    verbose: bool = True
) -> tuple[np.ndarray, int]:
    """
    Build a labeled skeleton volume from chains, removing short fibers (< min_chain_length).

    Parameters
    ----------
    fragments        : list[FiberFragment]
    chains           : list[FiberChain]
    seg_labels       : (D, H, W) int32 or None
    volume_shape     : (D, H, W)
    min_chain_length : int -- chains shorter than this are discarded (default: 5)
    out_skel_mmap    : optional pre-allocated or memmapped array to write to
    verbose          : bool

    Returns
    -------
    inst_skel : (D, H, W) int32 -- labeled skeleton (0 = background)
    n_chains  : int -- number of kept chains
    """
    kept = [c for c in chains if c.total_length >= min_chain_length]
    label_of = {fid: k for k, c in enumerate(kept, start=1) for fid in c.fragment_ids}
    skel = out_skel_mmap if out_skel_mmap is not None else np.zeros(volume_shape, np.int32)
    skel[:] = 0
    for f in fragments:  # fragment voxels
        if f.frag_id in label_of:
            skel[tuple(f.coords.T)] = label_of[f.frag_id]
    for k, c in enumerate(kept, start=1):  # straight bridge lines
        for a, b in c.bridge_coords:
            skel[line_nd(np.round(a).astype(int), np.round(b).astype(int), endpoint=True)] = k

    return skel, len(kept)

def build_foreground(inst_skel, foreground_mask=None, volume=None, intensity=None,
                     fg_threshold=0.10, vol_threshold=0.05):
    """Foreground = skeleton | given mask, or skeleton | volume > vol_threshold | intensity >= fg_threshold."""
    foreground = np.zeros(inst_skel.shape, bool)
    for z in range(inst_skel.shape[0]):                  # slice by slice: no volume-sized temporaries
        fg_slice = inst_skel[z] > 0
        if foreground_mask is not None:
            fg_slice |= foreground_mask[z] > 0
        else:
            if volume is not None:
                fg_slice |= volume[z] > vol_threshold
            if intensity is not None:
                fg_slice |= intensity[z] >= fg_threshold
        foreground[z] = fg_slice
    return foreground


def classify_components(n_cc, seed_cc, seed_labels):
    """Label of every component that contains exactly one fiber (else 0), and the components shared by several."""
    pairs = np.unique(np.column_stack([seed_cc, seed_labels]), axis=0)      # (component, label)
    n_fibers = np.bincount(pairs[:, 0], minlength=n_cc + 1)                # fibers per component
    label_of_component = np.zeros(n_cc + 1, np.int32)
    single = n_fibers[pairs[:, 0]] == 1                                     # pairs in one-fiber components
    label_of_component[pairs[single, 0]] = pairs[single, 1]
    return label_of_component, np.flatnonzero(n_fibers > 1)


def watershed_component(cc, c, box, seeds, seed_labels, out):
    """Geodesic nearest seed inside component c: watershed on a constant image, written into out[box]."""
    mask = cc[box] == c
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple((seeds - [s.start for s in box]).T)] = seed_labels
    region = out[box]
    region[mask] = watershed(np.zeros(mask.shape, np.uint8), markers=markers, mask=mask, connectivity=3)[mask]


def write_labels(components, label_of_component, nearest_seed_search, labeled_skeleton,
                 labels_out, chunk_size):
    """
    Write the final fiber labels into labels_out, slice by slice.

    Per voxel, in this order: the label of its component if the component holds one
    fiber; the watershed result already in labels_out; the label of the Euclidean
    nearest seed for components in nearest_seed_search; skeleton voxels keep their
    own label.
    """
    for z in range(components.shape[0]):
        component_slice = components[z]
        label_slice = label_of_component[component_slice]

        watershed_done = labels_out[z] > 0
        label_slice[watershed_done] = labels_out[z][watershed_done]

        for component, (seed_tree, seed_labels, z_range) in nearest_seed_search.items():
            if not z_range.start <= z < z_range.stop:
                continue
            y, x = np.nonzero(component_slice == component)
            points = np.column_stack([np.full(len(y), z), y, x]).astype(np.float32)
            for start in range(0, len(points), chunk_size):
                chunk = slice(start, start + chunk_size)
                _, nearest = seed_tree.query(points[chunk], workers=-1)
                label_slice[y[chunk], x[chunk]] = seed_labels[nearest]

        skeleton_slice = labeled_skeleton[z]
        on_skeleton = skeleton_slice > 0
        label_slice[on_skeleton] = skeleton_slice[on_skeleton]

        labels_out[z] = label_slice


def label_fibers(inst_skel, foreground_mask=None, volume=None, intensity=None, fg_threshold=0.10,
                 vol_threshold=0.05, chunk_size=1_000_000, max_watershed_voxels=2_000_000,
                 verbose=True, out_vol_mmap=None):

    out = out_vol_mmap if out_vol_mmap is not None else np.zeros(inst_skel.shape, np.int32)
    out[:] = 0
    seeds = np.argwhere(inst_skel)
    if len(seeds) == 0:
        return out
    seed_labels = inst_skel[tuple(seeds.T)]

    S26 = np.ones((3, 3, 3), bool)
    cc, n_cc = nd_label(build_foreground(inst_skel, foreground_mask, volume, intensity,
                                         fg_threshold, vol_threshold), S26)
    seed_cc = cc[tuple(seeds.T)]
    label_of_component, shared_components = classify_components(n_cc, seed_cc, seed_labels)

    boxes = find_objects(cc) if len(shared_components) else []
    trees = {}
    for component in shared_components:
        box, sel = boxes[component - 1], seed_cc == component
        if 0 < np.prod([s.stop - s.start for s in box]) <= max_watershed_voxels:
            watershed_component(cc, component, box, seeds[sel], seed_labels[sel], out)
        else:
            trees[component] = (cKDTree(seeds[sel].astype(np.float32)), seed_labels[sel], box[0])

    write_labels(cc, label_of_component, trees, inst_skel, out, chunk_size)
    if hasattr(out, 'flush'):
        out.flush()
    if verbose:
        print(f"  [label] {n_cc} foreground components, {len(shared_components)} with several fibers", flush=True)
    return out

