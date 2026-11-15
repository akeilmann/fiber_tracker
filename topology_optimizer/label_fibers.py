#TODO reference Griessner
#TODO watershed or euclidean

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
) -> tuple[np.ndarray, int]:
    """
    Draw the fiber chains into a labeled centerline volume.

    Chains with a total fragment length below *min_chain_length* voxels are
    discarded.  The remaining chains are numbered 1, 2, ... in the given
    order (longest first, as returned by optimize_topology).  The voxels of
    every fragment are set to the label of its chain, and every accepted
    bridge is drawn as a straight digital line with the same label.

    :param fragments: list[FiberFragment]
        fragments from build_fragment_graph
    :param chains: list[FiberChain]
        chains from optimize_topology
    :param seg_labels: np.ndarray or None
        unused; kept for compatibility with the driver
    :param volume_shape: tuple[int, int, int]
        shape (D, H, W) of the output volume
    :param min_chain_length: int, default 5
        minimum total fragment length of a chain in voxels
    :param out_skel_mmap: np.ndarray of int32, shape (D, H, W), or None
        array to write into, e.g. a memory-mapped file; a new array is
        created if None
    :return: tuple[np.ndarray, int]
        labeled centerline volume (int32, 0 = background) and the number of
        kept chains
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
    """
    Build the binary foreground (fiber voxels) to be labeled.

    A voxel is foreground if it is a centerline voxel or, if *foreground_mask*
    is given, set in that mask; otherwise if ``volume > vol_threshold`` or
    ``intensity >= fg_threshold``.  The mask is built slice by slice, so no
    volume-sized temporary arrays are created.

    :param inst_skel: np.ndarray of int, shape (D, H, W)
        labeled centerline volume, 0 = background
    :param foreground_mask: np.ndarray, shape (D, H, W), or None
        given foreground; if set, volume and intensity are ignored
    :param volume: np.ndarray, shape (D, H, W), or None
        binary or gray-value volume
    :param intensity: np.ndarray of float, shape (D, H, W), or None
        predicted centerline potential
    :param fg_threshold: float, default 0.10
        intensity threshold
    :param vol_threshold: float, default 0.05
        volume threshold
    :return: np.ndarray of bool, shape (D, H, W)
        foreground mask
    """
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
    """
    Find out how many fibers each connected component of the foreground contains.

    :param n_cc: int
        number of connected components, labeled 1, ..., n_cc
    :param seed_cc: np.ndarray of int, shape (n,)
        component of every centerline voxel (seed)
    :param seed_labels: np.ndarray of int, shape (n,)
        fiber label of every seed
    :return: tuple[np.ndarray, np.ndarray]
        label_of_component, shape (n_cc + 1,): the fiber label of every
        component that contains exactly one fiber, 0 otherwise (also for the
        background, index 0); and the components shared by several fibers
    """
    pairs = np.unique(np.column_stack([seed_cc, seed_labels]), axis=0)      # (component, label)
    n_fibers = np.bincount(pairs[:, 0], minlength=n_cc + 1)                # fibers per component
    label_of_component = np.zeros(n_cc + 1, np.int32)
    single = n_fibers[pairs[:, 0]] == 1                                     # pairs in one-fiber components
    label_of_component[pairs[single, 0]] = pairs[single, 1]
    return label_of_component, np.flatnonzero(n_fibers > 1)


def watershed_component(cc, c, box, seeds, seed_labels, out):
    """
    Label a shared component by the geodesically nearest seed.

    Runs a watershed (TODO reference) from the seeds on a constant
    image inside the component, so each voxel receives the label of the seed
    whose flooding reaches it first along a path within the component, i.e.
    the geodesic influence zones of the seeds (TODO reference).

    :param cc: np.ndarray of int, shape (D, H, W)
        connected-component labels of the foreground
    :param c: int
        component to label
    :param box: tuple[slice, slice, slice]
        bounding box of the component, from find_objects
    :param seeds: np.ndarray of int, shape (n, 3)
        (z, y, x) coordinates of the seeds in the component
    :param seed_labels: np.ndarray of int, shape (n,)
        fiber label of every seed
    :param out: np.ndarray of int, shape (D, H, W)
        output volume; the voxels of the component are written into out[box]
    """
    mask = cc[box] == c
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple((seeds - [s.start for s in box]).T)] = seed_labels
    region = out[box]
    region[mask] = watershed(np.zeros(mask.shape, np.uint8), markers=markers, mask=mask, connectivity=3)[mask]


def write_labels(components, label_of_component, nearest_seed_search, labeled_skeleton,
                 labels_out, chunk_size):
    """
    Write the final fiber labels into labels_out, slice by slice.

    Each foreground voxel receives, in this order: the label of its component
    if the component contains one fiber; the watershed result already in
    labels_out; for the components in *nearest_seed_search*, the label of the
    Euclidean nearest seed.  Centerline voxels keep their own label.

    :param components: np.ndarray of int, shape (D, H, W)
        connected-component labels of the foreground
    :param label_of_component: np.ndarray of int, shape (n_cc + 1,)
        from classify_components
    :param nearest_seed_search: dict[int, tuple[cKDTree, np.ndarray, slice]]
        for every component labeled by Euclidean distance: KD-tree of its
        seeds, their labels and the z range of the component
    :param labeled_skeleton: np.ndarray of int, shape (D, H, W)
        labeled centerline volume
    :param labels_out: np.ndarray of int, shape (D, H, W)
        output volume, already containing the watershed results
    :param chunk_size: int
        maximum number of voxels per KD-tree query
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
    """
    Assign every foreground voxel to a fiber, starting from the labeled centerlines.

    As in Griesser et al. (2023), the labeled centerlines serve as seeds.
    Within each 26-connected component of the foreground (*build_foreground*):

    * a component with the seeds of one fiber receives that fiber's label;
    * a component shared by several fibers is divided by the nearest seed:
      geodesically by a watershed (*watershed_component*) if its bounding box
      has at most *max_watershed_voxels* voxels, by Euclidean distance
      (*write_labels*) otherwise;
    * a component without seeds stays unlabeled (0).

    Labels are never passed between components, i.e. across background.

    :param inst_skel: np.ndarray of int, shape (D, H, W)
        labeled centerline volume from build_labeled_skeleton, 0 = background
    :param foreground_mask: np.ndarray, shape (D, H, W), or None
        given foreground, see build_foreground
    :param volume: np.ndarray, shape (D, H, W), or None
        binary or gray-value volume, see build_foreground
    :param intensity: np.ndarray of float, shape (D, H, W), or None
        predicted centerline potential, see build_foreground
    :param fg_threshold: float, default 0.10
        intensity threshold, see build_foreground
    :param vol_threshold: float, default 0.05
        volume threshold, see build_foreground
    :param chunk_size: int, default 1000000
        maximum number of voxels per KD-tree query
    :param max_watershed_voxels: int, default 2000000
        largest bounding box (in voxels) of a shared component that is
        labeled by watershed; 0 labels all shared components by Euclidean
        distance
    :param verbose: bool, default True
        whether to print the number of components
    :param out_vol_mmap: np.ndarray of int32, shape (D, H, W), or None
        array to write into, e.g. a memory-mapped file; a new array is
        created if None
    :return: np.ndarray of int32, shape (D, H, W)
        fiber label of every voxel, 0 = background or unlabeled
    """
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

