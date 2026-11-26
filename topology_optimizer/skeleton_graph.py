# TODO make sure "rules" are correct - are they new?
# TODO justify thresholds; max_rung_length Huang scales by fiber diameter?
# TODO further renaming/making consistent


import gc
import numpy as np
from scipy.ndimage import convolve, label as nd_label
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
from dataclasses import dataclass, field
import time


from topology_optimizer.orientation_utils import axial_mean

@dataclass
class FiberSegment:
    """
    A junction-free piece of the skeleton, i.e. a simple path of 26-connected voxels.

    Which end is called head and which tail is arbitrary.  Both end directions
    point outward, away from the segment.

    :param seg_id: int
        segment ID, starting at 1
    :param coords: np.ndarray of int, shape (n, 3)
        (z, y, x) coordinates of the segment voxels in raster order
    :param head_coord: np.ndarray of float, shape (3,)
        coordinates of the first end voxel
    :param tail_coord: np.ndarray of float, shape (3,)
        coordinates of the second end voxel
    :param head_ori: np.ndarray of float, shape (3,)
        fiber direction at the head from the orientation field, unit length
    :param tail_ori: np.ndarray of float, shape (3,)
        fiber direction at the tail from the orientation field, unit length
    :param head_tangent: np.ndarray of float, shape (3,)
        unit vector from the anchor voxel to the head (geometry only)
    :param tail_tangent: np.ndarray of float, shape (3,)
        unit vector from the anchor voxel to the tail (geometry only)

    The attribute *length* (int) is set to the number of voxels.
    """
    seg_id: int
    coords: np.ndarray

    head_coord:   np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tail_coord:   np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    head_ori: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tail_ori: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    head_tangent: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tail_tangent: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def __post_init__(self):
        self.length = len(self.coords)


def classify_segment(voxels, n_junctions,
                    trunk_oris, ori_z, ori_y, ori_x,
                    rung_thresh, perp_thresh):
    """
    Decides whether a short skeleton segment is an H-junction rung.

    Classification is inspired by Huang et al. (2016) (TODO plus more rules?).
    A rung is a short bridge between two fibers; the trunks are the long
    segments that clearly belong to a fiber::

        fiber 1       fiber 2
          o             o      <- trunk
          o             o
          #ooooooooooooo#      <- rung
          o             o
          o             o

        # = junction voxels

    The rules are checked in order and the first one that applies is
    returned:

    * **Rule 1** – the segment is a single voxel touching at least one
      junction.  A single voxel has no direction, so it is removed.
    * **Rule A** – the segment connects at least two junctions and its
      end-to-end direction deviates from the mean image orientation along
      the segment, i.e. ``|dot(segment_dir, o_mean)| < rung_thresh``.
    * **Rule B** – the segment direction is close to perpendicular to the
      mean orientation of the neighboring trunks, i.e.
      ``1 - |dot(segment_dir, trunk_mean_ori)| > perp_thresh``.

    :param voxels: np.ndarray of int, shape (n, 3)
        (z, y, x) coordinates of the segment voxels; the segment direction is
        taken from the first to the last row
    :param n_junctions: int
        number of distinct junction clusters the segment touches
    :param trunk_oris: np.ndarray of float, shape (k, 3)
        mean orientations of the long neighboring segments (trunks); may be
        empty, in which case Rule B is skipped
    :param ori_z: np.ndarray of float, shape (D, H, W)
        z component of the unit orientation field
    :param ori_y: np.ndarray of float, shape (D, H, W)
        y component of the unit orientation field
    :param ori_x: np.ndarray of float, shape (D, H, W)
        x component of the unit orientation field
    :param rung_thresh: float
        threshold for Rule A on ``|cos|`` of the angle between segment and
        image orientation; 0.60 means the segment makes more than ~53 deg
        with the image orientation
    :param perp_thresh: float
        threshold for Rule B on ``1 - |cos|`` of the angle between segment
        and trunk orientation; 0.50 means the segment makes more than
        60 deg with the trunks
    :return: str or None
        ``'1'``, ``'A'`` or ``'B'`` for the rule that marks the segment as a
        rung, or None if the segment is kept
    """
    # Rule 1: a single voxel has no direction; remove it if it touches a junction
    if len(voxels) == 1:
        return '1' if n_junctions >= 1 else None

    segment_vec = (voxels[-1] - voxels[0]).astype(np.float32)
    segment_dir = segment_vec / np.linalg.norm(segment_vec)

    # Rule A: compare graph orientation along the rung candidate to image orientation
    if n_junctions >= 2:
        idx = tuple(voxels.T)
        o_mean, _ = axial_mean(np.column_stack([ori_z[idx], ori_y[idx], ori_x[idx]]))
        if o_mean is not None and abs(np.dot(segment_dir, o_mean)) < rung_thresh:
            return 'A'

    # Rule B: compare rung candidate direction to fiber direction (long neighbors, trunks)
    # TODO point out differences to Huang
    if len(trunk_oris) > 0:
        trunk_mean_ori, _ = axial_mean(trunk_oris)
        if 1.0 - abs(np.dot(segment_dir, trunk_mean_ori)) > perp_thresh:
            return 'B'
    return None

def sever_h_junctions(
    skeleton: np.ndarray,
    ori_z: np.ndarray,
    ori_y: np.ndarray,
    ori_x: np.ndarray,
    max_rung_length: int,
    min_trunk_length: int,
    perp_thresh: float,
    rung_thresh: float,
    verbose: bool = True,
) -> np.ndarray:
    """
    Removes H-junction rungs from a binary skeleton, inspired by Huang et al. (2016).

    The skeleton is split into segment by removing all junction voxels
    (voxels with at least three 26-neighbors).  Junction voxels are grouped
    into junction clusters, and two segments are neighbors if they touch the
    same cluster.  Every segment with at most *max_rung_length* voxels is then
    classified by *classify_segment*, with its neighbors of at least
    *min_trunk_length* voxels serving as trunks.  Segments classified as
    rungs are deleted from a copy of the skeleton; junction voxels are kept.

    :param skeleton: np.ndarray of bool, shape (D, H, W)
        binary skeleton
    :param ori_z: np.ndarray of float, shape (D, H, W)
        z component of the unit orientation field
    :param ori_y: np.ndarray of float, shape (D, H, W)
        y component of the unit orientation field
    :param ori_x: np.ndarray of float, shape (D, H, W)
        x component of the unit orientation field
    :param max_rung_length: int
        segments with more voxels than this are never rungs
    :param min_trunk_length: int, default 30
        minimum number of voxels for a neighboring segment to count as a
        trunk in Rule B; avoids comparing short segments to each other
    :param perp_thresh: float, default 0.50
        sever if ``1 - |dot(rung_dir, trunk_ori)| > perp_thresh``;
        0.50 means the rung makes more than 60 deg with the trunks
    :param rung_thresh: float, default 0.60
        sever if ``|dot(rung_dir, image_ori)| < rung_thresh``;
        0.60 means the rung makes more than ~53 deg with the image
        orientation along it
    :param verbose: bool, default True
        whether to print the number of severed rungs and removed voxels
    :return: np.ndarray of bool, shape (D, H, W)
        copy of the skeleton with the rung segments removed
    """

    D, H, W = skeleton.shape
    NBRS = np.ones((3, 3, 3), np.uint8)
    NBRS[1, 1, 1] = 0  # the 26 neighbours, not the voxel itself
    S26 = np.ones((3, 3, 3), bool)  # 26-connectivity for labelling

    # Step 1: label all segments (non-junction components)
    n_nbrs_per_voxel = convolve(skeleton.astype(np.uint8), NBRS, mode='constant')
    junction_mask = skeleton & (n_nbrs_per_voxel >= 3)
    labeled_segments, n_segments = nd_label(skeleton & ~junction_mask, S26)  # segments = skeleton minus junctions
    if n_segments == 0:
        return skeleton.copy()

    # Step 2: compute per-segment orientations & collect coords
    segment_z, segment_y, segment_x = np.where(labeled_segments > 0)
    segment_ids = labeled_segments[segment_z, segment_y, segment_x]
    segmen_sizes = np.bincount(segment_ids, minlength=n_segments + 1)

    # Axial mean orientation
    v = np.column_stack([ori_z[segment_z, segment_y, segment_x], ori_y[segment_z, segment_y, segment_x], ori_x[segment_z, segment_y, segment_x]]).astype(np.float64)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    T = np.zeros((n_segments + 1, 3, 3))
    for i in range(3):
        for j in range(3):
            T[:, i, j] = np.bincount(segment_ids, weights=v[:, i] * v[:, j], minlength=n_segments + 1)
    _, evecs = np.linalg.eigh(T)  # per segment; eigenvalues in ascending order
    segment_mean_ori = evecs[:, :, -1]  # unit vector per segment, arbitrary sign; row 0 unused

    # Step 3: collect start/end coords per segment for rung_dir
    sort_order = np.argsort(segment_ids, kind='stable')
    segment_z, segment_y, segment_x, segment_ids = segment_z[sort_order], segment_y[sort_order], segment_x[sort_order], segment_ids[sort_order]
    unique_bids, split_idx = np.unique(segment_ids, return_index=True)
    coord_z_splits = np.split(segment_z, split_idx[1:])
    coord_y_splits = np.split(segment_y, split_idx[1:])
    coord_x_splits = np.split(segment_x, split_idx[1:])


    # Step 4: build segment adjacency THROUGH junction clusters
    # segment are separated by junction voxels; segment that touch the
    # same junction cluster are adjacent neighbors.
    jz, jy, jx = np.where(junction_mask)
    del junction_mask
    gc.collect()

    segment_to_juncs = defaultdict(set)
    junc_to_segment = defaultdict(set)

    if len(jz) > 0:
        # group junction voxels into junctions
        junc_coords = np.column_stack([jz, jy, jx])
        tree = cKDTree(junc_coords)
        pairs = tree.query_pairs(r=1.0, p=np.inf) # 26-neighbors
        # save pairs of junction voxels as adjacency matrix with ID of group
        if len(pairs) > 0:
            row, col = zip(*pairs)
            n_pts = len(junc_coords)

            adj_mat = csr_matrix((np.ones(len(row), dtype=bool), (row, col)), shape=(n_pts, n_pts))
            _, junc_labels = connected_components(adj_mat, directed=False)
        else:
            junc_labels = np.arange(len(junc_coords))

        # record each junction-segment-pair
        for k in range(len(junc_coords)):
            cz, cy, cx = int(jz[k]), int(jy[k]), int(jx[k])
            jid = int(junc_labels[k]) + 1
            z0, z1 = max(0, cz - 1), min(D, cz + 2)
            y0, y1 = max(0, cy - 1), min(H, cy + 2)
            x0, x1 = max(0, cx - 1), min(W, cx + 2)
            touching_bids = np.unique(labeled_segments[z0:z1, y0:y1, x0:x1])
            for segment_id in touching_bids:
                if segment_id > 0:
                    segment_to_juncs[segment_id].add(jid)
                    junc_to_segment[jid].add(segment_id)

    # We are completely done with labeled_segments (free 8.4 GB!)
    del labeled_segments
    gc.collect()

    # make each segment-pair with junction between each other neighbors
    adjacency: list[set] = [set() for _ in range(n_segments + 1)]
    for jid, br_set in junc_to_segment.items():
        br_list = list(br_set)
        for i in range(len(br_list)):
            for j in range(len(br_list)):
                if i != j:
                    adjacency[br_list[i]].add(br_list[j])

    # Step 5: classify each short segment - severing decision
    sever_mask    = np.zeros(n_segments + 1, dtype=bool)

    for idx, segment_id in enumerate(unique_bids):
        segment_id = int(segment_id)
        if segmen_sizes[segment_id] > max_rung_length:
            continue
        voxels = np.column_stack([coord_z_splits[idx], coord_y_splits[idx], coord_x_splits[idx]])
        trunk_neighbors = [nb for nb in adjacency[segment_id] if segmen_sizes[nb] >= min_trunk_length]
        rule = classify_segment(voxels, len(segment_to_juncs.get(segment_id, ())),
                               segment_mean_ori[trunk_neighbors], ori_z, ori_y, ori_x,
                               rung_thresh, perp_thresh)
        if rule is not None:
            sever_mask[segment_id] = rule

    # Step 6: remove severed segments (coordinate-based, 0 GB overhead)
    clean_skel = skeleton.copy()
    severed_bids_set = set(np.where(sever_mask)[0])
    for idx, segment_id in enumerate(unique_bids):
        if int(segment_id) in severed_bids_set:
            cz_s = coord_z_splits[idx]
            cy_s = coord_y_splits[idx]
            cx_s = coord_x_splits[idx]
            clean_skel[cz_s, cy_s, cx_s] = False

    if verbose:
        print(f"  [H-sever] Severed {int(sever_mask.sum())} H-junction rungs "
              f"(short<={max_rung_length}, neighbor>={min_trunk_length}, "
              f"perp>{perp_thresh}).", flush=True)
        removed = int(skeleton.sum()) - int(clean_skel.sum())
        print(f"  [H-sever] Skeleton: {int(skeleton.sum())} -> "
              f"{int(clean_skel.sum())} ({removed} voxels removed).", flush=True)

    return clean_skel


def end_voxels(coords, is_end):
    """
    Return the two end voxels of a segment.

    Every segment voxel has at most two neighbors, so a segment is a path
    (exactly two end voxels), a single voxel (both ends coincide), or a
    closed loop (no end voxels).  A loop has no natural ends; its first voxel
    and the voxel farthest from it are returned.

    :param coords: np.ndarray of int, shape (n, 3)
        (z, y, x) coordinates of the segment voxels
    :param is_end: np.ndarray of bool, shape (n,)
        True for voxels with exactly one neighbor within the segment
    :return: tuple[np.ndarray, np.ndarray]
        coordinates of the head and the tail voxel
    """
    ends = coords[is_end]
    if len(ends) == 2:
        return ends[0], ends[1]
    if len(coords) == 1:
        return coords[0], coords[0]
    far = np.argmax(((coords - coords[0]) ** 2).sum(1))      # closed loop: no natural ends
    return coords[0], coords[far]


def end_direction(coords, end, ori_z, ori_y, ori_x, offset, radius):
    """
    Estimate the fiber direction at one end of a segment.

    The estimate is taken around an anchor voxel *offset* voxels inside the
    segment (ranked by distance to the end), so that the junction or contact
    at the end itself does not disturb it.  The orientation is the axial mean
    of the orientation field in the (2 * radius + 1)^3 window around the
    anchor; averaging over the whole fiber cross-section suppresses the noise
    of single voxels.  It is signed to point outward, from the anchor through
    the end.  If the window contains no orientation, the geometric tangent is
    used instead.

    :param coords: np.ndarray of int, shape (n, 3)
        (z, y, x) coordinates of the segment voxels
    :param end: np.ndarray, shape (3,)
        coordinates of the end voxel
    :param ori_z: np.ndarray of float, shape (D, H, W)
        z component of the unit orientation field
    :param ori_y: np.ndarray of float, shape (D, H, W)
        y component of the unit orientation field
    :param ori_x: np.ndarray of float, shape (D, H, W)
        x component of the unit orientation field
    :param offset: int, default 5
        distance rank of the anchor voxel from the end
    :param radius: int, default 2
        half edge length of the averaging window in voxels
    :return: tuple[np.ndarray, np.ndarray]
        orientation and tangent, both float32 unit vectors of shape (3,)
        pointing outward; the tangent is the unit vector from the anchor to
        the end (geometry only), or the orientation if the two coincide
    """
    anchor = coords[np.argsort(((coords - end) ** 2).sum(1))[min(offset, len(coords) - 1)]]
    win = tuple(slice(max(a - radius, 0), a + radius + 1) for a in anchor)   # slices clip at the far edge
    ori, _ = axial_mean(np.column_stack([o[win].ravel() for o in (ori_z, ori_y, ori_x)]))
    out = (end - anchor).astype(np.float64)
    tangent = out / np.linalg.norm(out) if np.linalg.norm(out) > 1e-4 else None
    if ori is None:                       # no predicted orientation around the anchor
        ori = tangent if tangent is not None else np.array([1.0, 0.0, 0.0])
    if ori @ out < 0:
        ori = -ori
    return ori.astype(np.float32), (ori if tangent is None else tangent).astype(np.float32)


def build_segment_graph(
    skeleton: np.ndarray,
    ori_z: np.ndarray,
    ori_y: np.ndarray,
    ori_x: np.ndarray,
    min_segment_length: int,
    endpoint_avg_k: int,
    radius: int,
    verbose: bool = True,
    return_seg_labels: bool = False,
) -> tuple[list[FiberSegment], np.ndarray | None]:
    """
    Split a skeleton into segments and compute their ends and end directions.

    Junction voxels (more than two 26-neighbors, as in *sever_h_junctions*
    and Huang et al. (2016)) are removed; each remaining 26-connected
    component is a segment.  Its end voxels are the voxels with exactly one
    neighbor (*end_voxels*), and the fiber direction at each end is estimated
    by *end_direction*.  Junction voxels belong to no segment, so every
    junction leaves a gap of one to a few voxels that the linking step has to
    bridge.

    :param skeleton: np.ndarray of bool, shape (D, H, W)
        skeleton after H-junction severing
    :param ori_z: np.ndarray of float, shape (D, H, W)
        z component of the unit orientation field
    :param ori_y: np.ndarray of float, shape (D, H, W)
        y component of the unit orientation field
    :param ori_x: np.ndarray of float, shape (D, H, W)
        x component of the unit orientation field
    :param min_segment_length: int, default 1
        currently has no effect: all segments are kept (only printed)
    :param endpoint_avg_k: int, default 5
        distance rank of the anchor voxel used by *end_direction*
        (its *offset*), i.e. how far inside the segment the end direction
        is measured
    :param verbose: bool, default True
        whether to print the number and lengths of the segments
    :param return_seg_labels: bool, default False
        currently has no effect: no label map is created
    :return: tuple[list[FiberSegment], None]
        the segments, with IDs 1, 2, ...; the second element is always None
    """
    t0 = time.time()
    D, H, W = skeleton.shape

    # Step 1: remove junction voxels, label connected components (see sever_h_junctions)
    NBRS = np.ones((3, 3, 3), np.uint8)
    NBRS[1, 1, 1] = 0  # the 26 neighbors, not the voxel itself
    S26 = np.ones((3, 3, 3), bool)  # 26-connectivity for labeling
    non_junc = skeleton & (convolve(skeleton.astype(np.uint8), NBRS, mode='constant') < 3)
    labels, _ = nd_label(non_junc, S26)

    # find endpoints per segment
    is_end = non_junc & (convolve(non_junc.astype(np.uint8), NBRS, mode='constant') == 1)

    # collect non-zero voxel coords & filter by length
    pts = np.argwhere(labels)
    ids, ends = labels[tuple(pts.T)], is_end[tuple(pts.T)]
    del labels, is_end, non_junc  # the volume-sized arrays end here
    order = np.argsort(ids, kind='stable')  # stable: raster order within a segment
    pts, ids, ends = pts[order], ids[order], ends[order]
    cut = np.flatnonzero(np.diff(ids)) + 1

    # build FiberSegment objects
    if return_seg_labels:
        seg_labels = np.zeros((D, H, W), dtype=np.int32)
    else:
        seg_labels = None

    segments: list[FiberSegment] = []
    if len(pts) == 0:
        return [], None
    for segment_id, (p, e) in enumerate(zip(np.split(pts, cut), np.split(ends, cut)), start=1):
        head, tail = end_voxels(p, e)
        head_ori, head_tangent = end_direction(p, head, ori_z, ori_y, ori_x, endpoint_avg_k, radius)
        tail_ori, tail_tangent = end_direction(p, tail, ori_z, ori_y, ori_x, endpoint_avg_k, radius)
        segments.append(FiberSegment(seg_id=segment_id, coords=p.astype(np.int32),
                                     head_coord=head.astype(np.float32), tail_coord=tail.astype(np.float32),
                                     head_ori=head_ori, tail_ori=tail_ori,
                                     head_tangent=head_tangent, tail_tangent=tail_tangent))

    if verbose:
        lengths = [f.length for f in segments]
        print(f"  [graph] {len(segments)} segments kept (len >= {min_segment_length}). "
              f"Length: min={min(lengths) if lengths else 0}, "
              f"mean={int(np.mean(lengths)) if lengths else 0}, "
              f"max={max(lengths) if lengths else 0}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    return segments, seg_labels
