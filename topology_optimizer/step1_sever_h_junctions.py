# TODO make sure "rules" are correct - are they new?
# TODO justify thresholds; max_rung_length Huang scales by fiber diameter?


import gc
import numpy as np
from scipy.ndimage import convolve, label as nd_label
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict

from topology_optimizer.orientation_utils import axial_mean


def classify_branch(voxels, n_junctions,
                    trunk_oris, ori_z, ori_y, ori_x,
                    rung_thresh, perp_thresh):
    """
    Decides whether a short skeleton branch is an H-junction rung.

    Classification is inspired by Huang et al. (2016) (TODO plus more rules?).
    A rung is a short bridge between two fibers; the trunks are the long
    branches that clearly belong to a fiber::

        fiber 1       fiber 2
          o             o      <- trunk
          o             o
          #ooooooooooooo#      <- rung
          o             o
          o             o

        # = junction voxels

    The rules are checked in order and the first one that applies is
    returned:

    * **Rule 1** – the branch is a single voxel touching at least one
      junction.  A single voxel has no direction, so it is removed.
    * **Rule A** – the branch connects at least two junctions and its
      end-to-end direction deviates from the mean image orientation along
      the branch, i.e. ``|dot(branch_dir, o_mean)| < rung_thresh``.
    * **Rule B** – the branch direction is close to perpendicular to the
      mean orientation of the neighboring trunks, i.e.
      ``1 - |dot(branch_dir, trunk_mean_ori)| > perp_thresh``.

    :param voxels: np.ndarray of int, shape (n, 3)
        (z, y, x) coordinates of the branch voxels; the branch direction is
        taken from the first to the last row
    :param n_junctions: int
        number of distinct junction clusters the branch touches
    :param trunk_oris: np.ndarray of float, shape (k, 3)
        mean orientations of the long neighboring branches (trunks); may be
        empty, in which case Rule B is skipped
    :param ori_z: np.ndarray of float, shape (D, H, W)
        z component of the unit orientation field
    :param ori_y: np.ndarray of float, shape (D, H, W)
        y component of the unit orientation field
    :param ori_x: np.ndarray of float, shape (D, H, W)
        x component of the unit orientation field
    :param rung_thresh: float
        threshold for Rule A on ``|cos|`` of the angle between branch and
        image orientation; 0.60 means the branch makes more than ~53 deg
        with the image orientation
    :param perp_thresh: float
        threshold for Rule B on ``1 - |cos|`` of the angle between branch
        and trunk orientation; 0.50 means the branch makes more than
        60 deg with the trunks
    :return: str or None
        ``'1'``, ``'A'`` or ``'B'`` for the rule that marks the branch as a
        rung, or None if the branch is kept
    """
    # Rule 1: a single voxel has no direction; remove it if it touches a junction
    if len(voxels) == 1:
        return '1' if n_junctions >= 1 else None

    branch_vec = (voxels[-1] - voxels[0]).astype(np.float32)
    branch_dir = branch_vec / np.linalg.norm(branch_vec)

    # Rule A: compare graph orientation along the rung candidate to image orientation
    if n_junctions >= 2:
        idx = tuple(voxels.T)
        o_mean, _ = axial_mean(np.column_stack([ori_z[idx], ori_y[idx], ori_x[idx]]))
        if o_mean is not None and abs(np.dot(branch_dir, o_mean)) < rung_thresh:
            return 'A'

    # Rule B: compare rung candidate direction to fiber direction (long neighbors, trunks)
    # TODO point out differences to Huang
    if len(trunk_oris) > 0:
        trunk_mean_ori, _ = axial_mean(trunk_oris)
        if 1.0 - abs(np.dot(branch_dir, trunk_mean_ori)) > perp_thresh:
            return 'B'
    return None

def sever_h_junctions(
    skeleton: np.ndarray,
    ori_z: np.ndarray,
    ori_y: np.ndarray,
    ori_x: np.ndarray,
    max_rung_length: int,
    min_neighbor_length: int = 30,
    perp_thresh: float = 0.50,
    rung_thresh: float = 0.60,
    verbose: bool = True,
) -> np.ndarray:
    """
    Removes H-junction rungs from a binary skeleton, inspired by Huang et al. (2016).

    The skeleton is split into branches by removing all junction voxels
    (voxels with at least three 26-neighbors).  Junction voxels are grouped
    into junction clusters, and two branches are neighbors if they touch the
    same cluster.  Every branch with at most *max_rung_length* voxels is then
    classified by *classify_branch*, with its neighbors of at least
    *min_neighbor_length* voxels serving as trunks.  Branches classified as
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
        branches with more voxels than this are never rungs
    :param min_neighbor_length: int, default 30
        minimum number of voxels for a neighboring branch to count as a
        trunk in Rule B; avoids comparing short branches to each other
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
        copy of the skeleton with the rung branches removed
    """

    D, H, W = skeleton.shape
    NBRS = np.ones((3, 3, 3), np.uint8)
    NBRS[1, 1, 1] = 0  # the 26 neighbours, not the voxel itself
    S26 = np.ones((3, 3, 3), bool)  # 26-connectivity for labelling

    # Step 1: label all branch segments (non-junction components)
    n_nbrs_per_voxel = convolve(skeleton.astype(np.uint8), NBRS, mode='constant')
    junction_mask = skeleton & (n_nbrs_per_voxel >= 3)
    labeled_branches, n_branches = nd_label(skeleton & ~junction_mask, S26)  # branches = skeleton minus junctions
    if n_branches == 0:
        return skeleton.copy()

    # Step 2: compute per-branch orientations & collect coords
    branch_z, branch_y, branch_x = np.where(labeled_branches > 0)
    branch_ids = labeled_branches[branch_z, branch_y, branch_x]
    branch_sizes = np.bincount(branch_ids, minlength=n_branches + 1)

    # Axial mean orientation
    v = np.column_stack([ori_z[branch_z, branch_y, branch_x], ori_y[branch_z, branch_y, branch_x], ori_x[branch_z, branch_y, branch_x]]).astype(np.float64)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    T = np.zeros((n_branches + 1, 3, 3))
    for i in range(3):
        for j in range(3):
            T[:, i, j] = np.bincount(branch_ids, weights=v[:, i] * v[:, j], minlength=n_branches + 1)
    _, evecs = np.linalg.eigh(T)  # per branch; eigenvalues in ascending order
    branch_mean_ori = evecs[:, :, -1]  # unit vector per branch, arbitrary sign; row 0 unused

    # Step 3: collect start/end coords per branch for rung_dir
    sort_order = np.argsort(branch_ids, kind='stable')
    branch_z, branch_y, branch_x, branch_ids = branch_z[sort_order], branch_y[sort_order], branch_x[sort_order], branch_ids[sort_order]
    unique_bids, split_idx = np.unique(branch_ids, return_index=True)
    coord_z_splits = np.split(branch_z, split_idx[1:])
    coord_y_splits = np.split(branch_y, split_idx[1:])
    coord_x_splits = np.split(branch_x, split_idx[1:])


    # Step 4: build branch adjacency THROUGH junction clusters
    # Branches are separated by junction voxels; branches that touch the
    # same junction cluster are adjacent neighbors.
    jz, jy, jx = np.where(junction_mask)
    del junction_mask
    gc.collect()

    branch_to_juncs = defaultdict(set)
    junc_to_branches = defaultdict(set)

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

        # record each junction-branch-pair
        for k in range(len(junc_coords)):
            cz, cy, cx = int(jz[k]), int(jy[k]), int(jx[k])
            jid = int(junc_labels[k]) + 1
            z0, z1 = max(0, cz - 1), min(D, cz + 2)
            y0, y1 = max(0, cy - 1), min(H, cy + 2)
            x0, x1 = max(0, cx - 1), min(W, cx + 2)
            touching_bids = np.unique(labeled_branches[z0:z1, y0:y1, x0:x1])
            for branch_id in touching_bids:
                if branch_id > 0:
                    branch_to_juncs[branch_id].add(jid)
                    junc_to_branches[jid].add(branch_id)

    # We are completely done with labeled_branches (free 8.4 GB!)
    del labeled_branches
    gc.collect()

    # make each branch-pair with junction between each other neighbors
    adjacency: list[set] = [set() for _ in range(n_branches + 1)]
    for jid, br_set in junc_to_branches.items():
        br_list = list(br_set)
        for i in range(len(br_list)):
            for j in range(len(br_list)):
                if i != j:
                    adjacency[br_list[i]].add(br_list[j])

    # Step 5: classify each short branch - severing decision
    sever_mask    = np.zeros(n_branches + 1, dtype=bool)

    for idx, branch_id in enumerate(unique_bids):
        branch_id = int(branch_id)
        if branch_sizes[branch_id] > max_rung_length:
            continue
        voxels = np.column_stack([coord_z_splits[idx], coord_y_splits[idx], coord_x_splits[idx]])
        long_neighbors = [nb for nb in adjacency[branch_id] if branch_sizes[nb] >= min_neighbor_length]
        rule = classify_branch(voxels, len(branch_to_juncs.get(branch_id, ())),
                               branch_mean_ori[long_neighbors], ori_z, ori_y, ori_x,
                               rung_thresh, perp_thresh)
        if rule is not None:
            sever_mask[branch_id] = rule

    # Step 6: remove severed branches (coordinate-based, 0 GB overhead)
    clean_skel = skeleton.copy()
    severed_bids_set = set(np.where(sever_mask)[0])
    for idx, branch_id in enumerate(unique_bids):
        if int(branch_id) in severed_bids_set:
            cz_s = coord_z_splits[idx]
            cy_s = coord_y_splits[idx]
            cx_s = coord_x_splits[idx]
            clean_skel[cz_s, cy_s, cx_s] = False

    if verbose:
        print(f"  [H-sever] Severed {int(sever_mask.sum())} H-junction rungs "
              f"(short<={max_rung_length}, neighbor>={min_neighbor_length}, "
              f"perp>{perp_thresh}).", flush=True)
        removed = int(skeleton.sum()) - int(clean_skel.sum())
        print(f"  [H-sever] Skeleton: {int(skeleton.sum())} -> "
              f"{int(clean_skel.sum())} ({removed} voxels removed).", flush=True)

    return clean_skel
