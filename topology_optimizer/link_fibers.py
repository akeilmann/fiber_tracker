# TODO similar to but not quite Griesser?
# TODO get rid of greedy algo
# TODO explain weights


import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass
from collections import defaultdict

from sympy.sets import sets

from .skeleton_graph import FiberSegment
from topology_optimizer.orientation_utils import axial_mean



WEIGHT_GAP = 0.25                 # per voxel of gap length
WEIGHT_ORI_CHANGE = 5.0           # per radian between the two end directions
WEIGHT_TURN = 8.0                 # per radian of turning from end A into the gap and from the gap into end B
SHARP_TURN = np.deg2rad(55)       # a turn above this angle ...
SHARP_TURN_PENALTY = 50.0         # ... adds this; with max_commit_cost = 35 this rejects the bridge
GAP_SUPPORT_MIN_COS = 0.70        # end directions should agree with the field in the gap to |cos| >= this
WEIGHT_GAP_SUPPORT = 50.0         # per unit of |cos| missing (5.0 * 10 in the original)


@dataclass
class BridgeCandidate:
    """
    A possible straight bridge from an end of segment A to an end of segment B.

    All directions are expressed for traveling from A to B.

    :param seg_id_a: int
        seg_id of segment A
    :param seg_id_b: int
        seg_id of segment B
    :param end_a: str
        end of segment A that is bridged, 'head' or 'tail'
    :param end_b: str
        end of segment B that is bridged, 'head' or 'tail'
    :param coord_a: np.ndarray of float, shape (3,)
        (z, y, x) coordinates of the end voxel of A
    :param coord_b: np.ndarray of float, shape (3,)
        (z, y, x) coordinates of the end voxel of B
    :param ori_a: np.ndarray of float, shape (3,)
        fiber orientation at the end of A, pointing out of A into the gap
    :param ori_b: np.ndarray of float, shape (3,)
        fiber orientation at the end of B, pointing from the gap into B
    :param gap_distance: float
        Euclidean distance between the two end voxels in voxels
    :param forward_align_a: float
        cosine of the angle between ori_a and the gap direction A -> B
    :param forward_align_b: float
        cosine of the angle between the gap direction A -> B and ori_b
    :param path_ori_samples: np.ndarray of float, shape (n, 3)
        orientation field sampled along the gap (unit or zero vectors); for
        gaps of at most 2 voxels, ori_a and ori_b instead
    """
    seg_id_a: int
    seg_id_b: int
    end_a: str                     # 'head' or 'tail'
    end_b: str
    coord_a: np.ndarray            # (3,) end voxels
    coord_b: np.ndarray
    ori_a: np.ndarray              # (3,) fiber direction at A, leaving A into the gap
    ori_b: np.ndarray              # (3,) fiber direction at B, entering B from the gap
    gap_distance: float
    forward_align_a: float         # cos(ori_a, gap direction A -> B)
    forward_align_b: float         # cos(gap direction A -> B, ori_b)
    path_ori_samples: np.ndarray   # (n, 3) orientation field sampled along the gap

class UnionFind:
    """Union-find over arbitrary hashable keys, with path compression and union by rank."""
    def __init__(self, items=()):
        self.parent, self.rank = {}, {}
        for x in items:
            self.add(x)

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # already connected (adding the edge would form a cycle)
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

@dataclass
class FiberChain:
    """
    Segments linked into one fiber.

    :param chain_id: int
        chain ID, starting at 1
    :param segment_ids: list[int]
        seg_ids of the segment in the chain (not in path order)
    :param bridge_coords: list[tuple[np.ndarray, np.ndarray]]
        end voxel coordinates (coord_a, coord_b) of every accepted bridge
    :param total_length: int
        sum of the segment lengths in voxels; gaps are not counted
    """
    chain_id: int
    segment_ids: list[int]          # ordered segment IDs
    bridge_coords: list[tuple]       # list of (coord_a, coord_b) for each bridge
    total_length: int                # sum of segment lengths (no gap voxels)


def find_bridge_candidates(
        segments: list[FiberSegment],
        ori_z: np.ndarray,
        ori_y: np.ndarray,
        ori_x: np.ndarray,
        max_gap_distance: float,
        mutual_thresh: float,
        forward_thresh: float,
        durable_thresh: float,
        durable_min_fraction: float,
        n_gap_samples: int,
        verbose: bool = True,
) -> list[BridgeCandidate]:
    """
    Find all pairs of segment ends that could be joined by a straight bridge.

    As in Griesser et al. (2023), all pairs of ends of different segments
    within *max_gap_distance* are considered and judged by the dot products
    between the two end orientations and between each end orientation and
    the gap direction.  Here these act as filters; the choice among the
    remaining candidates is left to optimize_topology.  A pair is kept if

    1. both ends describe nearly the same line,
       ``|ori_a . ori_b| >= mutual_thresh``;
    2. neither end points away from the gap, ``cos >= forward_thresh`` for
       both turning angles;
    3. for gaps longer than 2 voxels, the gap runs through predicted fiber
       material along the gap: at least *durable_min_fraction* of the
       *n_gap_samples* field samples between 15 % and 85 % of the gap satisfy
       ``|sample . gap_dir| >= durable_thresh``.  Background samples are zero
       and never satisfy this.  Gaps of at most 2 voxels, typically the holes
       left by removed junction voxels, skip this test.

    The end orientations must point outward, away from their segment, as
    produced by build_segment_graph.

    :param segments: list[FiberSegment]
        segments from build_segment_graph
    :param ori_z: np.ndarray of float, shape (D, H, W)
        z component of the unit orientation field
    :param ori_y: np.ndarray of float, shape (D, H, W)
        y component of the unit orientation field
    :param ori_x: np.ndarray of float, shape (D, H, W)
        x component of the unit orientation field
    :param max_gap_distance: float, default 16.0
        maximum Euclidean distance between two ends in voxels
    :param mutual_thresh: float, default 0.65
        minimum ``|cos|`` between the two end orientations
    :param forward_thresh: float, default -0.30
        minimum cosine between each end orientation and the gap direction;
        -0.30 allows an end to point up to ~107 deg away from the gap
    :param durable_thresh: float, default 0.50
        minimum ``|cos|`` between a field sample and the gap direction for
        the sample to support the gap
    :param durable_min_fraction: float, default 0.50
        minimum fraction of supporting samples
    :param n_gap_samples: int, default 7
        number of field samples along the gap
    :param verbose: bool, default True
        whether to print the number of candidates
    :return: list[BridgeCandidate]
        the candidates, each pair of ends at most once, in no particular order
    """
    if not segments:
        return []

    # Build array of all segment endpoints (2 per segment)
    ends = [(f.seg_id, side, getattr(f, f'{side}_coord'), getattr(f, f'{side}_ori'))
            for f in segments for side in ('head', 'tail')]
    coords = np.array([e[2] for e in ends], np.float32)

    # Candidate search via KD-tree
    candidates: list[BridgeCandidate] = []
    for i, j in cKDTree(coords).query_pairs(r=max_gap_distance):
        (id_a, end_a, p_a, o_a), (id_b, end_b, p_b, o_b) = ends[i], ends[j]
        if id_a == id_b:  # the two ends of one segment
            continue
        gap = coords[j] - coords[i]
        dist = float(np.linalg.norm(gap))
        g = gap / dist  # gap direction A -> B
        fwd_a, fwd_b = float(o_a @ g), float(-o_b @ g)
        # 1. both ends describe (nearly) the same line; 2. neither end points away from the gap
        if abs(o_a @ o_b) < mutual_thresh or min(fwd_a, fwd_b) < forward_thresh:
            continue
        # 3. the predicted orientation inside the gap runs along the gap
        if dist > 2.0:
            s = np.round(coords[i] + np.linspace(0.15, 0.85, n_gap_samples)[:, None] * gap).astype(int)
            samples = np.column_stack([o[tuple(s.T)] for o in (ori_z, ori_y, ori_x)]).astype(np.float32)
            n = np.linalg.norm(samples, axis=1, keepdims=True)
            samples /= np.where(n < 1e-8, 1.0, n)
            if np.mean(np.abs(samples @ g) >= durable_thresh) < durable_min_fraction:
                continue
        else:
            samples = np.stack([o_a, -o_b])
        candidates.append(BridgeCandidate(id_a, id_b, end_a, end_b, coords[i], coords[j], o_a, -o_b,
                                          dist, fwd_a, fwd_b, samples))

    if verbose:
        print(f"  [bridge] Found {len(candidates)} admissible bridge candidates "
              f"(mutual>={mutual_thresh:.2f}, fwd>={forward_thresh:.2f}, "
              f"durable>={durable_min_fraction:.2f}).", flush=True)

    return candidates


def bridge_cost(c):
    """
    Compute the cost of a candidate bridge; lower is better.

    The cost adds up, with the weights defined at the top of the module:

    * the gap length, ``WEIGHT_GAP * gap_distance``;
    * the angle between the two end orientations,
      ``WEIGHT_ORI_CHANGE * arccos(|ori_a . ori_b|)``;
    * the turning angles from end A into the gap and from the gap into
      end B, ``WEIGHT_TURN * (theta_a + theta_b)``, plus SHARP_TURN_PENALTY
      if either exceeds SHARP_TURN;
    * the gap support: how far the end orientations miss the mean
      orientation m of the samples along the gap,
      ``WEIGHT_GAP_SUPPORT * sum(max(0, GAP_SUPPORT_MIN_COS - |ori . m|))``.

    The angle between the end orientations is at most theta_a + theta_b, so
    the second term partly repeats the third.  m is a signed mean of the
    field samples and can therefore depend on the arbitrary sign of the
    predicted orientation.

    :param c: BridgeCandidate
        candidate bridge from find_bridge_candidates
    :return: float
        cost of the bridge
    """
    theta_a, theta_b = np.arccos(np.clip([c.forward_align_a, c.forward_align_b], -1.0, 1.0))
    cost = (WEIGHT_GAP * c.gap_distance
            + WEIGHT_ORI_CHANGE * np.arccos(np.clip(abs(float(c.ori_a @ c.ori_b)), 0.0, 1.0))
            + WEIGHT_TURN * (theta_a + theta_b)
            + (SHARP_TURN_PENALTY if max(theta_a, theta_b) > SHARP_TURN else 0.0))
    m, _ = axial_mean(c.path_ori_samples)  # sign-free; zero (background) samples are ignored
    if m is not None:
        cost += WEIGHT_GAP_SUPPORT * (max(0.0, GAP_SUPPORT_MIN_COS - abs(float(c.ori_a @ m)))
                                      + max(0.0, GAP_SUPPORT_MIN_COS - abs(float(c.ori_b @ m))))
    return float(cost)


def optimize_topology(segments, candidates, max_commit_cost, verbose=True):
    """
    Link segments into fibers by greedily accepting the cheapest bridges.

    The candidates are sorted by bridge_cost and accepted cheapest first,
    the greedy strategy of Griesser et al. (2023); processing sorted
    candidates with a union-find structure.
    A candidate is skipped if

    * its cost exceeds *max_commit_cost*,
    * one of its two ends is already linked (fibers do not branch), or
    * its segments are already in the same chain (no cycles).

    A single-voxel segment has only one end and can therefore be linked
    only once.  Each resulting group of linked segments is one chain;
    unlinked segments form chains of their own.

    :param segments: list[FiberSegment]
        segments from build_segment_graph
    :param candidates: list[BridgeCandidate]
        candidate bridges from find_bridge_candidates, in any order
    :param max_commit_cost: float or None, default 35.0
        bridges with a higher cost are never accepted; None accepts all
    :param verbose: bool, default True
        whether to print the number and lengths of the chains
    :return: list[FiberChain]
        the chains, longest first
    """
    if not segments:
        return []
    index = {f.seg_id: i for i, f in enumerate(segments)}
    # a single-voxel segment has only one end ('head'): it can be linked once
    end_key = lambda fid, end: (fid, 'head' if segments[index[fid]].length == 1 else end)

    uf, used, accepted = UnionFind(range(len(segments))), set(), []
    for cost, c in sorted(((bridge_cost(c), c) for c in candidates), key=lambda t: t[0]):
        if max_commit_cost is not None and cost > max_commit_cost:
            break
        ends = end_key(c.seg_id_a, c.end_a), end_key(c.seg_id_b, c.end_b)
        if ends[0] in used or ends[1] in used:
            continue
        if not uf.union(index[c.seg_id_a], index[c.seg_id_b]):
            continue
        used.update(ends)
        accepted.append(c)

    # Build chains from union-find groups
    groups, bridges = defaultdict(list), defaultdict(list)
    for i in range(len(segments)):
        groups[uf.find(i)].append(i)
    for c in accepted:
        bridges[uf.find(index[c.seg_id_a])].append((c.coord_a, c.coord_b))
    chains = [FiberChain(cid, [segments[i].seg_id for i in members], bridges[root],
                         sum(segments[i].length for i in members))
              for cid, (root, members) in enumerate(groups.items(), start=1)]
    chains.sort(key=lambda ch: ch.total_length, reverse=True)

    if verbose:
        lengths = [c.total_length for c in chains]
        print(f"  [optimizer] Resolved into {len(chains)} fiber chains.", flush=True)
        if lengths:
            print(f"             Length: min={min(lengths)}, "
                  f"mean={np.mean(lengths):.0f}, max={max(lengths)}", flush=True)

    return chains

