"""
Tests for topology_optimizer.link_fibers: UnionFind, find_bridge_candidates,
bridge_cost and optimize_topology, plus two end-to-end checks of the
skeleton -> segments -> chains part of the pipeline (H-junction, X-crossing).
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import topology_optimizer.link_fibers as lf
from topology_optimizer.link_fibers import (
    BridgeCandidate, FiberChain, UnionFind, bridge_cost, find_bridge_candidates, optimize_topology,
)
from topology_optimizer.skeleton_graph import FiberSegment, build_segment_graph, sever_h_junctions

# Parameters used throughout; tests that probe a single parameter override it.
SEVER = dict(max_rung_length=10, min_trunk_length=10, perp_thresh=0.50, rung_thresh=0.60)
GRAPH = dict(min_segment_length=1, endpoint_avg_k=5, radius=2)
BRIDGE = dict(max_gap_distance=16.0, mutual_thresh=0.65, forward_thresh=-0.30,
              durable_thresh=0.50, durable_min_fraction=0.50, n_gap_samples=7)
MAX_COMMIT_COST = 35.0


def constant_field(shape, direction):
    """Orientation field (ori_z, ori_y, ori_x) with the same unit direction everywhere."""
    d = np.asarray(direction, np.float32)
    d = d / np.linalg.norm(d)
    return tuple(np.full(shape, d[i], np.float32) for i in range(3))


def zero_field(shape):
    """Orientation field that is zero (background) everywhere."""
    return tuple(np.zeros(shape, np.float32) for _ in range(3))


def h_skeleton():
    """Two parallel z-fibers at x = 5 and x = 14 (y = 5), joined at z = 20 by a rung along x."""
    sk = np.zeros((40, 20, 20), bool)
    sk[:, 5, 5] = True
    sk[:, 5, 14] = True
    sk[20, 5, 5:15] = True
    return sk, [(20, 5, x) for x in range(7, 13)]


def x_crossing():
    """
    Two straight fibers crossing at (20, 15, 15): one along z at (y, x) = (15, 15),
    one along x at (z, y) = (20, 15).  The orientation field is a tube of
    width 5 around each fiber; in the crossing it is the bisector.
    """
    shape = (40, 30, 40)
    sk = np.zeros(shape, bool)
    sk[:, 15, 15] = True
    sk[20, 15, :] = True
    oz, oy, ox = zero_field(shape)
    oz[:, 13:18, 13:18] = 1.0
    ox[18:23, 13:18, :] = 1.0
    both = (oz > 0) & (ox > 0)
    oz[both] = ox[both] = 1.0 / np.sqrt(2.0)
    return sk, (oz, oy, ox)


def line_segment(seg_id, start, stop, fixed=(10, 10)):
    """Straight FiberSegment along z from start to stop - 1 with outward end directions."""
    coords = np.zeros((stop - start, 3), np.int32)
    coords[:, 0] = np.arange(start, stop)
    coords[:, 1], coords[:, 2] = fixed
    e = np.array([1.0, 0.0, 0.0], np.float32)
    return FiberSegment(seg_id=seg_id, coords=coords,
                        head_coord=coords[0].astype(np.float32), tail_coord=coords[-1].astype(np.float32),
                        head_ori=-e, tail_ori=e.copy(), head_tangent=-e, tail_tangent=e.copy())


def aligned_candidate(id_a, end_a, id_b, end_b, gap, direction=(1.0, 0.0, 0.0), n_samples=7):
    """Perfectly aligned BridgeCandidate of length *gap*; its cost is exactly WEIGHT_GAP * gap."""
    d = np.asarray(direction, np.float32)
    d = d / np.linalg.norm(d)
    a = np.zeros(3, np.float32)
    return BridgeCandidate(id_a, id_b, end_a, end_b, a, a + gap * d, d.copy(), d.copy(),
                           float(gap), 1.0, 1.0, np.tile(d, (n_samples, 1)))


def graph(sk, field):
    # build_segment_graph returns (segments, None); change to
    # "return build_segment_graph(...)" once the second return value is removed
    segs = build_segment_graph(sk, *field, **GRAPH, verbose=False)
    return segs


def candidates(segs, field, **kw):
    return find_bridge_candidates(segs, *field, **dict(BRIDGE, **kw), verbose=False)


def link(segs, cands, max_commit_cost=MAX_COMMIT_COST):
    return optimize_topology(segs, cands, max_commit_cost, verbose=False)


def chain_sets(chains):
    return sorted(sorted(c.segment_ids) for c in chains)


def gapped_line(gap, shape=(60, 20, 20)):
    """z-line at (10, 10) from z = 2 to 15 and from 15 + gap to 34 + gap."""
    sk = np.zeros(shape, bool)
    sk[2:16, 10, 10] = True
    sk[15 + gap:35 + gap, 10, 10] = True
    return sk


# --------------------------------------------------------------------------- #
# UnionFind                                                                   #
# --------------------------------------------------------------------------- #
class TestUnionFind(unittest.TestCase):
    def test_union_and_find(self):
        uf = UnionFind(range(5))
        self.assertTrue(uf.union(0, 1))
        self.assertTrue(uf.union(3, 4))
        self.assertEqual(uf.find(0), uf.find(1))
        self.assertNotEqual(uf.find(1), uf.find(3))
        self.assertTrue(uf.union(1, 4))
        self.assertEqual(len({uf.find(i) for i in range(5)}), 2)

    def test_union_within_set_returns_false(self):
        uf = UnionFind([1, 2, 3])
        uf.union(1, 2)
        uf.union(2, 3)
        self.assertFalse(uf.union(1, 3))

    def test_arbitrary_keys_added_lazily(self):
        uf = UnionFind()
        self.assertEqual(uf.find(('a', 'head')), ('a', 'head'))
        self.assertTrue(uf.union('x', 'y'))
        self.assertEqual(uf.find('x'), uf.find('y'))

    def test_long_chain_is_compressed(self):
        uf = UnionFind(range(2000))
        for i in range(1999):
            uf.union(i, i + 1)
        root = uf.find(0)
        self.assertTrue(all(uf.find(i) == root for i in range(2000)))


# --------------------------------------------------------------------------- #
# find_bridge_candidates                                                      #
# --------------------------------------------------------------------------- #
class TestFindBridgeCandidates(unittest.TestCase):
    def test_collinear_gap(self):
        sk = gapped_line(5)
        field = constant_field(sk.shape, (1, 0, 0))
        segs = graph(sk, field)
        cands = candidates(segs, field)
        self.assertEqual(len(cands), 1)
        c = cands[0]
        self.assertEqual({(c.seg_id_a, c.end_a), (c.seg_id_b, c.end_b)}, {(1, 'tail'), (2, 'head')})
        self.assertAlmostEqual(c.gap_distance, 5.0, places=5)
        self.assertAlmostEqual(c.forward_align_a, 1.0, places=5)
        self.assertAlmostEqual(c.forward_align_b, 1.0, places=5)
        # directions are for traveling from A to B
        g = (c.coord_b - c.coord_a) / c.gap_distance
        np.testing.assert_allclose(c.ori_a, g, atol=1e-5)
        np.testing.assert_allclose(c.ori_b, g, atol=1e-5)
        self.assertEqual(c.path_ori_samples.shape, (BRIDGE['n_gap_samples'], 3))

    def test_gap_beyond_max_distance(self):
        sk = gapped_line(20)
        field = constant_field(sk.shape, (1, 0, 0))
        self.assertEqual(candidates(graph(sk, field), field), [])

    def test_forward_filter_rejects_end_pointing_away(self):
        # with a large search radius the far ends come into range, but they point away
        sk = gapped_line(5)
        field = constant_field(sk.shape, (1, 0, 0))
        cands = candidates(graph(sk, field), field, max_gap_distance=40.0)
        self.assertEqual(len(cands), 1)

    def test_mutual_filter_rejects_perpendicular_ends(self):
        sk = np.zeros((30, 30, 30), bool)
        sk[2:15, 15, 15] = True          # along z, tail at z = 14
        sk[18, 15, 17:28] = True         # along x, head at x = 17
        oz, oy, ox = zero_field(sk.shape)
        oz[:16] = 1.0
        ox[16:] = 1.0
        field = (oz, oy, ox)
        self.assertEqual(candidates(graph(sk, field), field), [])
        self.assertEqual(len(candidates(graph(sk, field), field, mutual_thresh=0.0,
                                        durable_min_fraction=0.0)), 1)

    def test_no_candidate_between_ends_of_one_segment(self):
        sk = np.zeros((20, 10, 10), bool)
        sk[5:9, 5, 5] = True
        field = constant_field(sk.shape, (1, 0, 0))
        self.assertEqual(candidates(graph(sk, field), field, forward_thresh=-1.0, mutual_thresh=0.0), [])

    def test_gap_through_background_rejected(self):
        sk = gapped_line(5)
        oz, oy, ox = zero_field(sk.shape)
        oz[sk] = 1.0                     # field only on the skeleton, zero in the gap
        field = (oz, oy, ox)
        self.assertEqual(candidates(graph(sk, field), field), [])

    def test_short_gap_skips_gap_support(self):
        sk = gapped_line(2)
        oz, oy, ox = zero_field(sk.shape)
        oz[sk] = 1.0
        field = (oz, oy, ox)
        cands = candidates(graph(sk, field), field)
        self.assertEqual(len(cands), 1)
        c = cands[0]
        np.testing.assert_allclose(c.path_ori_samples, np.stack([c.ori_a, c.ori_b]), atol=1e-6)

    def test_gap_field_across_gap_rejected(self):
        sk = gapped_line(5)
        oz, oy, ox = constant_field(sk.shape, (1, 0, 0))
        oz[16:20] = 0.0
        oy[16:20] = 1.0                  # field perpendicular to the gap
        field = (oz, oy, ox)
        self.assertEqual(candidates(graph(sk, field), field), [])

    def test_durable_min_fraction(self):
        # gap samples at z = 15.75 ... 19.25 round to 16, 16, 17, 18, 18, 19, 19;
        # supporting field only at z = 16, 17 -> 3 of 7 samples
        sk = gapped_line(5)
        oz, oy, ox = zero_field(sk.shape)
        oz[sk] = 1.0
        oz[16:18, 10, 10] = 1.0
        field = (oz, oy, ox)
        segs = graph(sk, field)
        self.assertEqual(len(candidates(segs, field, durable_min_fraction=0.4)), 1)
        self.assertEqual(len(candidates(segs, field, durable_min_fraction=0.5)), 0)

    def test_sign_of_field_irrelevant(self):
        sk = gapped_line(5)
        f = constant_field(sk.shape, (1, 0, 0))
        g = tuple(-o for o in f)
        a, b = candidates(graph(sk, f), f), candidates(graph(sk, g), g)
        self.assertEqual(len(a), len(b))
        self.assertAlmostEqual(bridge_cost(a[0]), bridge_cost(b[0]), places=5)

    def test_empty(self):
        self.assertEqual(candidates([], zero_field((5, 5, 5))), [])


# --------------------------------------------------------------------------- #
# bridge_cost                                                                 #
# --------------------------------------------------------------------------- #
class TestBridgeCost(unittest.TestCase):
    def test_aligned_bridge_costs_only_gap_length(self):
        c = aligned_candidate(1, 'tail', 2, 'head', gap=6.0)
        self.assertAlmostEqual(bridge_cost(c), lf.WEIGHT_GAP * 6.0, places=5)

    def test_monotone_in_gap_length(self):
        costs = [bridge_cost(aligned_candidate(1, 'tail', 2, 'head', gap=g)) for g in (1, 4, 9, 16)]
        self.assertEqual(costs, sorted(costs))

    def test_turn_terms(self):
        # end A turned by 30 deg in the y-z plane, field and end B along z
        th = np.deg2rad(30.0)
        c = aligned_candidate(1, 'tail', 2, 'head', gap=4.0, direction=(1, 0, 0))
        c.ori_a = np.array([np.cos(th), np.sin(th), 0.0], np.float32)
        c.forward_align_a = float(np.cos(th))
        expected = (lf.WEIGHT_GAP * 4.0 + lf.WEIGHT_ORI_CHANGE * th + lf.WEIGHT_TURN * th
                    + lf.WEIGHT_GAP_SUPPORT * max(0.0, lf.GAP_SUPPORT_MIN_COS - np.cos(th)))
        self.assertAlmostEqual(bridge_cost(c), expected, places=4)

    def test_sharp_turn_penalty(self):
        def cost(deg):
            th = np.deg2rad(deg)
            c = aligned_candidate(1, 'tail', 2, 'head', gap=1.0)
            c.forward_align_a = float(np.cos(th))
            return bridge_cost(c)
        below, above = cost(np.rad2deg(lf.SHARP_TURN) - 0.5), cost(np.rad2deg(lf.SHARP_TURN) + 0.5)
        self.assertGreater(above - below, lf.SHARP_TURN_PENALTY - 1.0)
        self.assertGreater(above, MAX_COMMIT_COST)

    def test_gap_support_term(self):
        c = aligned_candidate(1, 'tail', 2, 'head', gap=2.0)
        c.path_ori_samples = np.tile(np.array([0.0, 1.0, 0.0], np.float32), (7, 1))  # across the gap
        expected = lf.WEIGHT_GAP * 2.0 + 2 * lf.WEIGHT_GAP_SUPPORT * lf.GAP_SUPPORT_MIN_COS
        self.assertAlmostEqual(bridge_cost(c), expected, places=4)

    def test_background_samples_ignored(self):
        c = aligned_candidate(1, 'tail', 2, 'head', gap=3.0)
        c.path_ori_samples = np.zeros((7, 3), np.float32)
        self.assertAlmostEqual(bridge_cost(c), lf.WEIGHT_GAP * 3.0, places=5)

    def test_sign_of_gap_samples_irrelevant(self):
        # the docstring says m "can depend on the arbitrary sign of the predicted
        # orientation"; m is an axial mean, so it cannot
        c = aligned_candidate(1, 'tail', 2, 'head', gap=3.0)
        c.path_ori_samples = np.array([[1, 0.3, 0], [-1, -0.3, 0], [1, 0.2, 0], [-1, 0.1, 0]], np.float32)
        a = bridge_cost(c)
        c.path_ori_samples = -c.path_ori_samples
        self.assertAlmostEqual(bridge_cost(c), a, places=6)

    def test_returns_python_float(self):
        self.assertIsInstance(bridge_cost(aligned_candidate(1, 'tail', 2, 'head', gap=3.0)), float)


# --------------------------------------------------------------------------- #
# optimize_topology                                                           #
# --------------------------------------------------------------------------- #
class TestOptimizeTopology(unittest.TestCase):
    def three_segments(self):
        return [line_segment(1, 0, 10), line_segment(2, 12, 30), line_segment(3, 33, 38)]

    def test_links_collinear_segments(self):
        segs = self.three_segments()
        cands = [aligned_candidate(1, 'tail', 2, 'head', 2.0), aligned_candidate(2, 'tail', 3, 'head', 3.0)]
        chains = link(segs, cands)
        self.assertEqual(len(chains), 1)
        ch = chains[0]
        self.assertIsInstance(ch, FiberChain)
        self.assertEqual(sorted(ch.segment_ids), [1, 2, 3])
        self.assertEqual(ch.total_length, 10 + 18 + 5)
        self.assertEqual(len(ch.bridge_coords), 2)

    def test_unlinked_segments_are_own_chains(self):
        chains = link(self.three_segments(), [])
        self.assertEqual(chain_sets(chains), [[1], [2], [3]])
        self.assertTrue(all(c.bridge_coords == [] for c in chains))

    def test_chains_sorted_longest_first(self):
        chains = link(self.three_segments(), [aligned_candidate(1, 'tail', 3, 'head', 3.0)])
        lengths = [c.total_length for c in chains]
        self.assertEqual(lengths, sorted(lengths, reverse=True))
        self.assertEqual(chain_sets(chains), [[1, 3], [2]])

    def test_no_branching_cheapest_wins(self):
        segs = self.three_segments()
        cands = [aligned_candidate(1, 'tail', 2, 'head', 4.0),   # cost 1.0
                 aligned_candidate(1, 'tail', 3, 'head', 2.0)]   # cost 0.5, same end of 1
        self.assertEqual(chain_sets(link(segs, cands)), [[1, 3], [2]])

    def test_different_ends_of_one_segment_both_link(self):
        segs = self.three_segments()
        cands = [aligned_candidate(2, 'head', 1, 'tail', 2.0), aligned_candidate(2, 'tail', 3, 'head', 2.0)]
        self.assertEqual(chain_sets(link(segs, cands)), [[1, 2, 3]])

    def test_no_cycles(self):
        segs = self.three_segments()
        cands = [aligned_candidate(1, 'tail', 2, 'head', 1.0),
                 aligned_candidate(2, 'tail', 3, 'head', 2.0),
                 aligned_candidate(3, 'tail', 1, 'head', 3.0)]
        chains = link(segs, cands)
        self.assertEqual(chain_sets(chains), [[1, 2, 3]])
        self.assertEqual(len(chains[0].bridge_coords), 2)

    def test_max_commit_cost(self):
        segs = self.three_segments()
        expensive = [aligned_candidate(1, 'tail', 2, 'head', 200.0)]   # cost 50
        self.assertEqual(chain_sets(link(segs, expensive)), [[1], [2], [3]])
        self.assertEqual(chain_sets(link(segs, expensive, max_commit_cost=None)), [[1, 2], [3]])

    def test_cost_exactly_at_limit_accepted(self):
        segs = self.three_segments()
        cands = [aligned_candidate(1, 'tail', 2, 'head', 4.0)]   # cost 1.0
        self.assertEqual(chain_sets(link(segs, cands, max_commit_cost=1.0)), [[1, 2], [3]])

    def test_single_voxel_segment_links_once(self):
        segs = [line_segment(1, 0, 10), line_segment(2, 12, 13), line_segment(3, 15, 25)]
        cands = [aligned_candidate(1, 'tail', 2, 'head', 2.0), aligned_candidate(2, 'tail', 3, 'head', 2.5)]
        self.assertEqual(chain_sets(link(segs, cands)), [[1, 2], [3]])

    def test_candidate_order_irrelevant(self):
        segs = self.three_segments()
        cands = [aligned_candidate(1, 'tail', 2, 'head', 4.0), aligned_candidate(1, 'tail', 3, 'head', 2.0),
                 aligned_candidate(2, 'tail', 3, 'tail', 3.0)]
        self.assertEqual(chain_sets(link(segs, cands)), chain_sets(link(segs, cands[::-1])))

    def test_bridge_coords_stored(self):
        segs = self.three_segments()
        c = aligned_candidate(1, 'tail', 2, 'head', 2.0)
        bridges = link(segs, [c])[0].bridge_coords
        self.assertEqual(len(bridges), 1)
        np.testing.assert_array_equal(bridges[0][0], c.coord_a)
        np.testing.assert_array_equal(bridges[0][1], c.coord_b)

    def test_chain_ids_unique_from_one(self):
        chains = link(self.three_segments(), [])
        self.assertEqual(sorted(c.chain_id for c in chains), [1, 2, 3])

    def test_empty(self):
        self.assertEqual(link([], []), [])


# --------------------------------------------------------------------------- #
# skeleton -> chains                                                          #
# --------------------------------------------------------------------------- #
class TestSkeletonToChains(unittest.TestCase):
    def run_pipeline(self, sk, field, sever=True):
        if sever:
            sk = sever_h_junctions(sk, *field, **SEVER, verbose=False)
        segs = graph(sk, field)
        return segs, link(segs, candidates(segs, field))

    def test_h_junction_gives_two_fibers(self):
        sk, _ = h_skeleton()
        segs, chains = self.run_pipeline(sk, constant_field(sk.shape, (1, 0, 0)))
        self.assertEqual(len(chains), 2)
        by_id = {s.seg_id: s for s in segs}
        for ch in chains:
            xs = {int(by_id[i].coords[0, 2]) for i in ch.segment_ids}
            self.assertEqual(len(xs), 1, "a chain must stay on one fiber")

    def test_x_crossing_gives_two_straight_fibers(self):
        sk, field = x_crossing()
        segs, chains = self.run_pipeline(sk, field, sever=False)
        self.assertEqual(len(chains), 2)
        by_id = {s.seg_id: s for s in segs}
        for ch in chains:
            self.assertEqual(len(ch.segment_ids), 2)
            a, b = (by_id[i] for i in ch.segment_ids)
            self.assertAlmostEqual(abs(float(a.tail_ori @ b.tail_ori)), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
