"""
Tests for topology_optimizer.skeleton_graph: classify_segment,
sever_h_junctions, end_voxels, end_direction, build_segment_graph and
FiberSegment.

Tests marked expectedFailure document a mismatch between code and docstring;
they start passing (and are then reported as "unexpected success") once the
mismatch is resolved.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from topology_optimizer.skeleton_graph import (
    FiberSegment, build_segment_graph, classify_segment, end_direction, end_voxels,
    sever_h_junctions,
)


# Parameters used throughout; tests that probe a single parameter override it.
SEVER = dict(max_rung_length=10, min_trunk_length=10, perp_thresh=0.50, rung_thresh=0.60)
GRAPH = dict(min_segment_length=1, endpoint_avg_k=5, radius=2)


def constant_field(shape, direction):
    """Orientation field (ori_z, ori_y, ori_x) with the same unit direction everywhere."""
    d = np.asarray(direction, np.float32)
    d = d / np.linalg.norm(d)
    return tuple(np.full(shape, d[i], np.float32) for i in range(3))


def zero_field(shape):
    """Orientation field that is zero (background) everywhere."""
    return tuple(np.zeros(shape, np.float32) for _ in range(3))


def h_skeleton():
    """
    Two parallel z-fibers at x = 5 and x = 14 (y = 5), joined at z = 20 by a
    straight rung along x.  Returns the skeleton and the rung voxels that are
    not junction voxels (x = 7 ... 12).
    """
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
    Returns skeleton, (ori_z, ori_y, ori_x).
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


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def assert_vec(tc, a, b, tol=1e-5):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


# --------------------------------------------------------------------------- #
# FiberSegment                                                                #
# --------------------------------------------------------------------------- #
class TestFiberSegment(unittest.TestCase):
    def test_length_is_number_of_voxels(self):
        s = FiberSegment(seg_id=1, coords=np.zeros((7, 3), np.int32))
        self.assertEqual(s.length, 7)

    def test_default_vectors_are_independent(self):
        a = FiberSegment(seg_id=1, coords=np.zeros((1, 3)))
        b = FiberSegment(seg_id=2, coords=np.zeros((1, 3)))
        a.head_ori[0] = 1.0
        self.assertEqual(b.head_ori[0], 0.0)


# --------------------------------------------------------------------------- #
# classify_segment                                                            #
# --------------------------------------------------------------------------- #
class TestClassifySegment(unittest.TestCase):
    shape = (10, 10, 10)
    along_x = np.array([[5, 5, x] for x in range(2, 7)])
    along_z = np.array([[z, 5, 5] for z in range(2, 7)])
    no_trunks = np.zeros((0, 3))

    def classify(self, voxels, n_junctions, trunks, field, rung=0.60, perp=0.50):
        return classify_segment(voxels, n_junctions, trunks, *field, rung, perp)

    # Rule 1
    def test_rule1_single_voxel_at_junction(self):
        f = constant_field(self.shape, (1, 0, 0))
        self.assertEqual(self.classify(np.array([[5, 5, 5]]), 1, self.no_trunks, f), '1')

    def test_rule1_isolated_single_voxel_kept(self):
        f = constant_field(self.shape, (1, 0, 0))
        self.assertIsNone(self.classify(np.array([[5, 5, 5]]), 0, self.no_trunks, f))

    # Rule A
    def test_rule_a_segment_across_field(self):
        f = constant_field(self.shape, (1, 0, 0))
        self.assertEqual(self.classify(self.along_x, 2, self.no_trunks, f), 'A')

    def test_rule_a_is_sign_invariant(self):
        f = constant_field(self.shape, (-1, 0, 0))
        self.assertEqual(self.classify(self.along_x, 2, self.no_trunks, f), 'A')

    def test_rule_a_needs_two_junctions(self):
        f = constant_field(self.shape, (1, 0, 0))
        self.assertIsNone(self.classify(self.along_x, 1, self.no_trunks, f))

    def test_rule_a_segment_along_field_kept(self):
        f = constant_field(self.shape, (1, 0, 0))
        self.assertIsNone(self.classify(self.along_z, 2, self.no_trunks, f))

    def test_rule_a_skipped_without_orientation(self):
        self.assertIsNone(self.classify(self.along_x, 2, self.no_trunks, zero_field(self.shape)))

    def test_rule_a_threshold(self):
        # segment at 45 deg to the field: |cos| = 0.707
        f = constant_field(self.shape, (1, 0, 1))
        self.assertIsNone(self.classify(self.along_x, 2, self.no_trunks, f, rung=0.60))
        self.assertEqual(self.classify(self.along_x, 2, self.no_trunks, f, rung=0.80), 'A')

    # Rule B
    def test_rule_b_segment_perpendicular_to_trunks(self):
        f = constant_field(self.shape, (0, 0, 1))      # field agrees with segment: no Rule A
        trunks = np.array([[1.0, 0, 0], [-1.0, 0, 0]])  # signs are arbitrary
        self.assertEqual(self.classify(self.along_x, 2, trunks, f), 'B')

    def test_rule_b_segment_parallel_to_trunks_kept(self):
        f = constant_field(self.shape, (1, 0, 0))
        self.assertIsNone(self.classify(self.along_z, 1, np.array([[1.0, 0, 0]]), f))

    def test_rule_b_threshold(self):
        # 1 - |cos 45 deg| = 0.29
        f = constant_field(self.shape, (0, 0, 1))
        trunks = np.array([unit([1, 0, 1])])
        self.assertIsNone(self.classify(self.along_x, 1, trunks, f, perp=0.50))
        self.assertEqual(self.classify(self.along_x, 1, trunks, f, perp=0.20), 'B')

    def test_rule_a_takes_precedence_over_rule_b(self):
        f = constant_field(self.shape, (1, 0, 0))
        self.assertEqual(self.classify(self.along_x, 2, np.array([[1.0, 0, 0]]), f), 'A')


# --------------------------------------------------------------------------- #
# sever_h_junctions                                                           #
# --------------------------------------------------------------------------- #
class TestSeverHJunctions(unittest.TestCase):
    def test_removes_exactly_the_rung(self):
        sk, rung = h_skeleton()
        out = sever_h_junctions(sk, *constant_field(sk.shape, (1, 0, 0)), **SEVER, verbose=False)
        removed = {tuple(p) for p in np.argwhere(sk & ~out)}
        self.assertEqual(removed, set(rung))
        self.assertFalse(np.any(out & ~sk), "severing must not add voxels")

    def test_returns_copy(self):
        sk, _ = h_skeleton()
        before = sk.copy()
        out = sever_h_junctions(sk, *constant_field(sk.shape, (1, 0, 0)), **SEVER, verbose=False)
        self.assertIsNot(out, sk)
        np.testing.assert_array_equal(sk, before)

    def test_rung_longer_than_max_rung_length_kept(self):
        sk, _ = h_skeleton()  # rung has 6 non-junction voxels
        params = dict(SEVER, max_rung_length=5)
        out = sever_h_junctions(sk, *constant_field(sk.shape, (1, 0, 0)), **params, verbose=False)
        np.testing.assert_array_equal(out, sk)

    def test_rung_along_field_kept(self):
        # field along the rung (x) and trunks too short for Rule B
        sk, _ = h_skeleton()
        params = dict(SEVER, min_trunk_length=100)
        out = sever_h_junctions(sk, *constant_field(sk.shape, (0, 0, 1)), **params, verbose=False)
        np.testing.assert_array_equal(out, sk)

    def test_rule_b_alone_severs_rung(self):
        # field along x on the rung (Rule A silent), along z on the trunks (Rule B fires)
        sk, rung = h_skeleton()
        oz, oy, ox = constant_field(sk.shape, (1, 0, 0))
        oz[20, 5, 6:14] = 0.0
        ox[20, 5, 6:14] = 1.0
        out = sever_h_junctions(sk, oz, oy, ox, **SEVER, verbose=False)
        self.assertEqual({tuple(p) for p in np.argwhere(sk & ~out)}, set(rung))

    def test_straight_fiber_unchanged(self):
        sk = np.zeros((30, 10, 10), bool)
        sk[:, 5, 5] = True
        out = sever_h_junctions(sk, *constant_field(sk.shape, (1, 0, 0)), **SEVER, verbose=False)
        np.testing.assert_array_equal(out, sk)

    def test_empty_skeleton(self):
        sk = np.zeros((5, 5, 5), bool)
        out = sever_h_junctions(sk, *zero_field(sk.shape), **SEVER, verbose=False)
        self.assertFalse(out.any())
        self.assertIsNot(out, sk)

    def test_skeleton_touching_volume_border(self):
        # H lying in the faces y = 0, x = 0 and x = W - 1 of the volume
        sk = np.zeros((20, 6, 8), bool)
        sk[:, 0, 0] = True
        sk[:, 0, 7] = True
        sk[10, 0, :] = True
        out = sever_h_junctions(sk, *constant_field(sk.shape, (1, 0, 0)), **SEVER, verbose=False)
        self.assertTrue(out[:, 0, 0].all() and out[:, 0, 7].all())
        self.assertFalse(out[10, 0, 2:6].any())


# --------------------------------------------------------------------------- #
# end_voxels                                                                  #
# --------------------------------------------------------------------------- #
class TestEndVoxels(unittest.TestCase):
    def test_path(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        h, t = end_voxels(coords, np.array([True, False, True]))
        self.assertEqual({tuple(h), tuple(t)}, {(0, 0, 0), (2, 0, 0)})

    def test_single_voxel(self):
        h, t = end_voxels(np.array([[3, 4, 5]]), np.array([False]))
        assert_vec(self, h, [3, 4, 5])
        assert_vec(self, t, [3, 4, 5])

    def test_loop_returns_first_and_farthest(self):
        coords = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 0], [0, 1, -1]])
        h, t = end_voxels(coords, np.zeros(4, bool))
        assert_vec(self, h, [0, 0, 0])
        assert_vec(self, t, [0, 2, 0])


# --------------------------------------------------------------------------- #
# end_direction                                                               #
# --------------------------------------------------------------------------- #
class TestEndDirection(unittest.TestCase):
    shape = (30, 10, 10)
    coords = np.array([[z, 5, 5] for z in range(5, 25)])

    def test_points_outward_at_both_ends(self):
        f = constant_field(self.shape, (1, 0, 0))
        ori_t, tan_t = end_direction(self.coords, self.coords[-1], *f, 5, 2)
        ori_h, tan_h = end_direction(self.coords, self.coords[0], *f, 5, 2)
        assert_vec(self, ori_t, [1, 0, 0])
        assert_vec(self, tan_t, [1, 0, 0])
        assert_vec(self, ori_h, [-1, 0, 0])
        assert_vec(self, tan_h, [-1, 0, 0])

    def test_sign_of_field_irrelevant(self):
        f = constant_field(self.shape, (-1, 0, 0))
        ori, _ = end_direction(self.coords, self.coords[-1], *f, 5, 2)
        assert_vec(self, ori, [1, 0, 0])

    def test_orientation_from_field_tangent_from_geometry(self):
        f = constant_field(self.shape, (1, 1, 0))
        ori, tan = end_direction(self.coords, self.coords[-1], *f, 5, 2)
        assert_vec(self, ori, unit([1, 1, 0]))
        assert_vec(self, tan, [1, 0, 0])

    def test_returns_unit_float32(self):
        f = constant_field(self.shape, (1, 2, 3))
        for v in end_direction(self.coords, self.coords[-1], *f, 5, 2):
            self.assertEqual(v.dtype, np.float32)
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_fallback_to_tangent_without_field(self):
        ori, tan = end_direction(self.coords, self.coords[-1], *zero_field(self.shape), 5, 2)
        assert_vec(self, ori, [1, 0, 0])
        assert_vec(self, tan, [1, 0, 0])

    def test_offset_larger_than_segment(self):
        short = self.coords[:3]
        ori, tan = end_direction(short, short[-1], *constant_field(self.shape, (1, 0, 0)), 50, 2)
        assert_vec(self, ori, [1, 0, 0])
        assert_vec(self, tan, [1, 0, 0])

    def test_single_voxel_uses_field(self):
        one = self.coords[:1]
        ori, tan = end_direction(one, one[0], *constant_field(self.shape, (0, 1, 0)), 5, 2)
        self.assertAlmostEqual(abs(float(ori[1])), 1.0, places=5)
        assert_vec(self, tan, ori)

    def test_single_voxel_without_field(self):
        one = self.coords[:1]
        ori, tan = end_direction(one, one[0], *zero_field(self.shape), 5, 2)
        assert_vec(self, ori, [1, 0, 0])
        assert_vec(self, tan, [1, 0, 0])

    def test_window_clipped_at_volume_border(self):
        coords = np.array([[z, 0, 0] for z in range(0, 10)])
        ori, _ = end_direction(coords, coords[0], *constant_field(self.shape, (1, 0, 0)), 5, 3)
        assert_vec(self, ori, [-1, 0, 0])

    def test_local_field_measured_inside_segment(self):
        # wrong orientation right at the end, correct one around the anchor
        oz, oy, ox = constant_field(self.shape, (1, 0, 0))
        oz[23:, ...] = 0.0
        ox[23:, ...] = 1.0
        ori, _ = end_direction(self.coords, self.coords[-1], oz, oy, ox, 5, 2)
        assert_vec(self, ori, [1, 0, 0])


# --------------------------------------------------------------------------- #
# build_segment_graph                                                         #
# --------------------------------------------------------------------------- #
class TestBuildSegmentGraph(unittest.TestCase):
    def build(self, sk, field, **kw):
        return build_segment_graph(sk, *field, **dict(GRAPH, **kw), verbose=False)

    def test_straight_line(self):
        sk = np.zeros((30, 20, 20), bool)
        sk[5:25, 10, 10] = True
        segs = self.build(sk, constant_field(sk.shape, (1, 0, 0)))
        self.assertEqual(len(segs), 1)
        s = segs[0]
        self.assertEqual((s.seg_id, s.length), (1, 20))
        assert_vec(self, s.head_coord, [5, 10, 10])
        assert_vec(self, s.tail_coord, [24, 10, 10])
        assert_vec(self, s.head_ori, [-1, 0, 0])
        assert_vec(self, s.tail_ori, [1, 0, 0])
        assert_vec(self, s.head_tangent, [-1, 0, 0])
        assert_vec(self, s.tail_tangent, [1, 0, 0])
        self.assertEqual(s.coords.dtype, np.int32)

    def test_crossing_splits_into_four_segments(self):
        sk, field = x_crossing()
        segs = self.build(sk, field)
        self.assertEqual(len(segs), 4)
        self.assertEqual([s.seg_id for s in segs], [1, 2, 3, 4])
        # junction voxels belong to no segment
        covered = np.zeros_like(sk)
        for s in segs:
            covered[tuple(s.coords.T)] = True
        self.assertFalse(covered[20, 15, 15])
        self.assertFalse(np.any(covered & ~sk))
        # every segment is straight: both end directions are antiparallel
        for s in segs:
            self.assertAlmostEqual(float(s.head_ori @ s.tail_ori), -1.0, places=5)

    def test_segments_are_disjoint_and_ordered_raster(self):
        sk, field = x_crossing()
        segs = self.build(sk, field)
        seen = set()
        for s in segs:
            pts = {tuple(p) for p in s.coords}
            self.assertFalse(pts & seen)
            seen |= pts
            keys = [tuple(p) for p in s.coords]
            self.assertEqual(keys, sorted(keys))

    def test_single_voxel_segment(self):
        sk = np.zeros((10, 10, 10), bool)
        sk[5, 5, 5] = True
        segs = self.build(sk, constant_field(sk.shape, (1, 0, 0)))
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].length, 1)
        assert_vec(self, segs[0].head_coord, segs[0].tail_coord)

    def test_closed_loop(self):
        ring = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (4, 3), (4, 2), (4, 1), (3, 0), (2, 0), (1, 0)]
        sk = np.zeros((5, 8, 8), bool)
        for y, x in ring:
            sk[2, y + 1, x + 1] = True
        segs = self.build(sk, constant_field(sk.shape, (0, 1, 0)))
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].length, len(ring))
        self.assertGreater(np.linalg.norm(segs[0].head_coord - segs[0].tail_coord), 3.0)

    def test_empty(self):
        segs = self.build(np.zeros((5, 5, 5), bool), zero_field((5, 5, 5)))
        self.assertEqual(segs, [])

    def test_min_segment_length_has_no_effect(self):
        # documented: all segments are kept
        sk = np.zeros((20, 10, 10), bool)
        sk[2:4, 5, 5] = True
        segs = self.build(sk, constant_field(sk.shape, (1, 0, 0)), min_segment_length=100)
        self.assertEqual(len(segs), 1)

    def test_sign_of_field_irrelevant(self):
        sk, (oz, oy, ox) = x_crossing()
        a = self.build(sk, (oz, oy, ox))
        b = self.build(sk, (-oz, -oy, -ox))
        for s, t in zip(a, b):
            assert_vec(self, s.head_ori, t.head_ori)
            assert_vec(self, s.tail_ori, t.tail_ori)



if __name__ == '__main__':
    unittest.main()
