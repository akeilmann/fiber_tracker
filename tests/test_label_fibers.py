"""
Tests for topology_optimizer.label_fibers: build_labeled_skeleton,
build_foreground, classify_components, watershed_component, write_labels
(through label_fibers) and label_fibers.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.ndimage import label as nd_label

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from topology_optimizer.label_fibers import (
    build_foreground, build_labeled_skeleton, classify_components, label_fibers, watershed_component,
)
from topology_optimizer.link_fibers import FiberChain
from topology_optimizer.skeleton_graph import FiberSegment


def line_segment(seg_id, start, stop, fixed=(10, 10)):
    """Straight FiberSegment along z from start to stop - 1 at (y, x) = fixed."""
    coords = np.zeros((stop - start, 3), np.int32)
    coords[:, 0] = np.arange(start, stop)
    coords[:, 1], coords[:, 2] = fixed
    return FiberSegment(seg_id=seg_id, coords=coords)


def label(inst_skel, max_watershed_voxels=2_000_000, **kw):
    kw.setdefault('fg_threshold', 0.10)
    kw.setdefault('vol_threshold', 0.05)
    return label_fibers(inst_skel, max_watershed_voxels=max_watershed_voxels, verbose=False, **kw)


def two_touching_tubes(shape=(20, 20, 20)):
    """Two z-tubes of half width 2 at x = 7 and x = 12 that touch along x = 9/10."""
    skel = np.zeros(shape, np.int32)
    skel[:, 10, 7] = 1
    skel[:, 10, 12] = 2
    vol = np.zeros(shape, np.float32)
    vol[:, 8:13, 5:15] = 1.0
    return skel, vol


# --------------------------------------------------------------------------- #
# build_labeled_skeleton                                                      #
# --------------------------------------------------------------------------- #
class TestBuildLabeledSkeleton(unittest.TestCase):
    shape = (40, 20, 20)

    def segments(self):
        return [line_segment(1, 0, 10), line_segment(2, 14, 30), line_segment(3, 32, 34, fixed=(3, 3))]

    def test_segments_and_bridges_drawn(self):
        segs = self.segments()
        chains = [FiberChain(7, [1, 2], [(np.array([9., 10, 10]), np.array([14., 10, 10]))], 26)]
        skel, n = build_labeled_skeleton(segs, chains, None, self.shape, min_chain_length=5)
        self.assertEqual(n, 1)
        self.assertEqual(skel.dtype, np.int32)
        np.testing.assert_array_equal(skel[0:30, 10, 10], 1)   # segments + bridge, relabeled 1
        self.assertEqual(int((skel > 0).sum()), 30)
        self.assertFalse(skel[32:34, 3, 3].any(), "segments outside kept chains stay 0")

    def test_labels_follow_chain_order(self):
        segs = self.segments()
        chains = [FiberChain(5, [2], [], 16), FiberChain(1, [1], [], 10)]
        skel, n = build_labeled_skeleton(segs, chains, None, self.shape, min_chain_length=1)
        self.assertEqual(n, 2)
        self.assertEqual(skel[20, 10, 10], 1)
        self.assertEqual(skel[5, 10, 10], 2)

    def test_min_chain_length(self):
        segs = self.segments()
        chains = [FiberChain(1, [2], [], 16), FiberChain(2, [1], [], 10), FiberChain(3, [3], [], 2)]
        skel, n = build_labeled_skeleton(segs, chains, None, self.shape, min_chain_length=10)
        self.assertEqual(n, 2)
        self.assertEqual(set(np.unique(skel)), {0, 1, 2})

    def test_diagonal_bridge_is_connected(self):
        segs = [line_segment(1, 0, 5), line_segment(2, 10, 15, fixed=(14, 13))]
        chains = [FiberChain(1, [1, 2], [(np.array([4., 10, 10]), np.array([10., 14, 13]))], 10)]
        skel, _ = build_labeled_skeleton(segs, chains, None, self.shape, min_chain_length=1)
        _, n_cc = nd_label(skel > 0, np.ones((3, 3, 3), bool))
        self.assertEqual(n_cc, 1)

    def test_writes_into_given_array_and_clears_it(self):
        out = np.full(self.shape, 99, np.int32)
        skel, _ = build_labeled_skeleton(self.segments(), [FiberChain(1, [1], [], 10)], None, self.shape,
                                         min_chain_length=1, out_skel_mmap=out)
        self.assertIs(skel, out)
        self.assertEqual(int((out > 0).sum()), 10)

    def test_no_chains(self):
        skel, n = build_labeled_skeleton(self.segments(), [], None, self.shape)
        self.assertEqual(n, 0)
        self.assertFalse(skel.any())


# --------------------------------------------------------------------------- #
# build_foreground                                                            #
# --------------------------------------------------------------------------- #
class TestBuildForeground(unittest.TestCase):
    def setUp(self):
        self.skel = np.zeros((4, 5, 5), np.int32)
        self.skel[:, 0, 0] = 1
        self.vol = np.zeros((4, 5, 5), np.float32)
        self.vol[:, 1, 1] = 0.06
        self.vol[:, 1, 2] = 0.05            # not > 0.05
        self.inten = np.zeros((4, 5, 5), np.float32)
        self.inten[:, 2, 2] = 0.10          # >= 0.10
        self.inten[:, 2, 3] = 0.09

    def test_thresholds(self):
        fg = build_foreground(self.skel, 0.10, 0.05, volume=self.vol, intensity=self.inten)
        self.assertEqual(fg.dtype, bool)
        expected = np.zeros_like(fg)
        expected[:, 0, 0] = expected[:, 1, 1] = expected[:, 2, 2] = True
        np.testing.assert_array_equal(fg, expected)

    def test_skeleton_always_foreground(self):
        fg = build_foreground(self.skel, 0.10, 0.05)
        np.testing.assert_array_equal(fg, self.skel > 0)

    def test_mask_overrides_volume_and_intensity(self):
        mask = np.zeros((4, 5, 5), np.uint8)
        mask[:, 4, 4] = 1
        fg = build_foreground(self.skel, 0.10, 0.05, foreground_mask=mask, volume=self.vol, intensity=self.inten)
        np.testing.assert_array_equal(fg, (self.skel > 0) | (mask > 0))


# --------------------------------------------------------------------------- #
# classify_components                                                         #
# --------------------------------------------------------------------------- #
class TestClassifyComponents(unittest.TestCase):
    def test_single_shared_and_empty_components(self):
        seed_cc = np.array([1, 1, 2, 2, 2, 4])
        seed_labels = np.array([5, 5, 6, 7, 6, 8])
        label_of, shared = classify_components(4, seed_cc, seed_labels)
        np.testing.assert_array_equal(label_of, [0, 5, 0, 0, 8])
        np.testing.assert_array_equal(shared, [2])


# --------------------------------------------------------------------------- #
# watershed_component                                                         #
# --------------------------------------------------------------------------- #
class TestWatershedComponent(unittest.TestCase):
    def test_geodesic_not_euclidean(self):
        # C-shaped component in one slice: the voxel at the end of the lower arm
        # is Euclidean-close to seed 2 (across background) but geodesically
        # closer to seed 1
        cc = np.zeros((1, 9, 9), np.int32)
        cc[0, 0, 0:9] = 1      # upper arm
        cc[0, 0:9, 0] = 1      # back
        cc[0, 8, 0:9] = 1      # lower arm
        seeds = np.array([[0, 8, 2], [0, 0, 8]])
        out = np.zeros_like(cc)
        box = (slice(0, 1), slice(0, 9), slice(0, 9))
        watershed_component(cc, 1, box, seeds, np.array([1, 2]), out)
        self.assertEqual(out[0, 8, 8], 1)
        self.assertEqual(out[0, 0, 7], 2)
        self.assertTrue(np.all(out[cc == 1] > 0))
        self.assertFalse(out[cc == 0].any())

    def test_writes_only_inside_component(self):
        cc = np.zeros((3, 6, 6), np.int32)
        cc[:, :, :3] = 1
        cc[:, :, 3:] = 2
        out = np.full(cc.shape, 9, np.int32)
        box = (slice(0, 3), slice(0, 6), slice(0, 6))
        watershed_component(cc, 1, box, np.array([[1, 2, 1]]), np.array([4]), out)
        self.assertTrue(np.all(out[cc == 1] == 4))
        self.assertTrue(np.all(out[cc == 2] == 9))


# --------------------------------------------------------------------------- #
# label_fibers                                                                #
# --------------------------------------------------------------------------- #
class TestLabelFibers(unittest.TestCase):
    def test_separate_tubes(self):
        skel = np.zeros((20, 20, 20), np.int32)
        skel[:, 5, 5] = 1
        skel[:, 14, 14] = 2
        vol = np.zeros(skel.shape, np.float32)
        vol[:, 3:8, 3:8] = 1.0
        vol[:, 12:17, 12:17] = 1.0
        out = label(skel, volume=vol)
        self.assertEqual(out.dtype, np.int32)
        self.assertTrue(np.all(out[:, 3:8, 3:8] == 1))
        self.assertTrue(np.all(out[:, 12:17, 12:17] == 2))
        self.assertFalse(out[vol == 0].any())

    def check_touching_split(self, out):
        self.assertTrue(np.all(out[:, 8:13, 5:10] == 1))
        self.assertTrue(np.all(out[:, 8:13, 10:15] == 2))

    def test_touching_tubes_watershed(self):
        skel, vol = two_touching_tubes()
        self.check_touching_split(label(skel, volume=vol))

    def test_touching_tubes_euclidean(self):
        skel, vol = two_touching_tubes()
        self.check_touching_split(label(skel, volume=vol, max_watershed_voxels=0))

    def test_watershed_and_euclidean_agree_on_convex_component(self):
        skel, vol = two_touching_tubes()
        np.testing.assert_array_equal(label(skel, volume=vol), label(skel, volume=vol, max_watershed_voxels=0))

    def test_euclidean_chunking(self):
        skel, vol = two_touching_tubes()
        np.testing.assert_array_equal(label(skel, volume=vol, max_watershed_voxels=0, chunk_size=7),
                                      label(skel, volume=vol, max_watershed_voxels=0))

    def test_component_without_seeds_unlabeled(self):
        skel = np.zeros((10, 10, 10), np.int32)
        skel[:, 2, 2] = 1
        vol = np.zeros(skel.shape, np.float32)
        vol[:, 1:4, 1:4] = 1.0
        vol[:, 7:9, 7:9] = 1.0
        out = label(skel, volume=vol)
        self.assertFalse(out[:, 7:9, 7:9].any())

    def test_skeleton_keeps_own_label(self):
        # a seed of fiber 2 that sits geometrically inside fiber 1's half
        skel, vol = two_touching_tubes()
        skel[3, 10, 8] = 2
        for mw in (2_000_000, 0):
            out = label(skel, volume=vol, max_watershed_voxels=mw)
            self.assertEqual(out[3, 10, 8], 2)
            np.testing.assert_array_equal(out[skel > 0], skel[skel > 0])

    def test_labels_not_passed_across_background(self):
        skel = np.zeros((10, 10, 10), np.int32)
        skel[:, 2, 2] = 1
        vol = np.zeros(skel.shape, np.float32)
        vol[:, 1:4, 1:4] = 1.0
        vol[:, 1:4, 5:8] = 1.0     # one voxel of background between the blocks
        out = label(skel, volume=vol, max_watershed_voxels=0)
        self.assertFalse(out[:, 1:4, 5:8].any())

    def test_foreground_mask(self):
        skel = np.zeros((10, 10, 10), np.int32)
        skel[:, 5, 5] = 3
        mask = np.zeros(skel.shape, bool)
        mask[:, 4:7, 4:7] = True
        vol = np.ones(skel.shape, np.float32)       # ignored when a mask is given
        out = label(skel, foreground_mask=mask, volume=vol)
        np.testing.assert_array_equal(out > 0, mask)
        self.assertEqual(set(np.unique(out)), {0, 3})

    def test_writes_into_memmap(self):
        skel, vol = two_touching_tubes()
        with tempfile.TemporaryDirectory() as d:
            mm = np.lib.format.open_memmap(os.path.join(d, 'o.npy'), mode='w+', dtype=np.int32, shape=skel.shape)
            mm[:] = 77
            out = label(skel, volume=vol, out_vol_mmap=mm)
            self.assertIs(out, mm)
            self.check_touching_split(np.asarray(mm))
            self.assertFalse(np.any(np.asarray(mm)[vol == 0]))
            del out, mm

    def test_empty_skeleton(self):
        out = label(np.zeros((5, 5, 5), np.int32), volume=np.ones((5, 5, 5), np.float32))
        self.assertFalse(out.any())

    def test_skeleton_only(self):
        skel = np.zeros((6, 6, 6), np.int32)
        skel[1:5, 3, 3] = 4
        np.testing.assert_array_equal(label(skel), skel)


if __name__ == '__main__':
    unittest.main()
