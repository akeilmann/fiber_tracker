"""
Tests for topology_optimizer.fiber_io: export_uint16_tiff,
save_fiber_length_histogram, load_gt_centerlines_from_gad,
order_centerline_points and export_fibers_as_gad (including a round trip
export -> load).

Tests marked expectedFailure document known problems; they start passing
(and are then reported as "unexpected success") once the problem is fixed.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import tifffile
from skimage.draw import line_nd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from topology_optimizer.fiber_io import (
    export_fibers_as_gad, export_uint16_tiff, load_gt_centerlines_from_gad, order_centerline_points,
    save_fiber_length_histogram,
)


def polyline_voxels(*corners):
    """Voxels of the digital polyline through the given (z, y, x) corners, sorted."""
    pts = set()
    for a, b in zip(corners[:-1], corners[1:]):
        pts.update(zip(*line_nd(a, b, endpoint=True)))
    return np.array(sorted(pts))


def shuffled(p, seed=0):
    return p[np.random.default_rng(seed).permutation(len(p))]


class TmpDir(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.dir, name)


# --------------------------------------------------------------------------- #
# export_uint16_tiff                                                          #
# --------------------------------------------------------------------------- #
class TestExportUint16Tiff(TmpDir):
    def test_roundtrip_in_memory(self):
        a = np.arange(6 * 5 * 7, dtype=np.int32).reshape(6, 5, 7)
        export_uint16_tiff(a, self.path('a.tif'), verbose=False)
        b = tifffile.imread(self.path('a.tif'))
        self.assertEqual(b.dtype, np.uint16)
        np.testing.assert_array_equal(b, a)

    def test_roundtrip_memmap_streams_slices(self):
        mm = np.lib.format.open_memmap(self.path('a.npy'), mode='w+', dtype=np.int32, shape=(5, 7, 9))
        mm[:] = np.arange(mm.size).reshape(mm.shape)
        export_uint16_tiff(mm, self.path('a.tif'), verbose=False)
        np.testing.assert_array_equal(tifffile.imread(self.path('a.tif')), np.asarray(mm))
        del mm

    def test_2d(self):
        a = np.arange(12, dtype=np.int32).reshape(3, 4)
        export_uint16_tiff(a, self.path('a.tif'), verbose=False)
        np.testing.assert_array_equal(tifffile.imread(self.path('a.tif')), a)

    def test_creates_directory(self):
        export_uint16_tiff(np.zeros((2, 2, 2), np.int32), self.path('sub/dir/a.tif'), verbose=False)
        self.assertTrue(os.path.exists(self.path('sub/dir/a.tif')))

    @unittest.expectedFailure
    def test_labels_above_uint16_not_silently_wrapped(self):
        # label 65536 is written as 0 (background) and 65537 as 1 without warning
        a = np.array([[[1, 65536, 65537]]], np.int32)
        export_uint16_tiff(a, self.path('a.tif'), verbose=False)
        b = tifffile.imread(self.path('a.tif'))
        self.assertEqual(len(np.unique(b[b > 0])), 3)

    @unittest.expectedFailure
    def test_memmap_with_three_columns(self):
        # no photometric='minisblack': tifffile takes a last axis of length 3
        # as RGB and the streamed write fails
        mm = np.lib.format.open_memmap(self.path('a.npy'), mode='w+', dtype=np.int32, shape=(5, 7, 3))
        mm[:] = 1
        export_uint16_tiff(mm, self.path('a.tif'), verbose=False)
        np.testing.assert_array_equal(tifffile.imread(self.path('a.tif')), np.asarray(mm))


# --------------------------------------------------------------------------- #
# save_fiber_length_histogram                                                 #
# --------------------------------------------------------------------------- #
class TestSaveFiberLengthHistogram(TmpDir):
    def test_predicted_only(self):
        save_fiber_length_histogram([5, 10, 100, 1000], None, self.path('a.png'))
        self.assertGreater(os.path.getsize(self.path('a.png')), 0)

    def test_with_gt(self):
        save_fiber_length_histogram([5, 10, 100], [7, 50], self.path('b.png'))
        self.assertGreater(os.path.getsize(self.path('b.png')), 0)

    def test_no_fibers(self):
        # the runner calls this with an empty list when no fiber is found;
        # min([]) raises
        save_fiber_length_histogram([], None, self.path('a.png'))
        self.assertTrue(os.path.exists(self.path('a.png')))


# --------------------------------------------------------------------------- #
# order_centerline_points                                                     #
# --------------------------------------------------------------------------- #
class TestOrderCenterlinePoints(unittest.TestCase):
    def assert_path(self, ordered, original):
        self.assertEqual(len(ordered), len(original))
        self.assertEqual({tuple(p) for p in ordered.astype(int)}, {tuple(p) for p in original})
        steps = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
        self.assertLessEqual(steps.max(), np.sqrt(3) + 1e-9)

    def test_straight_line(self):
        p = polyline_voxels((0, 0, 0), (0, 0, 30))
        o = order_centerline_points(shuffled(p))
        self.assertEqual(o.dtype, np.float64)
        self.assert_path(o, p)
        self.assertEqual({o[0, 2], o[-1, 2]}, {0.0, 30.0})

    def test_curved_shapes(self):
        shapes = {
            'diagonal': [(0, 0, 0), (10, 12, 14)],
            'J': [(0, 0, 0), (0, 0, 30), (0, 6, 30), (0, 6, 24)],
            'hairpin': [(0, 0, 0), (0, 0, 30), (0, 4, 30), (0, 4, 0)],
            'spiral': [(0, 0, 0), (0, 0, 20), (0, 20, 20), (0, 20, 4), (0, 6, 4), (0, 6, 16), (0, 14, 16)],
        }
        for name, corners in shapes.items():
            with self.subTest(name):
                p = polyline_voxels(*corners)
                self.assert_path(order_centerline_points(shuffled(p)), p)

    def test_arc(self):
        t = np.deg2rad(np.arange(0, 300, 1.0))
        p = np.unique(np.round(np.column_stack([np.zeros_like(t), 10 * np.sin(t), 10 * np.cos(t)])).astype(int),
                      axis=0)
        self.assert_path(order_centerline_points(shuffled(p)), p)

    def test_two_points_or_fewer_unchanged(self):
        p = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(order_centerline_points(p), p)
        self.assertEqual(order_centerline_points(p).dtype, np.float64)

    def test_small_gap_bridged(self):
        p = np.vstack([polyline_voxels((0, 0, 0), (0, 0, 10)), polyline_voxels((0, 0, 14), (0, 0, 20))])
        self.assertEqual(len(order_centerline_points(shuffled(p))), len(p))


# --------------------------------------------------------------------------- #
# export_fibers_as_gad                                                        #
# --------------------------------------------------------------------------- #
class TestExportFibersAsGad(TmpDir):
    def setUp(self):
        super().setUp()
        self.out = self.path('f.gad')
        self.skel = np.zeros((30, 20, 25), np.int32)
        self.skel[2:28, 5, 5] = 3                                            # along z
        self.skel[tuple(polyline_voxels((10, 12, 2), (10, 12, 22)).T)] = 8   # along x

    def export(self, **kw):
        kw.setdefault('voxel_size', 1e-6)
        export_fibers_as_gad(self.skel, out_path=self.out, verbose=False, **kw)
        with open(self.out) as f:
            return json.load(f)

    def test_structure(self):
        g = self.export(voxel_size=2e-6)
        self.assertEqual(g['NumberOfObjects'], 2)
        self.assertEqual({k for k in g if k.startswith('Object')}, {'Object1', 'Object2'})
        dom = g['Domain']
        self.assertEqual(dom['VoxelLength'], [2e-6, 'm'])
        self.assertAlmostEqual(dom['LengthX'][0], 25 * 2e-6)
        self.assertAlmostEqual(dom['LengthY'][0], 20 * 2e-6)
        self.assertAlmostEqual(dom['LengthZ'][0], 30 * 2e-6)
        o = g['Object1']
        n_points = len([k for k in o if k.startswith('Point')])
        self.assertEqual(o['NumberOfSegments'], n_points - 1)
        self.assertEqual(o['Type'], 'CurvedCircularFiber')

    def test_coordinates_are_xyz_in_meters(self):
        obj = self.export()['Object1']               # label 3: along z at (y, x) = (5, 5)
        pts = np.array([obj[f'Point{i}'][0] for i in range(1, obj['NumberOfSegments'] + 2)])
        np.testing.assert_allclose(pts[:, 0], 5e-6)  # x
        np.testing.assert_allclose(pts[:, 1], 5e-6)  # y
        self.assertAlmostEqual(abs(pts[-1, 2] - pts[0, 2]), 25e-6)
        self.assertTrue(np.all(np.diff(pts[:, 2]) != 0), "points must be ordered along the fiber")

    def test_objects_follow_label_order(self):
        g = self.export()
        self.assertAlmostEqual(g['Object1']['Point1'][0][0], 5e-6)        # label 3 at x = 5
        self.assertAlmostEqual(g['Object2']['Point1'][0][1], 12e-6)       # label 8 at y = 12

    def test_default_diameter(self):
        g = self.export(default_diameter_vx=4.0)
        self.assertAlmostEqual(g['Object1']['Diameter'], 4e-6)

    def test_diameter_from_instance_volume(self):
        inst = np.zeros_like(self.skel)
        zz, yy, xx = np.mgrid[:30, :20, :25]
        inst[((yy - 5) ** 2 + (xx - 5) ** 2 <= 9) & (zz >= 2) & (zz < 28)] = 3
        g = self.export(inst_vol=inst)
        # 29 voxels per disc of radius 3 -> equivalent diameter 2 * sqrt(29 / pi) = 6.08
        self.assertAlmostEqual(g['Object1']['Diameter'] / 1e-6, 2 * np.sqrt(29 / np.pi), places=3)

    def test_min_points(self):
        self.skel[0, 0, 0] = 9
        self.assertEqual(self.export()['NumberOfObjects'], 2)
        self.assertEqual(self.export(min_points=1)['NumberOfObjects'], 3)
        self.assertEqual(self.export(min_points=26)['NumberOfObjects'], 1)

    def test_empty(self):
        self.skel[:] = 0
        self.assertEqual(self.export()['NumberOfObjects'], 0)

    def test_periodic_flag(self):
        dom = self.export(is_periodic=True)['Domain']
        self.assertTrue(dom['PeriodicX'] and dom['PeriodicY'] and dom['PeriodicZ'])
        dom = self.export(is_periodic=False)['Domain']
        self.assertFalse(dom['PeriodicX'] or dom['PeriodicY'] or dom['PeriodicZ'])

    def test_returns_path_and_creates_directory(self):
        out = self.path('sub/f.gad')
        self.assertEqual(export_fibers_as_gad(self.skel, out_path=out, verbose=False), out)
        self.assertTrue(os.path.exists(out))

    def test_roundtrip_through_loader(self):
        self.export()
        gt, n = load_gt_centerlines_from_gad(self.out, self.skel.shape)
        self.assertEqual(n, 2)
        # labels are renumbered 1..n, voxel sets are preserved
        for lbl in (3, 8):
            mask = self.skel == lbl
            ids = np.unique(gt[mask])
            self.assertEqual(len(ids), 1)
            np.testing.assert_array_equal(gt == ids[0], mask)


# --------------------------------------------------------------------------- #
# load_gt_centerlines_from_gad                                                #
# --------------------------------------------------------------------------- #
class TestLoadGtCenterlines(TmpDir):
    def write(self, objects, shape=(10, 12, 14), vs=1.0, point_style='list'):
        """GAD file with one object per list of (x, y, z) points in voxel units."""
        D, H, W = shape
        g = {'Domain': {'VoxelLength': [vs, 'm'], 'LengthX': [W * vs, 'm'], 'LengthY': [H * vs, 'm'],
                        'LengthZ': [D * vs, 'm']},
             'NumberOfObjects': len(objects)}
        for i, pts in enumerate(objects, start=1):
            o = {}
            for j, (x, y, z) in enumerate(pts, start=1):
                xyz = [x * vs, y * vs, z * vs]
                o[f'Point{j}'] = {'Coord': [xyz, 'm']} if point_style == 'dict' else [xyz, 'm']
            g[f'Object{i}'] = o
        with open(self.path('gt.gad'), 'w') as f:
            json.dump(g, f)
        return self.path('gt.gad')

    def test_straight_fiber(self):
        p = self.write([[(2, 3, 1), (2, 3, 8)]], vs=1e-6)
        gt, n = load_gt_centerlines_from_gad(p, (10, 12, 14))
        self.assertEqual(n, 1)
        self.assertEqual(gt.dtype, np.int32)
        np.testing.assert_array_equal(np.argwhere(gt), [[z, 3, 2] for z in range(1, 9)])

    def test_dict_point_format(self):
        p = self.write([[(2, 3, 1), (2, 3, 8)]], point_style='dict')
        gt, _ = load_gt_centerlines_from_gad(p, (10, 12, 14))
        self.assertEqual(int((gt == 1).sum()), 8)

    def test_points_sorted_numerically(self):
        # L-shaped fiber with 11 points; in string order (Point1, Point10,
        # Point11, Point2, ...) the polyline would cut diagonally across the L
        pts = [(1, 1, z) for z in range(6)] + [(x, 1, 5) for x in range(2, 7)]
        p = self.write([pts])
        gt, _ = load_gt_centerlines_from_gad(p, (10, 12, 14))
        expected = {(z, 1, 1) for z in range(6)} | {(5, 1, x) for x in range(1, 7)}
        self.assertEqual({tuple(q) for q in np.argwhere(gt)}, expected)

    def test_objects_with_one_point_skipped(self):
        p = self.write([[(1, 1, 1)], [(2, 3, 1), (2, 3, 8)]])
        gt, n = load_gt_centerlines_from_gad(p, (10, 12, 14))
        self.assertEqual(set(np.unique(gt)), {0, 2})
        self.assertEqual(n, 2)      # counts the skipped object too

    def test_multiple_fibers_labeled_by_object_index(self):
        p = self.write([[(2, 3, 1), (2, 3, 8)], [(9, 9, 0), (9, 9, 9)]])
        gt, _ = load_gt_centerlines_from_gad(p, (10, 12, 14))
        self.assertEqual(set(np.unique(gt)), {0, 1, 2})
        self.assertTrue(np.all(gt[:, 9, 9] == 2))

    @unittest.expectedFailure
    def test_periodic_fiber_crossing_the_border(self):
        # a short fiber crossing x = W in a periodic domain; after the modulo
        # wrap the interpolation runs back across the whole volume
        p = self.write([[(12, 5, 5), (16, 5, 5)]], shape=(10, 12, 14))
        gt, _ = load_gt_centerlines_from_gad(p, (10, 12, 14))
        self.assertLessEqual(int((gt > 0).sum()), 6)


if __name__ == '__main__':
    unittest.main()
