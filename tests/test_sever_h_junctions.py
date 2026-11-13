"""
Behavioural tests for topology_optimizer.step1_sever_h_junctions.sever_h_junctions.

Each test draws a small hand-made skeleton (one-voxel-wide digital lines in a
20 x 50 x 100 volume, all in the plane z = 10) together with an orientation field
on the skeleton voxels, and checks what the function must do with it:

  - an H-rung between two parallel fibers is cut, the fibers stay intact
  - a rung that is a single voxel between two junctions is cut (rule 0)
  - a bridge longer than max_rung_length is kept
  - an X crossing, a short branch aligned with its fiber, and an isolated short
    segment are left untouched
  - a short spur perpendicular to a long fiber is removed (rule B)
  - the result does not depend on the sign of the (noisy) orientation vectors
    (the orientation network is trained with the sign-free loss 1 - |u.v|).
    The original implementation, which averages signed vectors, fails this test
    for short branches at 30-50 degrees to their fiber; axial averaging passes.

A known limitation is documented as an expected failure: a rung consisting only
of junction voxels (fibers 3 voxels apart) is not cut.

The tests pass the parameters explicitly (PARAMS: max_rung_length=14,
min_neighbor_length=30, perp_thresh=0.5) and only check behaviour, so they
stay valid when the implementation or its defaults change.
"""

import os
import sys
import unittest

import numpy as np
from scipy.ndimage import label

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from topology_optimizer.step1_sever_h_junctions import sever_h_junctions

SHAPE = (20, 50, 100)

# Parameters the expectations below are based on (passed explicitly, so the tests do
# not depend on the function's defaults). A test can override them via run(**params).
PARAMS = dict(max_rung_length=14, min_neighbor_length=30, perp_thresh=0.50)
Z = 10
S26 = np.ones((3, 3, 3), bool)


def line(p0, p1):
    """26-connected, one-voxel-wide digital line from p0 to p1 (ZYX, inclusive)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = int(np.abs(p1 - p0).max())
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return np.unique(np.round(p0 + t * (p1 - p0)).astype(int), axis=0)


class Scene:
    """A skeleton built from named lines, with each line's direction as orientation."""

    def __init__(self):
        self.skel = np.zeros(SHAPE, bool)
        self.ori = np.zeros(SHAPE + (3,), np.float32)
        self.lines = {}

    def add(self, name, p0, p1, orientation=None):
        """Draws a line; `orientation` defaults to the line's own direction."""
        pts = line(p0, p1)
        d = np.asarray(p1, float) - np.asarray(p0, float)
        o = d / np.linalg.norm(d) if orientation is None else np.asarray(orientation, float)
        self.skel[tuple(pts.T)] = True
        self.ori[tuple(pts.T)] = o
        self.lines[name] = pts
        return self

    def run(self, flip_seed=None, noise=0.0, **params):
        """Runs sever_h_junctions.

        noise     : std of Gaussian noise added to the orientation vectors (fixed seed,
                    identical in every call), then renormalised
        flip_seed : if given, every orientation vector gets a random sign
        """
        ori = self.ori.copy()
        if noise > 0:
            on = np.linalg.norm(ori, axis=-1, keepdims=True) > 0
            ori = ori + np.random.default_rng(12345).normal(0, noise, ori.shape) * on
            ori /= np.where(on, np.linalg.norm(ori, axis=-1, keepdims=True), 1.0)
        if flip_seed is not None:
            signs = np.random.default_rng(flip_seed).choice([-1.0, 1.0], SHAPE)
            ori *= signs[..., None]
        # pass self.skel itself, so test_output_is_subset_and_input_unchanged can detect modification
        out = sever_h_junctions(self.skel, ori[..., 0], ori[..., 1], ori[..., 2], verbose=False,
                                **{**PARAMS, **params})
        return np.asarray(out, bool)


def connected(mask, a, b):
    """True if voxels a and b lie in the same 26-connected component of mask."""
    lab, _ = label(mask, S26)
    return lab[tuple(a)] != 0 and lab[tuple(a)] == lab[tuple(b)]


def two_parallel_fibers(dy, rung=True):
    """Two fibers along x at y = 20 and y = 20 + dy, optionally joined by a rung at x = 50.

    The rung carries the fibers' orientation (x), as a network would predict at a contact.
    """
    s = Scene().add('fiber_a', (Z, 20, 0), (Z, 20, 99)).add('fiber_b', (Z, 20 + dy, 0), (Z, 20 + dy, 99))
    if rung:
        s.add('rung', (Z, 21, 50), (Z, 20 + dy - 1, 50), orientation=(0, 0, 1))
    return s


class TestSeverHJunctions(unittest.TestCase):

    # ---- basic contract --------------------------------------------------------------

    def test_output_is_subset_and_input_unchanged(self):
        s = two_parallel_fibers(dy=10)
        before = s.skel.copy()
        out = s.run()
        self.assertEqual(out.shape, s.skel.shape)
        self.assertTrue(np.array_equal(s.skel, before), 'input skeleton was modified')
        self.assertFalse((out & ~s.skel).any(), 'output contains voxels not in the input')

    def test_empty_skeleton(self):
        out = Scene().run()
        self.assertFalse(out.any())

    # ---- what must be cut ------------------------------------------------------------

    def test_h_rung_is_cut_and_fibers_kept(self):
        s = two_parallel_fibers(dy=10)                       # rung of 9 voxels
        self.assertTrue(connected(s.skel, s.lines['fiber_a'][0], s.lines['fiber_b'][0]))
        out = s.run()
        self.assertFalse(connected(out, s.lines['fiber_a'][0], s.lines['fiber_b'][0]),
                         'the two fibers are still connected through the rung')
        for name in ('fiber_a', 'fiber_b'):
            self.assertTrue(out[tuple(s.lines[name].T)].all(), f'{name} lost voxels')

    def test_single_voxel_rung_is_cut(self):
        s = two_parallel_fibers(dy=4)                        # rung y = 21..23: one free voxel between two junctions
        out = s.run()
        self.assertFalse(connected(out, s.lines['fiber_a'][0], s.lines['fiber_b'][0]))
        for name in ('fiber_a', 'fiber_b'):
            self.assertTrue(out[tuple(s.lines[name].T)].all(), f'{name} lost voxels')

    def test_perpendicular_spur_is_removed(self):
        # documents rule B on branches with a single junction (a design choice; see module docstring)
        s = Scene().add('fiber', (Z, 20, 0), (Z, 20, 99)).add('spur', (Z, 21, 50), (Z, 28, 50))
        out = s.run()
        self.assertFalse(out[Z, 23:29, 50].any(), 'perpendicular spur was not removed')
        self.assertTrue(out[tuple(s.lines['fiber'].T)].all(), 'the fiber lost voxels')

    # ---- what must be kept -----------------------------------------------------------

    def test_long_bridge_is_kept(self):
        s = two_parallel_fibers(dy=22)                       # bridge of 21 voxels > max_rung_length
        out = s.run()
        self.assertTrue(np.array_equal(out, s.skel), 'a bridge longer than max_rung_length was changed')

    def test_x_crossing_is_kept(self):
        s = Scene().add('fiber_a', (Z, 30, 0), (Z, 30, 99)).add('fiber_b', (Z, 2, 50), (Z, 47, 50))
        out = s.run()
        self.assertTrue(np.array_equal(out, s.skel), 'an X crossing of long fibers was changed')

    def test_aligned_short_branch_is_kept(self):
        # a 10-voxel branch leaving the fiber at 30 degrees: well below the 60 degree limit
        a = np.deg2rad(30)
        s = Scene().add('fiber', (Z, 20, 0), (Z, 20, 99)).add('branch', (Z, 20, 50), (Z, 20 + 10 * np.sin(a), 50 + 10 * np.cos(a)))
        out = s.run()
        self.assertTrue(np.array_equal(out, s.skel), 'a short branch aligned with its fiber was removed')

    def test_isolated_short_segment_is_kept(self):
        s = Scene().add('fiber', (Z, 20, 0), (Z, 20, 99)).add('short', (Z, 40, 10), (Z, 40, 17))
        out = s.run()
        self.assertTrue(np.array_equal(out, s.skel), 'an isolated short segment was removed')

    # ---- invariance ------------------------------------------------------------------

    def test_result_does_not_depend_on_orientation_sign(self):
        def fiber_with_branch(angle_deg):
            a = np.deg2rad(angle_deg)
            return Scene().add('fiber', (Z, 20, 0), (Z, 20, 99)).add(
                'branch', (Z, 20, 50), (Z, 20 + 10 * np.sin(a), 50 + 10 * np.cos(a)))

        scenes = {
            'h_rung': two_parallel_fibers(dy=10),
            'single_voxel_rung': two_parallel_fibers(dy=4),
            'perpendicular_spur': Scene().add('fiber', (Z, 20, 0), (Z, 20, 99)).add('spur', (Z, 21, 50), (Z, 28, 50)),
            'x_crossing': Scene().add('a', (Z, 30, 0), (Z, 30, 99)).add('b', (Z, 2, 50), (Z, 47, 50)),
            # branches below the 60 degree limit: here sign-dependent averaging flips the decision
            'branch_30deg': fiber_with_branch(30),
            'branch_50deg': fiber_with_branch(50),
        }
        # Noise matters: flipping the signs of identical vectors keeps their sum along +-d, so
        # only noisy (realistic) orientations reveal sign-dependent averaging.
        for name, s in scenes.items():
            reference = s.run(noise=0.2)
            for seed in range(20):
                with self.subTest(scene=name, seed=seed):
                    self.assertTrue(np.array_equal(s.run(flip_seed=seed, noise=0.2), reference),
                                    'result changes when orientation signs are flipped')

    # ---- known limitation ------------------------------------------------------------

    @unittest.expectedFailure
    def test_rung_made_only_of_junction_voxels_is_cut(self):
        # Fibers 3 voxels apart: both rung voxels touch a fiber diagonally, so all of them are
        # junction voxels and no branch remains to be tested. Remove the decorator once handled.
        s = two_parallel_fibers(dy=3)
        out = s.run()
        self.assertFalse(connected(out, s.lines['fiber_a'][0], s.lines['fiber_b'][0]))


if __name__ == '__main__':
    unittest.main()
