"""
Tests for topology_optimizer.orientation_utils.

axial_mean(vectors) returns the mean direction of axial data (v and -v are the
same line) as the principal eigenvector of the orientation tensor mean(v v^T),
plus its eigenvalue as a measure of agreement (1 = parallel, 1/3 = isotropic
in 3D). Directions are compared up to sign throughout, since the sign of the
result is arbitrary.

axial_means(vectors, starts) is the batched version; its tests are skipped if
the function is not defined.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import topology_optimizer.orientation_utils as ou
from topology_optimizer.orientation_utils import axial_mean


def angle_deg(a, b):
    """Angle between the lines spanned by a and b, in [0, 90] degrees."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def noisy_axial_sample(direction, n, noise, seed):
    """n unit vectors scattered around `direction`, each with a random sign."""
    rng = np.random.default_rng(seed)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    v = d + rng.normal(0, noise, (n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * rng.choice([-1.0, 1.0], (n, 1))


class TestAxialMean(unittest.TestCase):

    # ---- simple cases with known answers --------------------------------------------

    def test_single_vector(self):
        d, lam = axial_mean([[0.0, 0.6, 0.8]])
        self.assertLess(angle_deg(d, [0, 0.6, 0.8]), 1e-6)
        self.assertAlmostEqual(lam, 1.0, places=10)

    def test_parallel_and_antiparallel_vectors_agree_fully(self):
        d, lam = axial_mean([[0, 0, 1], [0, 0, -1], [0, 0, 2], [0, 0, -5]])
        self.assertLess(angle_deg(d, [0, 0, 1]), 1e-6)
        self.assertAlmostEqual(lam, 1.0, places=10)

    def test_two_lines_give_their_bisector(self):
        # lines at +-30 degrees from y in the y-x plane: the mean axis is y, whatever the signs
        a, b = np.deg2rad(30), np.deg2rad(-30)
        v1, v2 = [0, np.cos(a), np.sin(a)], [0, np.cos(b), np.sin(b)]
        for w1, w2 in [(v1, v2), (v1, np.negative(v2)), (np.negative(v1), v2)]:
            d, lam = axial_mean([w1, w2])
            self.assertLess(angle_deg(d, [0, 1, 0]), 1e-6)
            self.assertAlmostEqual(lam, np.cos(np.deg2rad(30)) ** 2, places=10)

    def test_agreement_values(self):
        self.assertAlmostEqual(axial_mean([[1, 0, 0], [0, 1, 0]])[1], 0.5, places=10)   # crossing
        iso = np.random.default_rng(0).normal(size=(20000, 3))
        self.assertAlmostEqual(axial_mean(iso)[1], 1 / 3, delta=0.02)                  # no preferred direction

    # ---- properties --------------------------------------------------------------------

    def test_result_is_unit_vector(self):
        d, _ = axial_mean(noisy_axial_sample([1, 2, 3], 40, 0.3, seed=1))
        self.assertAlmostEqual(float(np.linalg.norm(d)), 1.0, places=10)

    def test_sign_invariance(self):
        v = noisy_axial_sample([0, 1, 1], 30, 0.2, seed=2)
        d_ref, lam_ref = axial_mean(v)
        rng = np.random.default_rng(3)
        for _ in range(10):
            d, lam = axial_mean(v * rng.choice([-1.0, 1.0], (len(v), 1)))
            self.assertLess(angle_deg(d, d_ref), 1e-6)
            self.assertAlmostEqual(lam, lam_ref, places=10)

    def test_order_and_length_invariance(self):
        v = noisy_axial_sample([1, 0, 1], 30, 0.2, seed=4)
        d_ref, lam_ref = axial_mean(v)
        rng = np.random.default_rng(5)
        scaled = v[rng.permutation(len(v))] * rng.uniform(0.1, 10.0, (len(v), 1))
        d, lam = axial_mean(scaled)
        self.assertLess(angle_deg(d, d_ref), 1e-6)
        self.assertAlmostEqual(lam, lam_ref, places=10)

    def test_recovers_direction_from_noisy_random_sign_sample(self):
        # this is where a signed mean fails: the vectors largely cancel
        truth = [0, 0.6, 0.8]
        v = noisy_axial_sample(truth, 50, 0.1, seed=6)
        d, lam = axial_mean(v)
        self.assertLess(angle_deg(d, truth), 3.0)
        self.assertGreater(lam, 0.9)

    # ---- input handling ----------------------------------------------------------------

    def test_zero_vectors_are_ignored(self):
        v = noisy_axial_sample([1, 1, 0], 20, 0.2, seed=7)
        d_ref, lam_ref = axial_mean(v)
        d, lam = axial_mean(np.vstack([v, np.zeros((15, 3))]))
        self.assertLess(angle_deg(d, d_ref), 1e-6)
        self.assertAlmostEqual(lam, lam_ref, places=10)

    def test_no_usable_vectors_returns_none(self):
        for empty in (np.zeros((0, 3)), np.zeros((4, 3)), [[0.0, 0.0, 1e-12]]):
            d, lam = axial_mean(empty)
            self.assertIsNone(d)
            self.assertEqual(lam, 0.0)

    def test_accepts_lists_float32_and_single_vector(self):
        self.assertLess(angle_deg(axial_mean([[0, 0, 1], [0, 0, -1]])[0], [0, 0, 1]), 1e-6)
        self.assertLess(angle_deg(axial_mean(np.array([[0, 3, 4]], np.float32))[0], [0, 3, 4]), 1e-5)
        self.assertLess(angle_deg(axial_mean(np.array([0, 3, 4]))[0], [0, 3, 4]), 1e-6)    # shape (3,)


if __name__ == '__main__':
    unittest.main()
