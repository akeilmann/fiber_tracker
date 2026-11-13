"""
Tests for the classical orientation source (orientation/structure_tensor.py)
and the sign-invariant averaging in topology_optimizer/step1_sever_h_junctions.py.

Run with:  python -m unittest discover tests
"""
import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from orientation.structure_tensor import compute_structure_tensor_orientation, canonicalize_sign
    _IMPORT_ERROR = None
except ImportError as e:  # e.g. the structure-tensor package is not installed
    _IMPORT_ERROR = e


def make_fiber_volume(direction, size=48, radius=3.0, noise=0.0, seed=0):
    """Single straight fiber through the volume centre. Axes (Z, Y, X), values in [0, 1]."""
    d = np.asarray(direction, dtype=np.float32)
    d /= np.linalg.norm(d)
    zz, yy, xx = np.meshgrid(*(np.arange(size, dtype=np.float32),) * 3, indexing='ij')
    rel = np.stack([zz, yy, xx], -1) - (size - 1) / 2.0
    perp = rel - (rel @ d)[..., None] * d
    dist = np.linalg.norm(perp, axis=-1)
    vol = np.clip(1.2 - dist / radius, 0.0, 1.0)
    if noise > 0:
        vol = vol + np.random.default_rng(seed).normal(0.0, noise, vol.shape)
    return vol.astype(np.float32), dist <= radius, d


@unittest.skipIf(_IMPORT_ERROR is not None, f"orientation module not importable: {_IMPORT_ERROR}")
class TestStructureTensorOrientation(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _run(self, vol):
        in_path = os.path.join(self.tmp_dir.name, 'vol.npy')
        out_path = os.path.join(self.tmp_dir.name, 'ori.npy')
        np.save(in_path, vol)
        compute_structure_tensor_orientation(in_path, out_path, sigma=1.0, rho=6.0)
        return np.load(out_path)

    def _mean_abs_cos(self, ori, fg, d):
        """Mean |cos(angle)| between the field and the true direction inside the fiber,
        ignoring the outer 8 voxels where filter boundary effects live."""
        inner = np.zeros_like(fg)
        inner[8:-8, 8:-8, 8:-8] = True
        mask = fg & inner
        return float(np.abs(np.einsum('c...,c->...', ori, d))[mask].mean())

    def test_output_contract(self):
        """float32, shape (3, D, H, W), unit length -- what the optimizer expects."""
        vol, _, _ = make_fiber_volume((0, 0, 1))
        ori = self._run(vol)
        self.assertEqual(ori.dtype, np.float32)
        self.assertEqual(ori.shape, (3,) + vol.shape)
        np.testing.assert_allclose(np.linalg.norm(ori, axis=0), 1.0, atol=1e-4)

    def test_component_order_axis_aligned(self):
        """Component 0 must point along array axis 0 (Z), component 2 along axis 2 (X).
        This is the test that catches a missing or extra [::-1]."""
        for axis, name in enumerate(('Z', 'Y', 'X')):
            with self.subTest(fiber_along=name):
                direction = np.zeros(3)
                direction[axis] = 1.0
                vol, fg, d = make_fiber_volume(direction)
                ori = self._run(vol)
                self.assertGreater(self._mean_abs_cos(ori, fg, d), 0.98,
                                   f"fiber along {name}: field does not point along axis {axis}")

    def test_oblique_fiber_with_noise(self):
        """Distinct components (z != y != x) so a swapped order cannot pass by symmetry."""
        vol, fg, d = make_fiber_volume((0.2, 0.5, 1.0), noise=0.1)
        ori = self._run(vol)
        self.assertGreater(self._mean_abs_cos(ori, fg, d), 0.97)

    def test_sign_canonicalization(self):
        rng = np.random.default_rng(1)
        v = rng.normal(size=(3, 10, 10, 10)).astype(np.float32)
        out = canonicalize_sign(v, axis=0)
        ref = np.array([0.802, 0.535, 0.267], dtype=np.float32)
        self.assertTrue(np.all(np.einsum('c...,c->...', out, ref) >= 0))
        # Only signs change, never directions
        np.testing.assert_allclose(np.abs(out), np.abs(v))



if __name__ == '__main__':
    unittest.main()
