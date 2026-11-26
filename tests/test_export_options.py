import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
import tifffile
import subprocess

from inference.run_topology_optimization import (
    run_direct_topology_optimization,
    DEFAULT_PARAMS
)


class TestExportOptions(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.shape = (32, 32, 32)

        # Synthetic test inputs
        self.vol_path = os.path.join(self.test_dir, 'test_vol.tif')
        self.int_path = os.path.join(self.test_dir, 'test_int.tif')
        self.ori_path = os.path.join(self.test_dir, 'test_ori.tif')

        vol = np.zeros(self.shape, dtype=np.uint8)
        # Create a straight fiber
        vol[8:24, 16, 16] = 255
        tifffile.imwrite(self.vol_path, vol)

        intensity = np.zeros(self.shape, dtype=np.float32)
        intensity[8:24, 16, 16] = 0.9
        tifffile.imwrite(self.int_path, intensity)

        ori = np.zeros((32, 32, 32, 3), dtype=np.float32)
        ori[8:24, 16, 16, 0] = 1.0  # Z-aligned
        tifffile.imwrite(self.ori_path, ori, photometric='minisblack')

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_topology_optimization_default_no_npy(self):
        out_dir = os.path.join(self.test_dir, 'out_no_npy')
        params = DEFAULT_PARAMS.copy()
        params['cut_border'] = 0
        params['min_fiber_length'] = 2
        params['min_chain_length'] = 2

        self.assertFalse(params.get('export_npy', False))

        metrics = run_direct_topology_optimization(
            intensity_path=self.int_path,
            orientation_path=self.ori_path,
            volume_path=self.vol_path,
            out_dir=out_dir,
            params=params,
            verbose=False,
        )

        # Verify TIFFs and histogram exist
        self.assertTrue(os.path.exists(os.path.join(out_dir, 'instance_volume.tif')))
        self.assertTrue(os.path.exists(os.path.join(out_dir, 'instance_skeleton.tif')))
        self.assertTrue(os.path.exists(os.path.join(out_dir, 'fiber_length_histogram.png')))

        # Verify NPY files are NOT exported by default
        self.assertFalse(os.path.exists(os.path.join(out_dir, 'instance_volume.npy')))
        self.assertFalse(os.path.exists(os.path.join(out_dir, 'instance_skeleton.npy')))
        self.assertNotIn('instance_volume_npy', metrics)
        self.assertNotIn('instance_skeleton_npy', metrics)

    def test_topology_optimization_with_export_npy(self):
        out_dir = os.path.join(self.test_dir, 'out_with_npy')
        params = DEFAULT_PARAMS.copy()
        params['cut_border'] = 0
        params['min_fiber_length'] = 2
        params['min_chain_length'] = 2
        params['export_npy'] = True

        metrics = run_direct_topology_optimization(
            intensity_path=self.int_path,
            orientation_path=self.ori_path,
            volume_path=self.vol_path,
            out_dir=out_dir,
            params=params,
            verbose=False,
        )

        # Verify TIFFs exist
        self.assertTrue(os.path.exists(os.path.join(out_dir, 'instance_volume.tif')))
        self.assertTrue(os.path.exists(os.path.join(out_dir, 'instance_skeleton.tif')))

        # Verify NPY files ARE exported when requested
        self.assertTrue(os.path.exists(os.path.join(out_dir, 'instance_volume.npy')))
        self.assertTrue(os.path.exists(os.path.join(out_dir, 'instance_skeleton.npy')))
        self.assertIn('instance_volume_npy', metrics)
        self.assertIn('instance_skeleton_npy', metrics)


    def test_cli_export_npy_defaults(self):
        # Verify parser defaults in run_end_to_end.py
        import run_end_to_end
        import argparse
        # Check that --export-npy is accepted by command line help
        res = subprocess.run([sys.executable, 'run_end_to_end.py', '--help'], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn('--export-npy', res.stdout)

        res = subprocess.run([sys.executable, 'inference/run_topology_optimization.py', '--help'], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn('--export-npy', res.stdout)

        res = subprocess.run([sys.executable, 'inference/run_inference.py', '--help'], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn('--export-npy', res.stdout)



if __name__ == '__main__':
    unittest.main()
