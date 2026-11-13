import numpy as np
import tifffile
from structure_tensor.multiprocessing import parallel_structure_tensor_analysis

_SIGN_REF = np.array([0.802, 0.535, 0.267], dtype=np.float32)


def canonicalize_sign(vec, axis=0):
    """Flip vectors so that v . _SIGN_REF >= 0. `vec` has 3 components on `axis`."""
    ref = _SIGN_REF.reshape([-1 if i == axis else 1 for i in range(vec.ndim)])
    dot = np.sum(vec * ref, axis=axis, keepdims=True)
    return np.where(dot < 0, -vec, vec)


def repair_nan_eigenvectors(S, vec):
    """Replace NaN eigenvectors from eig_special_3d with a robust NumPy solution.

    S   : (6, ...) structure tensor as returned by the package,
          components (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)
    vec : (3, ...) eigenvector of the smallest eigenvalue, components (x, y, z)
    Returns vec with every non-finite voxel recomputed; all other voxels unchanged.
    """
    # 1. Find the voxels where the fast solver failed
    bad = ~np.isfinite(vec).all(axis=0)
    if not bad.any():
        return vec

    # 2. Take the six tensor entries of only those voxels
    sxx, syy, szz, sxy, sxz, syz = (S[i][bad] for i in range(6))

    # 3. Rebuild full symmetric 3x3 matrices, one per bad voxel
    J = np.empty((sxx.size, 3, 3), dtype=np.float64)
    J[:, 0, 0], J[:, 1, 1], J[:, 2, 2] = sxx, syy, szz
    J[:, 0, 1] = J[:, 1, 0] = sxy
    J[:, 0, 2] = J[:, 2, 0] = sxz
    J[:, 1, 2] = J[:, 2, 1] = syz

    # 4. Solve them with NumPy's general symmetric eigen-solver
    _, ev = np.linalg.eigh(J)        # eigenvalues ascending, eigenvectors in columns

    # 5. Smallest eigenvalue -> column 0 -> fiber direction; write it back
    vec = vec.copy()
    vec[:, bad] = ev[:, :, 0].T
    return vec


def compute_structure_tensor_orientation(volume, out_npy_path, sigma,
                                         rho=6.0,
                                         n_workers=8, block_size=200):
    """Writes the orientation field as float32 (3, D, H, W), components (vz, vy, vx)."""
    vol = np.load(volume) if str(volume).endswith('.npy') else tifffile.imread(volume)
    if vol.dtype == np.uint8:
        vol = vol.astype(np.float32) / 255.0
    elif vol.dtype == np.uint16:
        vol = vol.astype(np.float32) / 65535.0
    else:
        vol = vol.astype(np.float32)

    # sigma = noise/gradient scale, rho = integration scale (scalars: spacing is ignored)
    S, val, vec = parallel_structure_tensor_analysis(
        vol, sigma=sigma, rho=rho,
        structure_tensor=np.float32,
        devices=['cpu'] * n_workers, block_size=block_size,
    )
    vec = repair_nan_eigenvectors(S, vec)
    del S, val
    assert vec.shape == (3,) + vol.shape, f"unexpected eigenvector shape {vec.shape}"

    ori = vec[::-1].astype(np.float32)   # package gives (vx, vy, vz); pipeline wants (vz, vy, vx)
    ori = canonicalize_sign(ori, axis=0)
    np.save(out_npy_path, ori)
    return out_npy_path