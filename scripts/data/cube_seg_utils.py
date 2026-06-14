"""Shared segmentation helpers for OGBench Cube data scripts.

Semantic labels (n_cubes-dependent):
    0        = background (floor, walls, target ghosts)
    1..N     = cube_0..cube_{N-1}  (cube i → label i+1)
    N+1      = robot arm (UR5e + Robotiq gripper)
    N+2      = shadow (optional; only on background pixels)

For single-cube this collapses to the legacy scheme: 0=bg, 1=cube, 2=arm, 3=shadow.

Shadow extraction: MuJoCo's segmentation renderer returns the geom visible
at each pixel, so shadows never appear in it.  Instead, render RGB twice
from the identical state -- once with shadows, once with all lights'
castshadow disabled -- and mark background pixels that brighten.
"""
import re

import numpy as np
from scipy.ndimage import binary_dilation

LABEL_BG = 0


def build_geom_to_label(model):
    """Map each MuJoCo geom id to a semantic label.

    Returns:
        lut: uint8 array of shape (ngeom,)
        n_cubes: number of distinct cube objects detected
        label_arm: label assigned to arm geoms
        label_shadow: label that add_shadow_class will use
    """
    # Detect cube count from geom names: object_0, object_1, ...
    cube_indices = set()
    for gid in range(model.ngeom):
        m = re.match(r'^object_(\d+)', model.geom(gid).name)
        if m:
            cube_indices.add(int(m.group(1)))
    n_cubes = len(cube_indices) if cube_indices else 1

    label_arm = n_cubes + 1
    label_shadow = n_cubes + 2

    lut = np.zeros(model.ngeom, dtype=np.uint8)
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if name.startswith('ur5e/'):
            lut[gid] = label_arm
            continue
        m = re.match(r'^object_(\d+)', name)
        if m and not name.startswith('target_object_'):
            lut[gid] = int(m.group(1)) + 1  # cube_i → label i+1

    return lut, n_cubes, label_arm, label_shadow


def seg_from_render(raw, lut, n_cubes, label_arm):
    """Convert a segmentation render (geom-id image) to a label mask.

    Each cube's pixels are dilated over adjacent arm pixels so cube boundaries
    are not lost to the gripper when grasped.
    """
    geom_ids = np.clip(raw[:, :, 0], -1, len(lut) - 1)
    mask = np.where(geom_ids >= 0, lut[geom_ids], LABEL_BG)
    for cube_label in range(1, n_cubes + 1):
        cube_dilated = binary_dilation(mask == cube_label)
        mask[(mask == label_arm) & cube_dilated] = cube_label
    return mask


def luminance(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def add_shadow_class(mask, rgb_shadow, rgb_noshadow, label_shadow, tau=4.0):
    """Mark background pixels that brighten when shadows are disabled."""
    diff = luminance(rgb_noshadow) - luminance(rgb_shadow)
    mask[(mask == LABEL_BG) & (diff > tau)] = label_shadow
    return mask
