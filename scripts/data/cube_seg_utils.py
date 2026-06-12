"""Shared segmentation helpers for OGBench Cube data scripts.

Semantic labels:
    0 = background (floor, walls, target ghost)
    1 = cube
    2 = robot arm (UR5e + Robotiq gripper)
    3 = shadow (optional; only on background pixels)

Shadow extraction: MuJoCo's segmentation renderer returns the geom visible
at each pixel, so shadows never appear in it.  Instead, render RGB twice
from the identical state -- once with shadows, once with all lights'
castshadow disabled -- and mark background pixels that brighten.
"""
import numpy as np
from scipy.ndimage import binary_dilation

LABEL_BG = 0
LABEL_CUBE = 1
LABEL_ARM = 2
LABEL_SHADOW = 3


def build_geom_to_label(model):
    """Map each MuJoCo geom id to a semantic label."""
    lut = np.zeros(model.ngeom, dtype=np.uint8)
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if name.startswith('object_') and not name.startswith('target_object_'):
            lut[gid] = LABEL_CUBE
        elif name.startswith('ur5e/'):
            lut[gid] = LABEL_ARM
    return lut


def seg_from_render(raw, lut):
    """Convert a segmentation render (geom-id image) to a label mask.

    Cube pixels are dilated over adjacent arm pixels so the cube does not
    lose its boundary to the gripper when grasped.
    """
    geom_ids = np.clip(raw[:, :, 0], -1, len(lut) - 1)
    mask = np.where(geom_ids >= 0, lut[geom_ids], LABEL_BG)
    cube_dilated = binary_dilation(mask == LABEL_CUBE)
    mask[(mask == LABEL_ARM) & cube_dilated] = LABEL_CUBE
    return mask


def luminance(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def add_shadow_class(mask, rgb_shadow, rgb_noshadow, tau=4.0):
    """Mark background pixels that brighten when shadows are disabled."""
    diff = luminance(rgb_noshadow) - luminance(rgb_shadow)
    mask[(mask == LABEL_BG) & (diff > tau)] = LABEL_SHADOW
    return mask
