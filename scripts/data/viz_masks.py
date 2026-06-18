"""Visualize segmentation mask quality from multi-cube mask H5s.

For each env_type, samples 5 random frames and renders a 2-row grid:
  top row: RGB  |  bottom row: colored mask overlay
Saves to visualizations/multicube_masks/{env_type}_masks.png
"""
import os
import sys

import h5py
import hdf5plugin  # registers blosc/zstd filters
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = '/home/nazirjon/Desktop/stable-worldmodel'
DATA = os.path.expanduser('~/.stable_worldmodel/datasets/ogbench')
OUT_DIR = os.path.join(REPO, 'visualizations', 'multicube_masks')
os.makedirs(OUT_DIR, exist_ok=True)

N_FRAMES = 5
RNG = np.random.default_rng(42)

# Per-label colors: bg=white, cubes=distinct, arm=gray, shadow=dark-gray
CUBE_COLORS = [
    [220, 50,  50],   # red
    [50,  130, 220],  # blue
    [50,  200, 80],   # green
    [230, 160, 20],   # orange
    [160, 50,  200],  # purple
    [20,  200, 200],  # cyan
    [200, 200, 20],   # yellow
    [200, 80,  160],  # pink
]
ARM_COLOR    = [160, 160, 160]
SHADOW_COLOR = [80,  80,  80]
BG_COLOR     = [240, 240, 240]

def label_to_color(mask, n_cubes):
    h, w = mask.shape
    rgb = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)
    for i in range(1, n_cubes + 1):
        rgb[mask == i] = CUBE_COLORS[(i - 1) % len(CUBE_COLORS)]
    rgb[mask == n_cubes + 1] = ARM_COLOR
    rgb[mask == n_cubes + 2] = SHADOW_COLOR
    return rgb

def mask_overlay(rgb, mask_color, alpha=0.55):
    out = rgb.astype(np.float32) * (1 - alpha) + mask_color.astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)

def legend_patches(n_cubes):
    patches = [mpatches.Patch(color=np.array(BG_COLOR)/255, label='bg')]
    for i in range(1, n_cubes + 1):
        c = np.array(CUBE_COLORS[(i-1) % len(CUBE_COLORS)]) / 255
        patches.append(mpatches.Patch(color=c, label=f'cube_{i-1}'))
    patches.append(mpatches.Patch(color=np.array(ARM_COLOR)/255, label='arm'))
    patches.append(mpatches.Patch(color=np.array(SHADOW_COLOR)/255, label='shadow'))
    return patches

for env_type in ('double', 'triple', 'quadruple'):
    path = os.path.join(DATA, f'cube_{env_type}_expert_masks.h5')
    if not os.path.exists(path):
        print(f'SKIP {env_type}: {path} not found')
        continue

    with h5py.File(path, 'r') as f:
        n_total = f['pixels'].shape[0]
        indices = sorted(RNG.choice(n_total, size=N_FRAMES, replace=False))
        pixels = f['pixels'][indices]       # (N, H, W, 3)
        segs   = f['segmentation'][indices] # (N, H, W)

    n_cubes = {'double': 2, 'triple': 3, 'quadruple': 4}[env_type]

    fig, axes = plt.subplots(2, N_FRAMES, figsize=(N_FRAMES * 3.5, 7))
    fig.suptitle(f'{env_type}  (n_cubes={n_cubes})  —  random frames', fontsize=13)

    for col, (rgb, seg) in enumerate(zip(pixels, segs)):
        mc = label_to_color(seg, n_cubes)
        overlay = mask_overlay(rgb, mc)

        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f'frame {indices[col]}', fontsize=8)
        axes[0, col].axis('off')

        axes[1, col].imshow(overlay)
        axes[1, col].axis('off')

        # label coverage stats in last column title
        if col == N_FRAMES - 1:
            total_px = seg.size
            stats = [f'bg={100*(seg==0).sum()/total_px:.0f}%']
            for i in range(1, n_cubes+1):
                stats.append(f'c{i}={100*(seg==i).sum()/total_px:.1f}%')
            stats.append(f'arm={100*(seg==n_cubes+1).sum()/total_px:.0f}%')
            stats.append(f'shd={100*(seg==n_cubes+2).sum()/total_px:.0f}%')
            axes[1, col].set_title('\n'.join(stats), fontsize=6)

    axes[0, 0].set_ylabel('RGB', fontsize=9)
    axes[1, 0].set_ylabel('mask overlay', fontsize=9)

    fig.legend(handles=legend_patches(n_cubes), loc='lower center',
               ncol=n_cubes + 3, fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out = os.path.join(OUT_DIR, f'{env_type}_masks.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'saved {out}')

print('done')
