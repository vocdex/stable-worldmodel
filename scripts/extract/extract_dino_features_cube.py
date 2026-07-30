"""Pre-extract DINOv2-small features for cube_single_expert.h5.

Trades disk for training speed: the encoder is frozen during PreJEPA
training, so its output is deterministic. Caching the features once
eliminates the per-batch DINOv2 forward (the dominant cost) and lets
training run on the small predictor + action embedder alone.

We only extract frameskip-aligned frames (every 5th by default). Adjacent
windows in the training dataset overlap nearly identically when the slide
is 1 step at a time; restricting starts to frameskip-aligned positions
loses ~5x redundant data while keeping ~370K diverse training samples
(still orders of magnitude more than needed for a small predictor).

Output layout (matches HDF5Dataset conventions):
  features      (N_kept, P=256, D=384)  bf16  with blosc-lz4 compression
  action        (N_kept, A=5)            float32
  proprio_*     (N_kept, ...)            float32  (each split column kept)
  ep_len, ep_offset, episode_idx, step_idx  recomputed for subsampled stride

Run locally first:
    MUJOCO_GL=egl uv run python scripts/extract/extract_dino_features_cube.py
"""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from torchvision.transforms.functional import resize as tvf_resize


def _default_datasets_dir() -> Path:
    """Resolve via swm's cache utility so $STABLEWM_HOME wins over $HOME."""
    from stable_worldmodel.data.utils import get_cache_dir
    return get_cache_dir(sub_folder='datasets')


def get_args():
    p = argparse.ArgumentParser()
    datasets_dir = _default_datasets_dir()
    p.add_argument('--src', type=str,
                   default=str(datasets_dir / 'cube_single_expert.h5'))
    p.add_argument('--dst', type=str,
                   default=str(datasets_dir / 'cube_single_expert_features.h5'))
    p.add_argument('--frameskip', type=int, default=5)
    p.add_argument('--backbone', type=str, default='facebook/dinov2-small')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=128,
                   help='# of frames forwarded through DINO per batch')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dtype', type=str, choices=['bfloat16', 'float16', 'float32'],
                   default='bfloat16')
    p.add_argument('--max_episodes', type=int, default=None,
                   help='Process only the first N episodes (sanity).')
    return p.parse_args()


class _FrameDataset(Dataset):
    """Yields one frame at a time from src h5, ImageNet-normalized, resized."""

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __init__(self, src_path: str, kept_indices: np.ndarray, image_size: int):
        self.src_path = src_path
        self.kept_indices = kept_indices
        self.image_size = image_size
        self._h5 = None  # open lazily per worker

    def __len__(self):
        return len(self.kept_indices)

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.src_path, 'r', swmr=True,
                                 rdcc_nbytes=256 * 1024 * 1024)

    def __getitem__(self, i: int):
        self._open()
        g = int(self.kept_indices[i])
        img = self._h5['pixels'][g]  # (H, W, 3) uint8
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W) in [0, 1]
        t = (t - self.IMAGENET_MEAN) / self.IMAGENET_STD
        if t.shape[-1] != self.image_size or t.shape[-2] != self.image_size:
            t = tvf_resize(t, [self.image_size, self.image_size], antialias=True)
        return t


@torch.inference_mode()
def main():
    args = get_args()
    print(f'Source: {args.src}')
    print(f'Dest:   {args.dst}')

    src_path = Path(args.src)
    dst_path = Path(args.dst)
    if not src_path.exists():
        raise FileNotFoundError(f'{src_path} does not exist')
    if dst_path.exists():
        raise FileExistsError(f'{dst_path} already exists; refusing to overwrite')
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
             'float32': torch.float32}[args.dtype]
    np_dtype = {'bfloat16': 'uint16',  # bf16 saved as uint16 view; we'll use the proper h5 dtype
                'float16': np.float16,
                'float32': np.float32}[args.dtype]

    # --- Discover kept frame indices ---
    with h5py.File(src_path, 'r') as f:
        ep_lens = f['ep_len'][:]
        ep_offs = f['ep_offset'][:]
        if args.max_episodes is not None:
            ep_lens = ep_lens[: args.max_episodes]
            ep_offs = ep_offs[: args.max_episodes]
            print(f'  --max_episodes applied -> {len(ep_lens)} eps')
        # Probe a frame to learn shape
        sample_img = f['pixels'][0]
        H, W, C = sample_img.shape
        print(f'Source frames considered: {sum(ep_lens):,}  shape={(H, W, C)}  episodes={len(ep_lens)}')

        # Per-episode: keep every `frameskip`-th frame.
        kept_indices_list = []
        new_ep_lens = []
        new_ep_offs = []
        new_ep_idx = []
        new_step_idx = []
        running_offset = 0
        for ep_id, (off, L) in enumerate(zip(ep_offs, ep_lens)):
            local = np.arange(0, L, args.frameskip)
            global_idx = off + local
            kept_indices_list.append(global_idx)
            new_ep_lens.append(len(local))
            new_ep_offs.append(running_offset)
            new_ep_idx.extend([ep_id] * len(local))
            new_step_idx.extend(local.tolist())
            running_offset += len(local)
        kept_indices = np.concatenate(kept_indices_list)
        n_kept = len(kept_indices)
        print(f'Kept frames (every {args.frameskip}-th): {n_kept:,}')

        # Preload action: for each kept frame, store the FRAMESKIP actions
        # taken between this kept frame and the next (so model sees the full
        # action stream per "model step" — matches what the pixel pipeline
        # at frameskip=5 would feed in via dim*frameskip). Shape: (n_kept,
        # frameskip * A) — flattened so prejepa.py's `dim * cfg.frameskip`
        # logic stays consistent with cfg.frameskip=1 on the cached dataset.
        action_full = f['action'][:]  # (N_total, A)
        A = action_full.shape[-1]
        # For each kept_index k, grab action[k : k+frameskip]. Pad with the
        # last action if we hit episode end (rare; only the last cached
        # frame per episode).
        n_kept = len(kept_indices)
        action = np.zeros((n_kept, args.frameskip * A), dtype=np.float32)
        N_total = action_full.shape[0]
        for i, k in enumerate(kept_indices):
            end = min(int(k) + args.frameskip, N_total)
            chunk = action_full[int(k):end]              # (≤frameskip, A)
            if chunk.shape[0] < args.frameskip:           # pad with last
                pad = np.tile(chunk[-1:], (args.frameskip - chunk.shape[0], 1))
                chunk = np.concatenate([chunk, pad], axis=0)
            action[i] = chunk.reshape(-1)                # (frameskip*A,)
        proprio_cols = {}
        for k in f.keys():
            if k.startswith('proprio_'):
                proprio_cols[k] = f[k][:][kept_indices]

    # --- Build backbone ---
    print(f'Loading backbone {args.backbone} ...')
    encoder = AutoModel.from_pretrained(args.backbone).to(args.device)
    encoder.eval()
    encoder.requires_grad_(False)
    # DINOv3 prepends register tokens after CLS ([CLS, reg*4, patches]) and
    # its RoPE-based forward has no interpolate_pos_encoding kwarg.
    n_prefix = 1 + getattr(encoder.config, 'num_register_tokens', 0)
    fwd_kwargs = (
        {'interpolate_pos_encoding': True}
        if 'interpolate_pos_encoding'
        in inspect.signature(encoder.forward).parameters
        else {}
    )
    print(f'Prefix tokens stripped: {n_prefix}  forward kwargs: {fwd_kwargs}')
    # Probe output shape (patches, dim)
    with torch.amp.autocast(args.device, dtype=dtype):
        probe = torch.zeros(1, 3, args.image_size, args.image_size, device=args.device)
        out = encoder(probe, **fwd_kwargs).last_hidden_state
        out = out[:, n_prefix:, :]  # drop cls (+ register) tokens
        P, D = out.shape[1], out.shape[2]
    print(f'Feature shape per frame: ({P}, {D})  dtype={args.dtype}')
    print(f'Feature memory: {n_kept * P * D * (2 if dtype != torch.float32 else 4) / 1e9:.1f} GB (raw)')

    # --- Build dataloader ---
    ds = _FrameDataset(str(src_path), kept_indices, args.image_size)
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    # --- Open dst h5 with resizable feature dataset ---
    print(f'Writing to {dst_path} ...')
    with h5py.File(dst_path, 'w', libver='latest') as f_out:
        # Match HDF5Dataset metadata convention
        f_out.create_dataset('ep_len', data=np.array(new_ep_lens, dtype=np.int32))
        f_out.create_dataset('ep_offset', data=np.array(new_ep_offs, dtype=np.int64))
        f_out.create_dataset('episode_idx', data=np.array(new_ep_idx[:n_kept], dtype=np.int32))
        f_out.create_dataset('step_idx', data=np.array(new_step_idx[:n_kept], dtype=np.int32))

        # action + proprio
        f_out.create_dataset('action', data=action.astype(np.float32))
        for k, v in proprio_cols.items():
            f_out.create_dataset(k, data=v.astype(np.float32))

        # Features: chunked + compressed (blosc-lz4)
        feat_h5_dtype = np.float16 if args.dtype == 'bfloat16' else np_dtype
        feats = f_out.create_dataset(
            'features',
            shape=(n_kept, P, D),
            dtype=feat_h5_dtype,
            chunks=(min(256, n_kept), P, D),
            compression=hdf5plugin.Blosc(
                cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
            ),
        )
        # NOTE: h5py doesn't have a native bfloat16 dtype; we cast bf16 -> fp16
        # on write. The numerical drift is irrelevant for downstream MSE-on-
        # frozen-features training (DINOv2 features have ~0 in [-3, 3] range,
        # well within fp16's safe representable range).

        # --- Forward loop ---
        import time
        t0 = time.time()
        n_done = 0
        for batch in loader:
            batch = batch.to(args.device, non_blocking=True)
            with torch.amp.autocast(args.device, dtype=dtype):
                out = encoder(batch, **fwd_kwargs).last_hidden_state
                out = out[:, n_prefix:, :]  # drop cls (+ registers)
            out = out.to(torch.float16).cpu().numpy()
            n = out.shape[0]
            feats[n_done : n_done + n] = out
            n_done += n
            if n_done % (args.batch_size * 50) == 0 or n_done == n_kept:
                rate = n_done / (time.time() - t0)
                eta_min = (n_kept - n_done) / rate / 60 if rate > 0 else 0
                print(f'  {n_done:,}/{n_kept:,}  {rate:.0f} f/s  ETA {eta_min:.1f} min')

    # Stats
    size_bytes = dst_path.stat().st_size
    print(f'\nDone. {dst_path} = {size_bytes / 1e9:.1f} GB')


if __name__ == '__main__':
    main()
