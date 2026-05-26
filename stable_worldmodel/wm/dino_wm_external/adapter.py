"""Adapter to plan with original DINO-WM (Zhou et al.) checkpoints.

The DINO-WM checkpoint is a dict of pickled `nn.Module` instances (predictor,
decoder, action_encoder, proprio_encoder), not a state-dict + hydra config, so
`stable_worldmodel.wm.utils.load_pretrained` cannot ingest it. The DINO-WM
encoder pipeline also resizes 224 to 196 px (14x14 patch grid), incompatible
with `prejepa.CausalPredictor`'s 16x16 layout.

This module sidesteps both issues by `sys.path`-prepending the on-disk DINO-WM
source tree, unpickling into the original classes, and wrapping the resulting
`VWorldModel` in a thin shim that exposes the planner's required surface:

  - `model.get_cost(info_dict, action_candidates) -> torch.Tensor of shape (B, N)`
  - `nn.Module` ops (`.to`, `.eval`, `.requires_grad_`)
  - settable `interpolate_pos_encoding` attribute (no-op for this model)

The cost mirrors DINO-WM's `objective_fn_last`
(<dino_wm>/planning/objectives.py): visual MSE + alpha * proprio MSE on the
last predicted timestep against the goal embedding.

NOTE on caching: PreJEPA caches goal/init encodings keyed on (id, step_idx).
The variation study runs the SAME episode/step under MULTIPLE variations
(same id, different goal pixels), so a naive cache would silently return a
stale, wrong-variation goal embedding. v1 deliberately re-encodes per
get_cost call. Re-introduce caching only after correctness is verified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange


def _patch_vit_attention_to_sdpa(vit_mod) -> None:
    """Replace DINO-WM's manual softmax attention with F.scaled_dot_product_attention.

    The original Attention.forward (vit.py:60-80) materializes a full
    (B, heads, T, T) attention matrix, which at planning scale (B*N up to
    several thousand, T = num_hist * num_patches = 588 for PushT) becomes
    the dominant memory and time cost. SDPA dispatches to FlashAttention
    on Ampere/Ada when shapes/dtypes allow, eliminating the full matrix.

    Idempotent: only patches if the original implementation is in place.
    """
    if getattr(vit_mod.Attention.forward, '_swm_sdpa_patched', False):
        return

    def forward(self, x):
        T = x.size(1)
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, 'b n (h d) -> b h n d', h=self.heads) for t in qkv
        )
        # Original mask is 1=attend, 0=block; SDPA expects bool with True=attend.
        attn_mask = self.bias[:, :, :T, :T].to(torch.bool)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    forward._swm_sdpa_patched = True
    vit_mod.Attention.forward = forward


def load_dino_wm_external(
    ckpt_dir: str | Path,
    dino_wm_src: str | Path = '/home/nazirjon/Desktop/dino_wm',
    encoder_name: str | None = None,
    alpha: float = 1.0,
    rollout_chunk: int = 64,
    dataset_h5: str | Path | None = None,
) -> 'DinoWMAdapter':
    """Load an original DINO-WM checkpoint and wrap it as a planner-compatible model.

    Args:
        ckpt_dir: directory containing `hydra.yaml` and `checkpoints/model_latest.pth`
            (e.g. `/home/.../outputs/pusht`).
        dino_wm_src: path to the original DINO-WM source repo. Must contain the
            `models/` package; we `sys.path`-prepend it before unpickling.
        encoder_name: torch.hub model name for the DINOv2 backbone. Defaults to
            the `encoder.name` field in the ckpt's hydra.yaml (e.g. `dinov2_vits14`).
        alpha: weight on the proprio MSE term in the planning cost. DINO-WM paper
            sweeps {0.1, 1.0}.

    Returns:
        DinoWMAdapter wrapping a fully-loaded `VWorldModel`.

    Notes:
        - The encoder is NOT in the checkpoint (DINO-WM freezes DINOv2). We
          rebuild it via `models.dino.DinoV2Encoder`, which calls
          `torch.hub.load("facebookresearch/dinov2", encoder_name)`. First call
          requires network; subsequent calls use the torch hub cache.
        - The pickled modules reference `models.{visual_world_model,vit,proprio,
          vqvae,dino}`. The `dino_wm_src` `sys.path` insertion makes them
          resolvable for `torch.load`.
    """
    ckpt_dir = Path(ckpt_dir)
    dino_wm_src = Path(dino_wm_src)
    if not (dino_wm_src / 'models').is_dir():
        raise FileNotFoundError(
            f'DINO-WM source not found at {dino_wm_src}/models. '
            'Set dino_wm_src to a checkout of the DINO-WM repo.'
        )

    src_str = str(dino_wm_src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    hydra_cfg_path = ckpt_dir / 'hydra.yaml'
    if not hydra_cfg_path.exists():
        raise FileNotFoundError(f'hydra.yaml not found in {ckpt_dir}')
    with open(hydra_cfg_path) as f:
        hydra_cfg = yaml.safe_load(f)

    ckpt_path = ckpt_dir / 'checkpoints' / 'model_latest.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'model_latest.pth not found in {ckpt_dir}/checkpoints')

    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    enc_name = encoder_name or hydra_cfg['encoder']['name']
    # Imports are dynamic — resolved via the sys.path insertion above. Pylance
    # / static analyzers can't see them; that's expected.
    from models.dino import DinoV2Encoder  # type: ignore[import-not-found]
    from models.visual_world_model import VWorldModel  # type: ignore[import-not-found]
    from models import vit as _vit_mod  # type: ignore[import-not-found]

    _patch_vit_attention_to_sdpa(_vit_mod)

    encoder = DinoV2Encoder(name=enc_name, feature_key='x_norm_patchtokens')

    # The semantics of `VWorldModel.{proprio_dim, action_dim}` are the EMBEDDED
    # dims (used by `separate_emb` to slice the concatenated latent, see
    # /home/nazirjon/Desktop/dino_wm/models/visual_world_model.py:185-190).
    # The raw input dims (what the env supplies / what CEM samples) are the
    # encoders' `in_chans`. For PushT they're {raw: 4 / 10, emb: 10 / 10}.
    proprio_emb_dim = ck['proprio_encoder'].emb_dim
    action_emb_dim = ck['action_encoder'].emb_dim
    proprio_input_dim = ck['proprio_encoder'].in_chans
    action_input_dim = ck['action_encoder'].in_chans
    num_hist = hydra_cfg['num_hist']
    num_pred = hydra_cfg['num_pred']
    concat_dim = hydra_cfg.get('concat_dim', 1)
    num_action_repeat = hydra_cfg.get('num_action_repeat', 1)
    num_proprio_repeat = hydra_cfg.get('num_proprio_repeat', 1)
    image_size = hydra_cfg.get('img_size', 224)

    vwm = VWorldModel(
        image_size=image_size,
        num_hist=num_hist,
        num_pred=num_pred,
        encoder=encoder,
        proprio_encoder=ck['proprio_encoder'],
        action_encoder=ck['action_encoder'],
        decoder=ck.get('decoder', None),
        predictor=ck['predictor'],
        proprio_dim=proprio_emb_dim,
        action_dim=action_emb_dim,
        concat_dim=concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=num_proprio_repeat,
        train_encoder=False,
        train_predictor=False,
        train_decoder=False,
    )

    # --- Action / proprio normalization stats ---
    # DINO-WM trains with normalize_action=true and normalize_proprio
    # (datasets/pusht_dset.py:91-110). The CEM planner samples actions in env
    # units; we must normalize them before feeding to the predictor (whose
    # action_encoder was trained on standardized inputs). Same for proprio.
    #
    # Compute stats from the source h5 if provided; otherwise fall back to
    # identity (no normalization), which matches behavior of older DINO-WM
    # ckpts that didn't normalize.
    normalize_action = hydra_cfg.get('normalize_action', False)
    if normalize_action and dataset_h5 is not None:
        import h5py
        import numpy as np
        with h5py.File(str(dataset_h5), 'r') as f:
            act = f['action'][:].astype(np.float32)
            prop = f['proprio'][:].astype(np.float32)
        action_mean = torch.from_numpy(act.mean(0)).float()
        action_std = torch.from_numpy(act.std(0).clip(min=1e-6)).float()
        proprio_mean = torch.from_numpy(prop.mean(0)).float()
        proprio_std = torch.from_numpy(prop.std(0).clip(min=1e-6)).float()
        print(f'[DinoWMAdapter] action_mean={action_mean.tolist()} '
              f'action_std={action_std.tolist()}')
        print(f'[DinoWMAdapter] proprio_mean={proprio_mean.tolist()} '
              f'proprio_std={proprio_std.tolist()}')
    else:
        if normalize_action:
            print('[DinoWMAdapter] WARN: ckpt has normalize_action=true but '
                  'no dataset_h5 was provided — using identity normalization '
                  '(model will see OOD inputs). Pass dino_wm_dataset_h5 in '
                  'the plan config to fix.')
        # Identity stats: shape inferred from raw encoder in_chans.
        # action_input_dim = frameskip * env_action_dim; we don't know the
        # split, so use action_input_dim and assume frameskip=1 fallback.
        env_action_dim_guess = action_input_dim
        action_mean = torch.zeros(env_action_dim_guess)
        action_std = torch.ones(env_action_dim_guess)
        proprio_mean = torch.zeros(proprio_input_dim)
        proprio_std = torch.ones(proprio_input_dim)

    return DinoWMAdapter(
        vwm=vwm,
        num_hist=num_hist,
        action_input_dim=action_input_dim,
        proprio_input_dim=proprio_input_dim,
        action_mean=action_mean,
        action_std=action_std,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        alpha=alpha,
        rollout_chunk=rollout_chunk,
    )


class DinoWMAdapter(nn.Module):
    """Wraps a DINO-WM `VWorldModel` to satisfy the CEM planner contract.

    The planner only ever calls `get_cost(info_dict, action_candidates)`. All
    other interactions (`.to`, `.eval`, etc.) flow through the underlying
    `VWorldModel` as a normal `nn.Module` child.
    """

    def __init__(
        self,
        vwm: nn.Module,
        num_hist: int,
        action_input_dim: int,
        proprio_input_dim: int,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        proprio_mean: torch.Tensor,
        proprio_std: torch.Tensor,
        alpha: float = 1.0,
        rollout_chunk: int = 64,
    ) -> None:
        super().__init__()
        self.vwm = vwm
        self.num_hist = num_hist
        self.action_input_dim = action_input_dim
        self.proprio_input_dim = proprio_input_dim
        self.alpha = alpha
        # Normalization stats — registered as buffers so .to(device) carries
        # them along with the model. action_mean/std are PER ENV-STEP
        # (shape: (env_action_dim,)); frameskip-bundled candidates get
        # reshaped, normalized per-step, then reshaped back.
        self.register_buffer('action_mean', action_mean)
        self.register_buffer('action_std', action_std)
        self.register_buffer('proprio_mean', proprio_mean)
        self.register_buffer('proprio_std', proprio_std)
        self.env_action_dim = int(action_mean.numel())
        self.frameskip = action_input_dim // self.env_action_dim
        # DINO-WM's predictor materializes a full (B*N, heads, T*P, T*P)
        # attention matrix per layer — at T=3, P=196 each row is 588 tokens, so
        # one float32 attention map is ~6.6 GB at N=300. Chunk the rollout
        # across N so peak memory scales with rollout_chunk instead of N.
        self.rollout_chunk = rollout_chunk
        # Set by eval_wm.py:103 for prejepa; ignored here because DINO-WM
        # resizes the image before encoding rather than interpolating positional
        # embeddings.
        self.interpolate_pos_encoding = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _apply(self, fn, *args, **kwargs):
        # DINO-WM's Attention stores its causal mask as a plain `self.bias`
        # attribute (not register_buffer), so nn.Module._apply skips it.
        # Walk submodules and apply `fn` to any such loose tensor attributes
        # so .to(device) / .cuda() actually moves them.
        super()._apply(fn, *args, **kwargs)
        for sub in self.modules():
            for name, val in list(vars(sub).items()):
                if torch.is_tensor(val) and not isinstance(val, nn.Parameter):
                    if name in sub._parameters or name in sub._buffers:
                        continue  # already handled by nn.Module._apply
                    setattr(sub, name, fn(val))
        return self

    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample planning cost for CEM.

        Args:
            info_dict: requires
                pixels       (B, T, 3, 224, 224) float32, ImageNet-normalized
                goal         (B, 1, 3, 224, 224) float32, ImageNet-normalized
                proprio      (B, T, proprio_dim) float
                goal_proprio (B, 1, proprio_dim) float
                T must be >= num_hist; we use the trailing num_hist frames.
            action_candidates: (B, N, H, action_dim) where action_dim equals
                the ckpt's `action_encoder.in_chans` (= frameskip * env_action_dim).

        Returns:
            cost: (B, N) torch.Tensor; lower is better. CEM minimizes it.
        """
        device = self.device

        candidates = action_candidates.to(device)
        B = candidates.shape[0]
        N = candidates.shape[1]
        H = candidates.shape[2]
        A = candidates.shape[3]

        # CEMSolver pre-expands info tensors along the sample dim before calling
        # get_cost (cem.py:128-135), so info[k] is shaped (B, N, T, ...) instead
        # of (B, T, ...). Per-env observations are identical across samples —
        # only action_candidates differ — so collapse with [:, 0] (mirrors
        # PreJEPA.rollout / get_cost which do the same).
        def _strip(x):
            if x.shape[0] == B and x.ndim >= 2 and x.shape[1] == N:
                return x[:, 0]
            return x

        pixels = _strip(info_dict['pixels']).to(device)
        proprio = _strip(info_dict['proprio']).to(device)
        # MegaWrapper stacks observations to history_size, including 'goal'
        # (replicates the same goal frame). The goal is one image, not a
        # history — collapse the time dim down to 1 with the last frame.
        goal = _strip(info_dict['goal']).to(device)[:, -1:]
        goal_proprio = _strip(info_dict['goal_proprio']).to(device)[:, -1:]

        # If world.history_size < ckpt num_hist (e.g. history_size=1 against a
        # num_hist=3 ckpt), pad-repeat the last available frame. The predictor
        # was trained on a fixed 3-frame window (pos_embedding sized for
        # num_hist*num_patches tokens) and would error otherwise.
        T_avail = pixels.shape[1]
        if T_avail < self.num_hist:
            pad = self.num_hist - T_avail
            pixels = torch.cat(
                [pixels[:, :1].expand(-1, pad, -1, -1, -1), pixels], dim=1
            )
            proprio = torch.cat(
                [proprio[:, :1].expand(-1, pad, -1), proprio], dim=1
            )
        if A != self.action_input_dim:
            raise ValueError(
                f'action_candidates last dim {A} != ckpt action_encoder.in_chans '
                f'{self.action_input_dim}. Check plan_config.action_block matches '
                'the DINO-WM frameskip.'
            )

        # --- Normalize inputs to match DINO-WM training distribution ---
        # CEM samples actions in env units; DINO-WM was trained on normalized
        # actions (per-env-step), and proprio likewise. Without these, the
        # model sees OOD inputs and planning collapses.
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        goal_proprio = (goal_proprio - self.proprio_mean) / self.proprio_std
        # candidates: (B, N, H, F*A_env) -> reshape, normalize per step, flatten
        candidates = candidates.view(B, N, H, self.frameskip, self.env_action_dim)
        candidates = (candidates - self.action_mean) / self.action_std
        candidates = candidates.view(B, N, H, self.frameskip * self.env_action_dim)

        # Encode goal once per env (independent of N).
        z_goal = self.vwm.encode_obs({
            'visual': goal.float(),
            'proprio': goal_proprio.float(),
        })
        # z_goal['visual']:  (B, 1, P, D_visual)
        # z_goal['proprio']: (B, 1, proprio_emb_dim)

        # Take trailing num_hist frames as planning history.
        pixels_h = pixels[:, -self.num_hist:].float()
        proprio_h = proprio[:, -self.num_hist:].float()

        # CRITICAL PERF/MEMORY: encode the visual history ONCE per env (B *
        # num_hist forward passes through DINO), then expand the resulting
        # latent across N samples. Without this, vwm.rollout would re-encode
        # B * N * num_hist frames per CEM iteration — for N=300 that's a 300x
        # blow-up, and DINOv2 activations OOM on 16GB cards. Pass the result
        # via the `features` key, which encode_obs uses to skip the encoder
        # (visual_world_model.py:125-127).
        v_in = self.vwm.encoder_transform(
            rearrange(pixels_h, 'b t c h w -> (b t) c h w')
        )
        v_emb_b = self.vwm.encoder.forward(v_in)
        v_emb_b = rearrange(v_emb_b, '(b t) p d -> b t p d', b=B)

        # Expand history latents and proprio across N candidates.
        v_emb_exp = v_emb_b.repeat_interleave(N, dim=0)        # (B*N, num_hist, P, D)
        proprio_h_exp = proprio_h.repeat_interleave(N, dim=0)  # (B*N, num_hist, proprio_dim)
        # `visual` key is read-only when `features` is present, but rollout's
        # signature still touches `obs_0['visual'].shape[1]` for num_obs_init.
        # Provide a tiny placeholder of the right shape (no allocation cost
        # since num_hist is small).
        pixels_shape_proxy = torch.empty(
            B * N, self.num_hist, 0, device=device, dtype=v_emb_exp.dtype
        )

        # Build the act sequence VWorldModel.rollout expects:
        #   act of shape (B*N, num_hist + H, action_dim).
        # The first num_hist slots are the actions taken WITHIN the history window
        # — unknown during MPC, so we use zeros (a standard approximation; matches
        # what the original DINO-WM planner does for the initial slots). The
        # remaining H slots are the candidate horizon actions.
        candidates_flat = candidates.reshape(B * N, H, A)
        zero_past = torch.zeros(
            B * N, self.num_hist, A,
            device=device, dtype=candidates_flat.dtype,
        )
        act_full = torch.cat([zero_past, candidates_flat], dim=1)

        # Expand goal across N once, then slice per chunk.
        z_goal_visual_full = z_goal['visual'].repeat_interleave(N, dim=0)
        z_goal_proprio_full = z_goal['proprio'].repeat_interleave(N, dim=0)

        BN = B * N
        chunk = max(1, self.rollout_chunk)
        cost_chunks = []
        # bfloat16 autocast halves predictor compute on Ada; the ckpt was
        # trained fp32 but inference-only fp16/bf16 typically drifts <0.1%
        # on MSE objectives. Bypass with DINO_WM_FP32=1 env var.
        import os as _os
        use_amp = _os.environ.get('DINO_WM_FP32', '0') != '1'
        for s in range(0, BN, chunk):
            e = min(s + chunk, BN)
            v_chunk = v_emb_exp[s:e]
            p_chunk = proprio_h_exp[s:e]
            a_chunk = act_full[s:e]
            proxy = torch.empty(
                e - s, self.num_hist, 0, device=device, dtype=v_chunk.dtype
            )
            with torch.amp.autocast(
                'cuda', dtype=torch.bfloat16, enabled=use_amp
            ):
                z_obses, _ = self.vwm.rollout(
                    obs_0={
                        'visual': proxy,
                        'proprio': p_chunk,
                        'features': v_chunk,
                    },
                    act=a_chunk,
                )
            z_pred_v = z_obses['visual'][:, -1:].float()
            z_pred_p = z_obses['proprio'][:, -1:].float()
            visual_loss = F.mse_loss(
                z_pred_v, z_goal_visual_full[s:e].float(), reduction='none'
            ).mean(dim=tuple(range(1, z_pred_v.ndim)))
            proprio_loss = F.mse_loss(
                z_pred_p, z_goal_proprio_full[s:e].float(), reduction='none'
            ).mean(dim=tuple(range(1, z_pred_p.ndim)))
            cost_chunks.append(visual_loss + self.alpha * proprio_loss)

        cost = torch.cat(cost_chunks, dim=0)
        return cost.view(B, N)
