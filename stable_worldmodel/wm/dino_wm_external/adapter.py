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


def load_dino_wm_external(
    ckpt_dir: str | Path,
    dino_wm_src: str | Path = '/home/nazirjon/Desktop/dino_wm',
    encoder_name: str | None = None,
    alpha: float = 1.0,
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

    return DinoWMAdapter(
        vwm=vwm,
        num_hist=num_hist,
        action_input_dim=action_input_dim,
        proprio_input_dim=proprio_input_dim,
        alpha=alpha,
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
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.vwm = vwm
        self.num_hist = num_hist
        # action_input_dim: the size CEM/the env supplies per step (= frameskip *
        # env_action_dim for PushT). The action_encoder embeds it to emb_dim.
        self.action_input_dim = action_input_dim
        self.proprio_input_dim = proprio_input_dim
        self.alpha = alpha
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
        if A != self.action_input_dim:
            raise ValueError(
                f'action_candidates last dim {A} != ckpt action_encoder.in_chans '
                f'{self.action_input_dim}. Check plan_config.action_block matches '
                'the DINO-WM frameskip.'
            )

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

        # Expand history across N candidates.
        pixels_h_exp = pixels_h.repeat_interleave(N, dim=0)   # (B*N, num_hist, 3, 224, 224)
        proprio_h_exp = proprio_h.repeat_interleave(N, dim=0) # (B*N, num_hist, proprio_dim)

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

        # Roll out. z_obses['visual']:  (B*N, num_hist + H + 1, P, D_visual)
        #          z_obses['proprio']: (B*N, num_hist + H + 1, proprio_dim)
        # The trailing index is the final unconditional next-state prediction
        # (i.e. the state after applying all H candidate actions starting from
        # the current frame).
        z_obses, _ = self.vwm.rollout(
            obs_0={'visual': pixels_h_exp, 'proprio': proprio_h_exp},
            act=act_full,
        )

        z_pred_visual = z_obses['visual'][:, -1:]    # (B*N, 1, P, D_visual)
        z_pred_proprio = z_obses['proprio'][:, -1:]  # (B*N, 1, proprio_dim)

        # Expand goal across N to align with predictions.
        z_goal_visual = z_goal['visual'].repeat_interleave(N, dim=0)
        z_goal_proprio = z_goal['proprio'].repeat_interleave(N, dim=0)

        # DINO-WM objective_fn_last: per-sample MSE averaged over all non-batch dims.
        visual_loss = F.mse_loss(
            z_pred_visual, z_goal_visual, reduction='none'
        ).mean(dim=tuple(range(1, z_pred_visual.ndim)))
        proprio_loss = F.mse_loss(
            z_pred_proprio, z_goal_proprio, reduction='none'
        ).mean(dim=tuple(range(1, z_pred_proprio.ndim)))

        cost = visual_loss + self.alpha * proprio_loss  # (B*N,)
        return cost.view(B, N)
