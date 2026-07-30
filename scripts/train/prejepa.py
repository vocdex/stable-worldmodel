from collections import OrderedDict
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForImageClassification,
    AutoVideoProcessor,
)

# fmt: off
ENCODER_CONFIGS = {
    'resnet': {
        'prefix': 'microsoft/resnet-',
        'model_class': AutoModelForImageClassification,
        'embedding_attr': lambda m: m.config.hidden_sizes[-1],
        'post_init': lambda m: setattr(m.classifier, '1', nn.LayerNorm(m.config.hidden_sizes[-1])),
        'interpolate_pos_encoding': False,
    },
    'vit':    {'prefix': 'google/vit-'},
    'dino':   {'prefix': 'facebook/dino-'},
    'dinov2':  {'prefix': 'facebook/dinov2-'},
    'dinov3':  {
        'prefix': 'facebook/dinov3-',
        # RoPE positions — forward() has no interpolate_pos_encoding kwarg.
        'interpolate_pos_encoding': False,
        # last_hidden_state = [CLS, n_register (4 for vits16), patches].
        'num_prefix_tokens': lambda m: 1 + getattr(m.config, 'num_register_tokens', 0),
    },
    'webssl':  {'prefix': 'facebook/webssl-'},
    'mae':    {'prefix': 'facebook/vit-mae-'},
    'ijepa':  {'prefix': 'facebook/ijepa'},
    'vjepa2':  {'prefix': 'facebook/vjepa2-vit'},
    'siglip2': {'prefix': 'google/siglip2-'},
}
# fmt: on


def get_encoder(cfg):
    """Load a pretrained vision encoder and return (backbone, embed_dim, num_patches, interp_pos_enc, num_prefix_tokens)."""
    encoder_cfg = next(
        (
            c
            for c in ENCODER_CONFIGS.values()
            if cfg.backbone.name.startswith(c['prefix'])
        ),
        None,
    )
    if encoder_cfg is None:
        raise ValueError(f'Unsupported backbone: {cfg.backbone.name}')

    backbone = encoder_cfg.get('model_class', AutoModel).from_pretrained(
        cfg.backbone.name
    )
    if hasattr(backbone, 'vision_model'):  # CLIP-style
        backbone = backbone.vision_model
    if 'post_init' in encoder_cfg:
        encoder_cfg['post_init'](backbone)

    embed_dim = encoder_cfg.get(
        'embedding_attr', lambda m: m.config.hidden_size
    )(backbone)
    is_cnn = cfg.backbone.name.startswith('microsoft/resnet-')
    num_patches = 1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2
    interp_pos_enc = encoder_cfg.get('interpolate_pos_encoding', True)
    num_prefix_tokens = encoder_cfg.get(
        'num_prefix_tokens', lambda m: 1
    )(backbone)

    backbone_patch = getattr(backbone.config, 'patch_size', None)
    if not is_cnn and backbone_patch is not None:
        assert backbone_patch == cfg.patch_size, (
            f'cfg.patch_size={cfg.patch_size} but {cfg.backbone.name} has '
            f'patch_size={backbone_patch}; num_patches would be wrong.'
        )

    return backbone, embed_dim, num_patches, interp_pos_enc, num_prefix_tokens


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def get_img_preprocessor(source, target, img_size=224):
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )


def get_column_normalizer(dataset, source, target):
    data = torch.from_numpy(dataset.get_col_data(source)[:])
    data = data[~torch.isnan(data).any(dim=1)]
    mean, std = (
        data.mean(0, keepdim=True).clone(),
        data.std(0, keepdim=True).clone(),
    )
    return spt.data.transforms.WrapTorchTransform(
        lambda x: ((x - mean) / std).float(),
        source=source,
        target=target,
    )


class VideoPipeline(spt.data.transforms.Transform):
    def __init__(self, processor, source='image', target='image'):
        super().__init__()
        self.processor, self.source, self.target = processor, source, target

    def __call__(self, x):
        frames = self.nested_get(x, self.source)
        self.nested_set(
            x,
            self.processor(frames, return_tensors='pt')[
                'pixel_values_videos'
            ].squeeze(0),
            self.target,
        )
        return x


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval=1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0:
            self._save(pl_module.model, epoch)
        if epoch == trainer.max_epochs:
            self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        save_pretrained(model, run_name=self.run_name, config=self.cfg, filename=f'weights_epoch_{epoch}.pt')


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _strip_action_dims(tensor, action_range):
    """Remove the action dimensions from the last axis."""
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]],
        dim=-1,
    )


def dinowm_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0).squeeze()

    batch = self.model.encode(
        batch,
        target='embed',
        is_video=cfg.backbone.get('is_video_encoder', False),
    )

    embedding = batch['embed'][:, : cfg.wm.history_size, ...]
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch['embed'][:, cfg.wm.num_preds :, ...].detach()

    # Per-modality losses
    pixels_dim = batch['pixels_embed'].size(-1)
    batch['pixels_loss'] = F.mse_loss(
        pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim]
    )

    start, action_range = pixels_dim, [0, 0]
    for key in self.model.extra_encoders:
        dim = batch[f'{key}_embed'].size(-1)
        lo, hi = start, start + dim
        if key == 'action':
            action_range = [lo, hi]
        else:
            batch[f'{key}_loss'] = F.mse_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
            )
        start = hi

    # Actionless embeddings (for probes and total loss)
    batch['actionless_embed'] = _strip_action_dims(
        batch['embed'], action_range
    )
    batch['actionless_prev_embed'] = _strip_action_dims(
        embedding, action_range
    )
    batch['actionless_pred_embed'] = _strip_action_dims(
        pred_embedding, action_range
    )
    batch['actionless_target_embed'] = _strip_action_dims(
        target_embedding, action_range
    )

    batch['loss'] = F.mse_loss(
        batch['actionless_pred_embed'],
        batch['actionless_target_embed'].detach(),
    )

    if batch['loss'].isnan():
        raise ValueError('NaN loss encountered!')

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in batch.items() if '_loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path='./config', config_name='prejepa')
def run(cfg):
    # --- Dataset ---
    # Pre-extracted-features path: when cfg.use_cached_features=true the dataset
    # loads frozen DINOv2 features (column 'features', shape (T, P, D) per frame)
    # instead of raw 'pixels'. Skips the image preprocessor and (via the
    # 4D-passthrough in PreJEPA._encode_image) the encoder forward at train
    # time. Extract once with scripts/extract/extract_dino_features_cube.py.
    use_cached_features = bool(cfg.get('use_cached_features', False))
    visual_key = 'features' if use_cached_features else 'pixels'

    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    keys_to_load = [visual_key] + encoding_keys

    dataset = swm.data.HDF5Dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_load=keys_to_load,
        keys_to_cache=encoding_keys,
    )

    normalizers = [
        get_column_normalizer(dataset, col, col)
        for col in cfg.wm.get('encoding', {})
    ]

    if use_cached_features:
        # No image preprocessing — features are already DINO-encoded.
        # Rename 'features' -> 'pixels' so the rest of the pipeline (the
        # model's encode() / PreJEPA forward) is untouched.
        class _RenameAndCastFeatures(spt.data.transforms.Transform):
            def __call__(self, x):
                feats = self.nested_get(x, 'features')
                # h5 stored as fp16 (lz4-compressed). Cast to fp32 on read.
                self.nested_set(x, feats.float(), 'pixels')
                return x

        transform = spt.data.transforms.Compose(
            _RenameAndCastFeatures(), *normalizers,
        )
    elif cfg.backbone.get('is_video_encoder', False):
        processor = AutoVideoProcessor.from_pretrained(cfg.backbone.name)
        transform = spt.data.transforms.Compose(
            VideoPipeline(processor, source='pixels', target='pixels'),
            spt.data.transforms.Resize(
                cfg.image_size, source='pixels', target='pixels'
            ),
            *normalizers,
        )
    else:
        transform = spt.data.transforms.Compose(
            get_img_preprocessor('pixels', 'pixels', cfg.image_size),
            *normalizers,
        )
    dataset.transform = transform

    with open_dict(cfg) as cfg:
        cfg.extra_dims = {}
        for key in cfg.wm.get('encoding', {}):
            if key not in dataset.column_names:
                raise ValueError(
                    f"Encoding key '{key}' not found in dataset columns."
                )
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = (
                dim if key != 'action' else dim * cfg.frameskip
            )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # --- Model ---
    encoder, embed_dim, num_patches, interp_pos_enc, num_prefix_tokens = (
        get_encoder(cfg)
    )
    embed_dim += sum(cfg.wm.get('encoding', {}).values())

    if cfg.backbone.get('is_video_encoder', False):
        num_patches += num_patches * (cfg.n_steps // 4)

    predictor_kwargs = {k: v for k, v in cfg.predictor.items() if k != 'size'}
    predictor = swm.wm.prejepa.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.wm.history_size,
        dim=embed_dim,
        **predictor_kwargs,
    )

    extra_encoders = nn.ModuleDict(
        OrderedDict(
            (
                key,
                swm.wm.prejepa.Embedder(
                    in_chans=cfg.extra_dims[key], emb_dim=emb_dim
                ),
            )
            for key, emb_dim in cfg.wm.get('encoding', {}).items()
        )
    )

    world_model = swm.wm.PreJEPA(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=cfg.wm.history_size,
        num_pred=cfg.wm.num_preds,
        interpolate_pos_encoding=interp_pos_enc,
        num_prefix_tokens=num_prefix_tokens,
    )

    world_model = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={
            'model_opt': {'modules': 'model', 'optimizer': dict(cfg.optimizer)}
        },
    )

    # --- Training ---
    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Run ID: {run_id}')

    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enable:
        logger = WandbLogger(
            name='dino_wm',
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            resume='allow' if run_id else None,
            id=run_id or None,
            log_model=False,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    # NOTE: do NOT explicitly add CPUOffloadCallback — stable_pretraining
    # auto-registers it (and other callbacks) via the
    # 'stablepretraining_callbacks' entry point, which Lightning resolves
    # at Trainer init. Adding it manually here triggers Lightning's
    # "Found more than one stateful callback of type CPUOffloadCallback".
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg,
                epoch_interval=5,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path=run_dir / f'{cfg.output_model_name}_weights.ckpt',
    )
    manager()


if __name__ == '__main__':
    run()
