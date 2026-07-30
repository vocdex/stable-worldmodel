import json
import urllib.request
from pathlib import Path
import torch

from loguru import logger as logging
from tqdm import tqdm

from stable_worldmodel.utils import HF_BASE_URL
from stable_worldmodel.data import get_cache_dir, ensure_dir_exists


def save_pretrained(
    model: torch.nn.Module,
    run_name: str,
    config: dict | None = None,
    config_key: str | None = None,
    filename: str = 'weights.pt',
    cache_dir: str = None,
):
    from omegaconf import OmegaConf

    ckpt_dir = get_cache_dir(cache_dir, sub_folder='checkpoints') / run_name
    ensure_dir_exists(ckpt_dir)

    checkpoint_path = ckpt_dir / filename
    torch.save(model.state_dict(), checkpoint_path)

    if config is None:
        logging.warning('No config! Loading will have to be done manually.')
        return

    if config_key is not None and config_key in config:
        config = config[config_key]

    config_path = ckpt_dir / 'config.json'

    config = OmegaConf.to_container(config, resolve=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logging.info(f'📦📦📦 Model saved to {checkpoint_path} 📦📦📦')

    return


def load_pretrained(name: str, cache_dir: str = None, extra_args=None):
    """Load a model from a local checkpoint or a HuggingFace repository.

    Supported formats for `name`:

    1. **`.pt` file** — path to a specific checkpoint file.
       A `config.json` must live in the same directory.

        ```python
        model = load_pretrained('my_run/weights_epoch_10.pt')
        ```

    2. **Folder** — path to a directory containing exactly one `.pt` file
       and a `config.json`.

        ```python
        model = load_pretrained('my_run/')
        ```

    3. **HuggingFace repo** (`<user>/<repo>`) — loaded from the local cache
       if already present, otherwise fetched from HF.

        ```python
        model = load_pretrained('nice-user/my-worldmodel')
        ```

    All local paths are resolved relative to `<cache_dir>/checkpoints/`.
    """
    from hydra.utils import instantiate

    cache_dir = get_cache_dir(cache_dir, sub_folder='checkpoints')
    ensure_dir_exists(cache_dir)
    checkpoint_path, config = _resolve(name, cache_dir)
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # assume keys with the dotted notation
    if extra_args is not None:
        for key, value in extra_args.items():
            parts = key.split('.')
            d = config
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

    # If the config has a _target_, use hydra to instantiate. Otherwise it's
    # a training-cfg dump (e.g. prejepa save_pretrained writes the full
    # training cfg, not a model-only Hydra spec) — try detecting known shapes
    # and reconstructing manually.
    if '_target_' in config:
        model = instantiate(config)
    elif {'predictor', 'wm', 'backbone'} <= set(config):
        model = _build_prejepa_from_training_config(config)
    else:
        raise ValueError(
            f"config has no _target_ and doesn't match known model schemas; "
            f"top-level keys: {sorted(config.keys())}"
        )
    model.load_state_dict(state_dict)
    return model


def _build_prejepa_from_training_config(cfg: dict) -> torch.nn.Module:
    """Reconstruct a PreJEPA module from a training-cfg dump.

    Mirrors the model assembly in scripts/train/prejepa.py:run() — frozen
    backbone via EvalOnly, CausalPredictor, optional extra_encoders (action,
    proprio, ...). Used when save_pretrained wrote the full training cfg
    instead of a Hydra-instantiable model spec.
    """
    from collections import OrderedDict

    import stable_pretraining as spt
    import torch.nn as nn
    from transformers import AutoModel, AutoModelForImageClassification

    import stable_worldmodel as swm

    # --- backbone ---
    backbone_name = cfg['backbone']['name']
    if backbone_name.startswith('microsoft/resnet-'):
        backbone = AutoModelForImageClassification.from_pretrained(backbone_name)
        embed_dim = backbone.config.hidden_sizes[-1]
        backbone.classifier[1] = nn.LayerNorm(embed_dim)
        num_patches = 1
        interp_pos_enc = False
    else:
        backbone = AutoModel.from_pretrained(backbone_name)
        if hasattr(backbone, 'vision_model'):
            backbone = backbone.vision_model
        embed_dim = backbone.config.hidden_size
        num_patches = (cfg['image_size'] // cfg['patch_size']) ** 2
        # DINOv3 uses RoPE positions (no interpolate_pos_encoding kwarg) and
        # prepends register tokens after CLS; mirror get_encoder in
        # scripts/train/prejepa.py.
        interp_pos_enc = backbone.config.model_type != 'dinov3_vit'

    # extra encodings widen the per-token embedding dim
    extra_enc_cfg = cfg.get('wm', {}).get('encoding', {}) or {}
    embed_dim_full = embed_dim + sum(extra_enc_cfg.values())

    # --- predictor ---
    predictor_kwargs = {
        k: v for k, v in cfg['predictor'].items() if k != 'size'
    }
    predictor = swm.wm.prejepa.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg['wm']['history_size'],
        dim=embed_dim_full,
        **predictor_kwargs,
    )

    # --- extra encoders (action embedder, optionally proprio) ---
    extra_dims = cfg.get('extra_dims', {}) or {}
    extra_encoders = nn.ModuleDict(
        OrderedDict(
            (
                key,
                swm.wm.prejepa.Embedder(
                    in_chans=extra_dims.get(key, emb_dim_default),
                    emb_dim=emb_dim_default,
                ),
            )
            for key, emb_dim_default in extra_enc_cfg.items()
        )
    )

    model = swm.wm.prejepa.PreJEPA(
        encoder=spt.backbone.EvalOnly(backbone),
        predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=cfg['wm']['history_size'],
        num_pred=cfg['wm']['num_preds'],
        interpolate_pos_encoding=interp_pos_enc,
        num_prefix_tokens=1 + getattr(backbone.config, 'num_register_tokens', 0),
    )
    return model


def _resolve(name: str, cache_dir: Path) -> tuple[Path, dict]:
    """Return ``(checkpoint_path, config_dict)`` for *name*.

    Resolution order:
      1. ``<cache_dir>/<name>``  as a ``.pt`` file
      2. ``<cache_dir>/<name>``  as a folder
      3. HuggingFace repo (cached locally under ``<cache_dir>/<user>/<repo>/``)
    """
    local = cache_dir / name

    # format 1: explicit .pt file
    if local.suffix == '.pt':
        if not local.exists():
            raise FileNotFoundError(f'Checkpoint not found: {local}')
        return local, _load_config(local.parent)

    # format 2: folder containing a .pt and config.json
    if local.is_dir():
        return _resolve_folder(local)

    # format 3: HuggingFace repo (<user>/<repo>)
    if '/' in name:
        return _resolve_hf(name, cache_dir)

    raise ValueError(
        f"Cannot resolve '{name}': not a .pt file, a folder, or a HF repo id."
    )


def _resolve_folder(folder: Path) -> tuple[Path, dict]:
    """Load from a folder containing one ``.pt`` file and a ``config.json``."""
    pt_files = list(folder.glob('*.pt'))
    if not pt_files:
        raise FileNotFoundError(f'No .pt file found in {folder}')
    if len(pt_files) > 1:
        raise ValueError(
            f'Ambiguous checkpoint: multiple .pt files in {folder}. '
            'Specify the file directly.'
        )
    logging.info(f'Loading checkpoint from folder {folder}...')
    return pt_files[0], _load_config(folder)


def _resolve_hf(repo_id: str, cache_dir: Path) -> tuple[Path, dict]:
    """Resolve a HuggingFace repo id, using a local cache when available.

    Local layout: ``<cache_dir>/models--<user>--<repo>/``
    """
    local_dir = cache_dir / f'models--{repo_id.replace("/", "--")}'

    if local_dir.is_dir():
        logging.info(f'Loading {repo_id} from local cache...')
        return _resolve_folder(local_dir)

    logging.info(f'Downloading {repo_id} from HuggingFace...')
    local_dir.mkdir(parents=True, exist_ok=True)
    for filename in ('config.json', 'weights.pt'):
        url = f'{HF_BASE_URL}/{repo_id}/resolve/main/{filename}'
        dest = local_dir / filename
        logging.info(f'Fetching {url}')
        _download(url, dest)

    return _resolve_folder(local_dir)


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a tqdm progress bar."""
    response = urllib.request.urlopen(url)
    total = int(response.headers.get('Content-Length', 0)) or None
    with (
        open(dest, 'wb') as f,
        tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as bar,
    ):
        while chunk := response.read(8192):
            f.write(chunk)
            bar.update(len(chunk))


def _load_config(folder: Path) -> dict:
    config_path = folder / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'config.json not found in {folder}')
    with open(config_path) as f:
        return json.load(f)


__all__ = ['load_pretrained', 'save_pretrained']
