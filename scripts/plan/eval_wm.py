"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import os

os.environ['MUJOCO_GL'] = 'egl'

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def img_transform(cfg):
    # DINO-WM's training pipeline normalizes pixels to [-1, 1] via
    # Normalize(mean=0.5, std=0.5) BEFORE feeding to the (frozen) DINOv2
    # encoder. The predictor was therefore trained on the feature
    # distribution DINOv2 produces from [-1, 1]-normalized inputs — not
    # from ImageNet-stats inputs. Match that to avoid feature distribution
    # shift at planning time. PreJEPA / SWM-native models trained with
    # ImageNet stats should override via cfg.eval.normalize='imagenet'.
    norm = cfg.eval.get('normalize', 'imagenet')
    if cfg.get('policy_kind') == 'dino_wm_external':
        norm = cfg.eval.get('normalize', 'half')

    if norm == 'half':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif norm == 'imagenet':
        normalize = transforms.Normalize(**spt.data.dataset_stats.ImageNet)
    else:
        raise ValueError(f'unknown normalize: {norm!r}')

    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            normalize,
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        'pixels': img_transform(cfg),
        'goal': img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    ep_indices, _ = np.unique(
        stats_dataset.get_col_data(col_name), return_index=True
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # -- run evaluation
    policy = cfg.get('policy', 'random')

    if policy != 'random':
        if cfg.get('policy_kind') == 'dino_wm_external':
            from stable_worldmodel.wm.dino_wm_external import (
                load_dino_wm_external,
            )

            model = load_dino_wm_external(
                cfg.policy,
                dino_wm_src=cfg.get(
                    'dino_wm_src', '/home/nazirjon/Desktop/dino_wm'
                ),
                alpha=cfg.get('dino_wm_alpha', 1.0),
                rollout_chunk=cfg.get('dino_wm_rollout_chunk', 64),
            )
        else:
            model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        # If the model carries DINO-WM training-time normalization constants
        # (`dw_*_mean/std`, set by load_dino_wm_external), override the
        # sklearn-fit StandardScalers so what's fed to the model AND what's
        # denormalized into env-units exactly matches the training distribution.
        # The sklearn fit on the full h5 gives stats that diverge from
        # DINO-WM's hardcoded constants — most importantly, proprio dim-1
        # mean is 298 (sklearn) vs 264 (DINO-WM), a 34-unit shift larger than
        # the 20-unit success threshold.
        if getattr(model, 'dw_action_mean', None) is not None:
            if 'action' in process:
                process['action'].mean_ = model.dw_action_mean
                process['action'].scale_ = model.dw_action_std
                process['action'].var_ = model.dw_action_std ** 2
            if 'proprio' in process:
                process['proprio'].mean_ = model.dw_proprio_mean
                process['proprio'].scale_ = model.dw_proprio_std
                process['proprio'].var_ = model.dw_proprio_std ** 2
            if 'goal_proprio' in process:
                process['goal_proprio'].mean_ = model.dw_proprio_mean
                process['goal_proprio'].scale_ = model.dw_proprio_std
                process['goal_proprio'].var_ = model.dw_proprio_std ** 2
            print('[eval_wm] Overrode process[action/proprio/goal_proprio] '
                  'with DINO-WM training-time stats from adapter.')

        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
        if cfg.policy != 'random'
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)['step_idx']

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    world.set_policy(policy)

    variation_overrides = cfg.eval.get('variation_overrides')
    if variation_overrides is not None:
        variation_overrides = OmegaConf.to_container(
            variation_overrides, resolve=True
        )
        # YAML overrides come in as lists; the env's variation_space stores
        # numpy arrays (with specific dtypes — RGBBox is uint8, position boxes
        # are float64, etc.), and Box.contains is dtype-strict, so coerce each
        # list value to the matching space's dtype before passing through.
        vv = variation_overrides.get('variation_values') or {}
        for k, v in vv.items():
            if isinstance(v, list):
                space = world.single_variation_space
                for part in k.split('.'):
                    space = space[part]
                vv[k] = np.array(v, dtype=space.dtype)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(
            cfg.eval.get('callables'), resolve=True
        ),
        video_path=results_path,
        variation_overrides=variation_overrides,
        stop_on_success=cfg.eval.get('stop_on_success', False),
    )
    end_time = time.time()

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()
