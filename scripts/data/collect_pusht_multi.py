"""Data collection for the multi-object PushT environment.

Driven entirely by hydra config (`scripts/data/config/pusht_multi.yaml`).
A single config switch picks which compositional subset to enable
(via `enabled_objects` and the corresponding `obj.<oid>.enabled`
variation-values overrides), so RQ1 (count) and RQ2 (pair-comp) splits
share the same script.

Each invocation produces a single HDF5 file at
`<cache>/datasets/<dataset_name>.h5`, matching the layout of the other
single-file datasets (e.g. `cube_single_expert.h5`). `record_dataset`
supports resuming, so re-running with the same `dataset_name` continues
where a previous run left off.

Example:
    python scripts/data/collect_pusht_multi.py \\
        dataset_name=pusht_multi_AB \\
        enabled_objects='[A,B]' \\
        num_traj=10000
"""

import hydra
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht_multi import (
    MultiObjectGoalPolicy,
    MultiObjectWeakPolicy,
)


def _build_options(cfg) -> dict:
    """Translate config-level `enabled_objects` into reset-time options.

    The env reads `obj.<oid>.enabled` from the variation space; this
    function packages those plus the default position/angle variations
    into the `options` dict that `world.record_dataset` forwards through
    to `env.reset(options=...)`.

    In `subset_mode=random`, per-episode enabled flags are overridden
    inside the env's reset and the values supplied here are ignored,
    so we still emit them harmlessly — keeps the H5 schema identical.
    """
    objects = tuple(cfg.objects)
    enabled = set(cfg.enabled_objects)
    for oid in enabled:
        if oid not in objects:
            raise ValueError(
                f'enabled_objects contains {oid!r} which is not in objects={objects}'
            )

    variation_keys = [
        'agent.start_position',
        *[f'obj.{oid}.start_position' for oid in objects],
        *[f'obj.{oid}.angle' for oid in objects],
        *[f'obj.{oid}.goal_position' for oid in objects],
        *[f'obj.{oid}.goal_angle' for oid in objects],
        *[f'obj.{oid}.enabled' for oid in objects],
    ]
    variation_values = {
        f'obj.{oid}.enabled': int(oid in enabled) for oid in objects
    }
    return {
        'variation': variation_keys,
        'variation_values': variation_values,
    }


def _build_policy(cfg):
    kind = cfg.policy.get('kind', 'weak')
    if kind == 'weak':
        return MultiObjectWeakPolicy(
            dist_constraint=cfg.policy.dist_constraint,
            switch_every=cfg.policy.switch_every,
            p_wedge=cfg.policy.p_wedge,
            seed=cfg.seed,
        )
    if kind == 'expert':
        return MultiObjectGoalPolicy(
            push_distance=cfg.policy.push_distance,
            noise_std=cfg.policy.noise_std,
            switch_every=cfg.policy.expert_switch_every,
            switch_pos_tol=cfg.policy.switch_pos_tol,
            seed=cfg.seed,
        )
    raise ValueError(f'Unknown policy kind {kind!r}; expected weak|expert')


@hydra.main(
    version_base=None, config_path='./config', config_name='pusht_multi'
)
def run(cfg):
    objects = tuple(cfg.objects)
    enabled = tuple(cfg.enabled_objects)
    subset_mode = cfg.get('subset_mode', 'fixed')
    subset_k_range = tuple(cfg.get('subset_k_range', (1, len(objects))))
    logging.info(
        f'PushTMulti collection: dataset_name={cfg.dataset_name}, '
        f'objects={objects}, enabled={enabled}, '
        f'subset_mode={subset_mode}, episodes={cfg.num_traj}, '
        f'policy={cfg.policy.get("kind", "weak")}'
    )

    world = swm.World(
        'swm/PushTMulti-v1',
        objects=list(objects),
        enabled_objects=list(enabled),
        subset_mode=subset_mode,
        subset_k_range=list(subset_k_range),
        **cfg.world,
        render_mode='rgb_array',
    )
    world.set_policy(_build_policy(cfg))

    world.record_dataset(
        cfg.dataset_name,
        episodes=cfg.num_traj,
        seed=cfg.seed,
        cache_dir=cfg.cache_dir,
        options=_build_options(cfg),
    )

    logging.success(f' 🎉 Done collecting {cfg.dataset_name}')


if __name__ == '__main__':
    run()
