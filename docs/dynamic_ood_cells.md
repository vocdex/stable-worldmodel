# Dynamic OOD cells (DCS-style) — handoff for running on CJEPA

Audience: the session that will run SlotContrast-WM / LeWM through the cjepa
planning pipeline on the new PushT perturbation cells. Everything here lives
on the stable-worldmodel branch **`ood-dynamic-distractors`** (cut from
`pusht-multi`; contains all OOD eval fixes incl. the cube goal-frame fix
`1f552ef`). Use this branch for the SWM checkout that cjepa imports.

![cells](assets/ood_dynamic_cells.png)

Rows: baseline, static distractor (legacy cell), **distractor_moving**,
**bg_natural_static**, **bg_video_dynamic**. Columns: env steps t=0..12 under
zero actions + the goal frame the planner receives. Note the dynamic cells
change *during* the rollout while the goal frame stays frozen.

## What changed in the PushT env (`stable_worldmodel/envs/pusht/env.py`)

Two new variation leaves; **defaults reproduce the legacy env byte-for-byte**:

1. **`distractor.motion`** — Box `(amplitude_px, period_steps)`, init `(0, 40)`.
   The (visual-only, collision-free) distractor moves on a circle of radius
   `amplitude` around `distractor.position`, advancing one phase step per env
   step. Cosine phase ⇒ the amplitude already offsets the distractor at t=0,
   so start/goal frames reflect the cell. `amplitude=0` = legacy static.

2. **`background.texture_id`** — Discrete 0..16, init 0 (= plain background
   color). A value k≥1 selects the k-th sorted entry of the texture dir:
   an image **file** is a static background; a **directory** of frames is a
   clip advancing one frame per env step (looping). Drawn behind
   goal-marker/block/agent. A missing texture **raises** (no silent no-op).

Texture dir resolution: `texture_dir` env kwarg → `$SWM_TEXTURE_DIR` → swm
cache `~/.stable_worldmodel/textures/`. Populate once with:

```bash
python scripts/data/fetch_textures.py            # in stable-worldmodel
# 1: 01_mountains.jpg  2: 02_forest.jpg  3: 03_city.jpg   (static)
# 4: 04_cockatoo/      5: 05_newtonscradle/                (clips)
```

## Canonical cell definitions

All cells are **single-factor** (decision 2026-06-11) so the same definition
runs unmodified through cjepa's single-factor override runner and the SWM
launcher:

| cell | variation_values |
|---|---|
| `distractor_moving` | `distractor.motion=[40,20]` (color/scale/position stay at the defaults: gray, 20, [80,80]) |
| `bg_natural_static` | `background.texture_id=1` |
| `bg_video_dynamic`  | `background.texture_id=4` |

(SWM-side launcher: `scripts/cluster/plan_ood/pusht.sh`, cells 10–12.)

## Running through cjepa

- All three cells run through the existing single-factor runner as-is:
  `variation.factor=distractor.motion variation.value=[40,20]`, resp.
  `variation.factor=background.texture_id variation.value=1` (or `4`).
  `background.texture_id` is an int — make sure the value is passed as an
  int, not a list (the runner's uint8-array coercion must not apply).
  Varying any `distractor.*` leaf enables the distractor; the unvaried
  leaves keep their defaults.
- Set `SWM_TEXTURE_DIR` in the job env and run `fetch_textures.py` once on
  the cluster before the `bg_*` cells.
- Export `SWM_EVAL_TRUST_CHECKS=1`: `evaluate_from_dataset` then verifies at
  reset that the planner's goal frame corresponds to the H5 goal state and
  that the success target matches the H5 (the cube-bug class fails loudly
  instead of producing plausible-but-wrong SRs). cjepa's pusht config already
  has the `_set_goal_state` callable, so the checks pass on a correct setup.

## Protocol notes

- **Frozen goal frame**: in `bg_video_dynamic`, the goal frame uses the clip
  frame current at goal-render time (t=0). The background then moves during
  the rollout while the goal background stays fixed — that mismatch is the
  point of the cell (task-irrelevant dynamic content), same as DCS.
- **Before interpreting model SRs**: run a random-policy floor for each new
  cell, and eyeball 2–3 rollout videos (goal panel vs live panel).
- Goal-row semantic: the eval goal is the state `goal_offset_steps − 1` steps
  after the start (chunk end is exclusive) — pinned by
  `tests/eval/test_ood_trust_chain.py::goal_row`.

## Sanity checks after switching the SWM checkout

```bash
uv run pytest tests/eval tests/envs/test_pusht_distractor_motion.py \
    tests/envs/test_pusht_background_texture.py -q     # ~10 s, all green
```

The trust suite's T6 sweep auto-covers every variation leaf, so if your SWM
checkout is on the wrong branch (no texture support) or the texture dir is
missing, these tests say so immediately.
