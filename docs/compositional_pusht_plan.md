# Compositional Generalization in Object-Centric World Models — Plan

Research plan for the `pusht-multi` branch. Adds a multi-object PushT
environment with per-object segmentation to test whether an
object-centric world model exhibits compositional generalization when
the encoder is held fixed.

**Scope**: only the object-centric pipeline (SlotContrast encoder +
slot-WM). No patch-encoder (DINOv2) baseline. The encoder is trained
once on the full visual diversity and held fixed; the world model is the
only variable.

**Slot accounting.** In an object-centric setting the *agent* and the
*background* are also entities the encoder must bind a slot to, so a
scene with `k` manipulable objects actually requires `k + 2` slots
(agent + bg + each manipulable). Throughout this doc `k` refers to the
number of non-agent, non-background objects:

| Split   | k | Slots needed              |
|---------|---|---------------------------|
| AB      | 2 | agent, bg, A, B           |
| BC      | 2 | agent, bg, B, C           |
| ABC     | 3 | agent, bg, A, B, C        |
| k=6     | 6 | agent, bg, A, B, C, D, E, F |

SlotContrast is trained with `num_slots = 8` (= max k + 2) for headroom
across every split.

---

## 1. Research questions

### RQ1 — Object-count generalization

> SlotContrast is trained on scenes containing the full library of N=6
> objects + background. The world model is trained only on scenes with
> `k < N` objects. Test the WM on scenes with up to N objects. At what
> training `k` (if any) does compositional generalization to N objects
> emerge?

### RQ2 — Pairwise compositional generalization

> The encoder and the world model are both exposed only to pairs
> `{A,B}` and `{B,C}` — never the triple `{A,B,C}`. Does the WM
> generalize to the held-out triple?

---

## 2. Hypotheses

| H  | Statement                                                                                                | Predicted result                                                                              |
|----|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| H1 | A slot-WM trained on small `k` cannot extrapolate to large N.                                            | Joint-success drops monotonically as N − k_train grows.                                       |
| H2 | Compositionality, if it emerges, does so at a threshold `k*`.                                            | Success at N saturates once `k_train ≥ k*`, where `k* < N`.                                   |
| H3 | A slot-WM trained on `{A,B} ∪ {B,C}` partially generalizes to `{A,B,C}` — better than `{A,B}`-only, worse than `{A,B,C}`-oracle. | Ordered: AB-only < AB∪BC < ABC-oracle.                                       |
| H4 | Per-object position error is roughly flat in N for the slot-WM (consistent with per-slot dynamics).      | Per-object error roughly constant across k=1..N; joint error grows due to compounding.        |

---

## 3. Environment — `swm/PushTMulti-v1`

New env, no edits to existing `swm/PushT-v1`. Lives under
`stable_worldmodel/envs/pusht_multi/`.

### 3.1 Canonical object library — *as implemented*

Each named identity has a fixed segmentation label so identity↔label is
stable across all splits (needed so per-slot analysis is comparable
between `{A,B}`, `{B,C}`, `{A,B,C}`).

```
LABEL_BG    = 0
LABEL_AGENT = 1

OBJECT_LIBRARY = {
    'A': {shape: 'T',      color: LightSlateGray, scale: 20, mass: 1.0, friction: 1.0, has_orientation: True, label: 2},
    'B': {shape: 'I',      color: Orange,         scale: 30, mass: 1.5, friction: 0.8, has_orientation: True, label: 3},
    'C': {shape: 'Z',      color: SeaGreen,       scale: 30, mass: 0.5, friction: 0.3, has_orientation: True, label: 4},
    'D': {shape: 'square', color: Purple,         scale: 30, mass: 2.5, friction: 1.5, has_orientation: True, label: 5},
    'E': {shape: '+',      color: Crimson,        scale: 30, mass: 0.8, friction: 1.2, has_orientation: True, label: 6},
    'F': {shape: 'L',      color: Gold,           scale: 30, mass: 1.2, friction: 0.6, has_orientation: True, label: 7},
}
```

All identities are non-circular polygons — the agent is the only circle
in any scene, so the pusher is always visually unique. Each identity
has a distinct `(mass, friction)` signature, so the WM has to bind
**dynamics** to identity, not just appearance. `ObjectSpec` also exposes
a `bounding_radius` property (rotation-invariant circumscribed-circle
radius computed from the shape's polygon vertices) used by the
non-overlap sampler. A's default scale is 20 (rather than the original
PushT default of 30) — at scale 30 its stem ran 120 px long and
dominated all multi-object placement.

### 3.2 Constructor

```python
PushTMulti(
    objects=('A','B','C','D','E','F'),   # available identities
    enabled_objects=None,                # active this episode (default: all)
    success_pos_tol=20,
    success_angle_tol=np.pi/9,
    resolution=224,
    relative=True,
    render_mode='rgb_array',
    init_value=None,
)
```

### 3.3 Variation space

Top-level keys: `agent`, `obj`, `background`, `rendering`. The
per-object sub-Dicts live **nested under** `obj`, so the swm dot-path
traversal resolves `obj.A.start_position` → `space['obj']['A']['start_position']`.

```
agent:
  color, scale, angle, start_position, velocity
obj.<oid>:
  enabled (Discrete 2), color, scale, angle,
  start_position, goal_position, goal_angle
background:
  color
rendering:
  render_goal  (Discrete 2, init_value=0)
```

`render_goal` defaults to **0** — the live RGB does *not* render a
faded goal outline. The goal is delivered through `info['goal']` only.
Drawing a ghost in the live obs would create static object-shaped
regions that have no segmentation supervision and would eat slot
capacity in an object-centric encoder.

**Per-shape spawn boxes.** `start_position` and `goal_position` bounds
are computed from each shape's `bounding_radius` so the body fits
inside the walls (`[5, 506]`) with a 10 px margin without help from
the sampler:

```python
low  = 5    + bounding_radius + 10
high = 506  - bounding_radius - 10
```

So A (T, r≈81) spawns in a tight central box, F (L, r≈67) in a
slightly wider one, and the agent (r=22.5 at max scale) anywhere.

**Sampling order**: `[background, obj, agent, rendering]`. Inside
`obj`: each canonical id in order. Default reset variations resample
`agent.start_position`, plus per-object `start_position`, `angle`,
`goal_position`, `goal_angle`. The per-object `enabled` flags are *not*
in the default list — they come from `options['variation_values']` per
collection script.

### 3.4 Physics — *as implemented*

- `space.damping = 0.1`. Pymunk semantics: retention-per-second, so
  per env-step (~0.1 s of sim) bodies lose ~20% of their velocity.
  Without this, objects coast forever (original PushT runs damping=0
  but only has one heavy block, so it's not noticeable).
- Per-identity `mass` and `friction` plumb into all polygon
  constructors.
- **Constructor invariant**: order is `body.center_of_gravity →
  body.angle → body.position`. For shapes with non-trivial COG (T, L,
  Z, +, small_tee) any other order makes `body.position` read back at
  a different world location than was set (pymunk rotates around COG,
  not local origin). Original PushT's `_set_goal_state` docstring warns
  about this; we apply the same discipline at construction time too.

### 3.5 Info schema (fixed keys, NaN for disabled)

```
info = {
    'env_name': 'PushTMulti',
    'pos_agent', 'vel_agent',
    'pose.<oid>':      (3,)   for each oid in `objects`,   NaN if disabled
    'goal_pose.<oid>': (3,)   for each oid in `objects`,   NaN if disabled
    'enabled.<oid>':   bool   for each oid in `objects`,
    'segmentation':       (H,W) uint8,
    'goal_segmentation':  (H,W) uint8,
    'goal':               (H,W,3) uint8,
    'goal_state', 'goal_proprio', 'n_contacts',
    'pixels': added by MegaWrapper,
}
```

Fixed keys with NaN sentinels keep the H5 schema identical across all
splits, so one dataloader iterates any of them.

### 3.6 Per-object segmentation (analytic rasterization)

`_render_segmentation()` returns `(H, W)` uint8:

1. 8-bit palette `pygame.Surface` filled with `LABEL_BG`.
2. Rasterize the agent first (label 1), then each enabled object body.
   `pymunk.Poly` → world-space vertices via `body.local_to_world` →
   `pygame.draw.polygon(surface, label, points)`.
   `pymunk.Circle` (agent) → `pygame.draw.circle(surface, label,
   center, radius)`.
3. `cv2.resize(..., interpolation=cv2.INTER_NEAREST)` to `render_size`.

`goal_segmentation` uses the same path inside the goal-frame
snapshot/restore.

**Render hygiene**: `space.debug_draw` is restricted to
`DRAW_SHAPES` only (no collision-point or constraint overlays in
either the live obs or the goal frame).

### 3.7 Goal handling

Per-object goal pose `(x, y, θ)`. `θ` is ignored in the success
criterion for objects with `has_orientation=False` (currently no such
object since C is a Z). The goal frame shows the walls, the agent at
the goal-state agent position, and every enabled object at its goal
pose — same render path as the live frame.

`_render_goal_frame` extends the single-object idempotent
render-then-restore pattern to N bodies. Discipline: snapshot all
bodies' pose+velocity, set each body's angle BEFORE position,
`space.reindex_shapes_for_body(body)` on every moved body, render,
restore in reverse order. The goal frame is rendered **before** any
physics step so it can't pick up cached contact info.

### 3.8 Non-overlap placement

Sequential placement, not joint rejection. Sort participating entities
by `bounding_radius` descending; for each, sample its position
(uniformly within its per-shape spawn box) until it's at least
`r_self + r_other + margin (8 px)` from every previously placed
entity. Per-entity cap of 500 attempts; if exhausted, accept and let
pymunk's first-step solver push penetrating bodies apart.

- `start_position`: the agent participates (so the pusher can't spawn
  inside an object).
- `goal_position`: object-only (agent has no goal pose).

Verified clean across 50 seeds for both `{A,B,C}` and `{A..F}`.

### 3.9 Success / reward / termination

```python
pos_ok[oid]   = ||pos[oid] - goal_pos[oid]|| < pos_tol            # 20 px
angle_ok[oid] = angle_diff(angle[oid], goal_angle[oid]) < angle_tol  # π/9
                  # skipped if `has_orientation=False`
obj_success[oid]   = pos_ok[oid] and angle_ok[oid]

joint_success    = all(obj_success[oid] for oid in enabled)   # primary, terminates
mean_obj_success = mean(obj_success[oid] for oid in enabled)  # secondary
mean_pos_error   = mean(||pos - goal_pos||)
mean_angle_error = mean(angle_diff)

reward     = -sum(pos_dist + angle_dist*20 over enabled)      # smooth
terminated = joint_success
```

Tolerances do **not** scale with N — keeps numbers comparable to
existing single-object PushT.

### 3.10 Episode budget

`max_episode_steps = 60 + 60 * k`. Single PushT uses 100; k=6 → 420.

---

## 4. Data-collection policy — `MultiObjectWeakPolicy`

We do **not** have a goal-solving expert. Like the existing PushT
pipeline, data is collected with a *weak coverage policy* —
deliberately random with a bias toward object contact — and goals are
satisfied by the planner at evaluation time (CEM/MPPI over the WM),
not at collection time. The "expert" name is inherited from the
existing single-object PushT file naming (`expert_policy.py`).

Mechanism, generalizing `WeakPolicy`:

- Every `switch_every` steps, pick a *focus*. With prob `1 - p_wedge`,
  focus = a uniformly chosen enabled object. With prob `p_wedge`,
  focus = midpoint between two enabled objects (drives pairwise
  contact sampling — important for RQ2).
- Sample uniform random action, scale to env units, clip to a
  `dist_constraint`-sided square neighborhood around the focus point.

For ablation purposes a `MultiObjectGoalPolicy` (per-object PD toward
goal in turn) is easy to add later, but isn't required for the
compositional generalization experiment.

---

## 5. Datasets

### 5.1 SlotContrast training (shared across all experiments)

| Name                 | Enabled            | k       | Episodes | Used for       |
|----------------------|--------------------|---------|----------|----------------|
| `pusht_multi_sc`     | random subset of {A..F} per episode, all k from 1..6 | mixed | 30 000 | SC training only |

One pretrained SC checkpoint is used for every WM experiment below. SC
sees full visual diversity (all 6 identities, all scene complexities,
all subset compositions). Per-episode `enabled_objects` is sampled
uniformly so the encoder is never the bottleneck.

### 5.2 RQ1 — object-count

| Name              | Enabled               | k | Episodes | Used for         |
|-------------------|-----------------------|---|----------|------------------|
| `pusht_multi_k1`  | random 1 of {A..F}    | 1 | 10 000   | WM train         |
| `pusht_multi_k2`  | random 2 of {A..F}    | 2 | 10 000   | WM train         |
| `pusht_multi_k3`  | random 3 of {A..F}    | 3 | 10 000   | WM train         |
| `pusht_multi_k4`  | random 4 of {A..F}    | 4 | 10 000   | WM train         |
| `pusht_multi_k5`  | random 5 of {A..F}    | 5 | 10 000   | WM train         |
| `pusht_multi_k6`  | all 6                 | 6 | 10 000   | WM oracle + eval |

Train one WM per `k_train ∈ {1,2,3,4,5}` on `pusht_multi_k{k_train}`
plus one oracle on `pusht_multi_k6`. Evaluate every WM on
`pusht_multi_k6` (held-out episodes). Read off the curve.

### 5.3 RQ2 — pair-composition

| Name                       | Enabled  | k | Episodes | Used for           |
|----------------------------|----------|---|----------|--------------------|
| `pusht_multi_AB`           | A, B     | 2 | 10 000   | WM train           |
| `pusht_multi_BC`           | B, C     | 2 | 10 000   | WM train           |
| `pusht_multi_ABC_oracle`   | A, B, C  | 3 | 10 000   | WM oracle          |
| `pusht_multi_ABC_eval`     | A, B, C  | 3 |    500   | held-out eval      |

Three WMs:
- WM_AB         — trained on `pusht_multi_AB` only.
- WM_AB∪BC      — trained on `pusht_multi_AB ∪ pusht_multi_BC`.
- WM_ABC_oracle — trained on `pusht_multi_ABC_oracle`.

All evaluated on `pusht_multi_ABC_eval`.

All datasets collected via `scripts/data/collect_pusht_multi.py`
parameterized by an `enabled_objects` config (mirrors
`collect_weak_pusht.py`).

---

## 6. Branch & file layout — *as implemented*

Branch: `pusht-multi`.

```
stable_worldmodel/envs/pusht_multi/
    __init__.py              # PushTMulti, MultiObjectWeakPolicy, OBJECT_LIBRARY
    env.py                   # PushTMulti gym.Env
    objects.py               # canonical identity registry
    expert_policy.py         # MultiObjectWeakPolicy
stable_worldmodel/envs/__init__.py             # registers swm/PushTMulti-v1
scripts/data/collect_pusht_multi.py
scripts/data/config/pusht_multi.yaml
scripts/visualization/visualize_pusht_multi.py # rollout-to-MP4 sanity check
tests/envs/test_pusht_multi.py                 # 11 invariant tests
docs/compositional_pusht_plan.md               # this file
```

`swm/PushT-v1` and the original `WeakPolicy` are untouched, so cjepa
keeps building unchanged.

---

## 7. Implementation status

All steps below are landed on `pusht-multi`. The corresponding commit
history is on the branch.

1. ✅ `objects.py` library + per-identity mass/friction/bounding_radius.
2. ✅ `env.py` skeleton with single-object parity check.
3. ✅ Multi-object reset with sequential non-overlap placement.
4. ✅ `_render_segmentation()` and `_render_goal_frame` extended to N bodies.
5. ✅ Per-object success / reward; `info` schema with NaN sentinels.
6. ✅ `MultiObjectWeakPolicy` with wedge mode.
7. ✅ `collect_pusht_multi.py` + hydra config (`scripts/data/config/pusht_multi.yaml`).
8. ✅ Smoke tests: 11 invariants in `tests/envs/test_pusht_multi.py`
   covering label stability across subsets, segmentation cleanliness,
   info-schema identity across splits, bounding-radius non-overlap.

---

## 8. Evaluation & metrics

For each WM × training-set combination, report on the appropriate
held-out eval set:

- **Joint success rate** (primary).
- **Per-object success rate**.
- **Mean position error, mean angle error**.
- **Slot-stability** (qualitative): fraction of timesteps where a
  slot's argmax label is consistent within an episode — flags whether
  failures are slot-binding failures or dynamics-prediction failures.

Headline plots:

- **RQ1**: x = `k_train`, y = joint success at k=6. Curve with oracle
  asymptote. Look for the knee — that's the emergence point `k*`.
- **RQ2**: bar chart with three bars (WM_AB, WM_AB∪BC, WM_ABC_oracle),
  evaluated on `pusht_multi_ABC_eval`. Gap between WM_AB∪BC and oracle
  is the "compositional deficit".

---

## 9. Known risks / things to watch

- **Pairwise interaction frequency**: if the weak policy rarely
  produces A↔B contacts, RQ2 collapses into the trivially-factorizable
  regime. Mitigation: wedge mode (`p_wedge=0.3` default) + check
  per-shard contact-frequency stats from `info['n_contacts']` before
  training a WM.
- **Goal-frame variation overrides**: PushT's `_set_goal_state` is
  delicate (angle-before-position, reindex). Preserved in the N-body
  extension (see `_render_goal_frame`).
- **Render cost**: an extra segmentation pass per step. Fine for N≤6;
  gate behind a flag if profiling shows it dominating.
- **Damping divergence**: `space.damping = 0.1` (vs original
  PushT's 0). Don't backport.
- **No expert policy**: data is collected with a weak random policy.
  If the SC encoder or WM has trouble distinguishing identities
  because the weak policy under-samples interesting object motions,
  consider adding `MultiObjectGoalPolicy` (per-object PD toward goal)
  to mix in.
