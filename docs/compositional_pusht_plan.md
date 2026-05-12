# Compositional Generalization in Object-Centric World Models — Plan

Research plan for the `pusht-multi` branch. Adds a multi-object PushT
environment with per-object segmentation to test whether an
object-centric world model exhibits compositional generalization when
the encoder is held fixed.

**Scope**: only the object-centric pipeline (SlotContrast encoder +
slot-WM). No patch-encoder (DINOv2) baseline. The encoder is trained
once on the full visual diversity and held fixed; the world model is the
only variable.

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

### 3.1 Canonical object library

Each named identity has a fixed segmentation label so identity↔label is
stable across all splits (needed so per-slot analysis is comparable
between `{A,B}`, `{B,C}`, `{A,B,C}`).

```
LABEL_BG = 0
LABEL_AGENT = 1

OBJECT_LIBRARY = {
    'A': {shape: 'T',      color: LightSlateGray, scale: 30, mass: 1.0, friction: 1.0, has_orientation: True,  label: 2},
    'B': {shape: 'I',      color: Orange,         scale: 30, mass: 1.0, friction: 1.0, has_orientation: True,  label: 3},
    'C': {shape: 'o',      color: SeaGreen,       scale: 30, mass: 0.5, friction: 0.3, has_orientation: False, label: 4},
    'D': {shape: 'square', color: Purple,         scale: 30, mass: 2.0, friction: 1.5, has_orientation: True,  label: 5},
    'E': {shape: '+',      color: Crimson,        scale: 30, mass: 1.0, friction: 1.0, has_orientation: True,  label: 6},
    'F': {shape: 'L',      color: RoyalBlue,      scale: 30, mass: 1.0, friction: 1.0, has_orientation: True,  label: 7},
}
```

Mass/friction (currently hardcoded `mass=1, friction=1` in the existing
`add_*` constructors) are plumbed through so dynamics — not just
appearance — vary by identity. This is what makes "the WM has to bind
dynamics to identity" non-trivial.

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

- `agent`: same as today.
- For each `oid in objects`, a sub-Dict `obj.<oid>` with: `enabled`
  (Discrete 2), `color`, `scale`, `start_position`, `angle`,
  `goal_position`, `goal_angle`.
- `background.color`, `rendering.render_goal`.
- Sampling order: `[background, obj.<oid>..., agent, rendering]`.
- **Non-overlap rejection** in `reset()` for both start and goal poses
  (min pairwise separation ≥ 1.5 × max scale).

### 3.4 Physics

- `space.damping = 0.85` (currently 0). Without damping, un-contacted
  objects coast forever, which complicates multi-object trajectories
  without making the experiment more meaningful.
- Per-identity `mass` and `friction` flow into `add_*`.

### 3.5 Info schema (fixed keys, NaN for disabled)

```
info = {
    'env_name': 'PushTMulti',
    'pos_agent', 'vel_agent',
    'pose.<oid>': (3,)        for each oid in objects,    # NaN if disabled
    'goal_pose.<oid>': (3,)   for each oid in objects,    # NaN if disabled
    'enabled.<oid>': bool     for each oid in objects,
    'segmentation': (H,W) uint8,
    'goal_segmentation': (H,W) uint8,
    'goal': (H,W,3) uint8,
    'pixels': added by MegaWrapper,
}
```

Fixed keys with NaN sentinels keep the H5 schema identical across all
subsets, so one dataloader iterates any of them.

### 3.6 Per-object segmentation (analytic rasterization)

`_render_segmentation()` returns `(H, W)` uint8:

1. `pygame.Surface` filled with `LABEL_BG`.
2. For each enabled body (agent + objects), iterate `body.shapes`:
   - `pymunk.Poly` → world-space vertices via `body.local_to_world` →
     `pygame.draw.polygon(surface, label, points)`.
   - `pymunk.Circle` → `pygame.draw.circle(surface, label, center, radius)`.
3. `cv2.resize(..., interpolation=cv2.INTER_NEAREST)` to `render_size`.

Same path renders `goal_segmentation`.

### 3.7 Goal handling

Per-object goal pose `(x, y, θ)`; `θ` ignored for `has_orientation=False`
(circle).

`_set_goal_state` extends the existing single-object idempotent
render-then-restore pattern to N bodies. Same discipline: snapshot, set
angle BEFORE position, `reindex_shapes_for_body`, render, restore in
reverse order.

### 3.8 Success / reward / termination

```python
pos_ok[oid]   = ||pos[oid] - goal_pos[oid]|| < pos_tol
angle_ok[oid] = angle_diff(angle[oid], goal_angle[oid]) < angle_tol  # skipped if not has_orientation
obj_success[oid] = pos_ok[oid] and angle_ok[oid]

joint_success    = all(obj_success[oid] for oid in enabled)   # primary, terminates
mean_obj_success = mean(obj_success[oid] for oid in enabled)  # secondary
mean_pos_error   = mean(||pos - goal_pos||)
mean_angle_error = mean(angle_diff)

reward     = -sum(state_dist[oid] for oid in enabled)
terminated = joint_success
```

Tolerances do **not** scale with N — keeps numbers comparable to
existing single-object PushT.

### 3.9 Episode budget

`max_episode_steps = 60 + 60 * k`. Single PushT uses 100; k=6 → 420.

---

## 4. Expert policy — `MultiObjectWeakPolicy`

Direct generalization of `WeakPolicy`:

- Every `switch_every` steps, pick a *focus*. With prob `1 - p_wedge`,
  focus = a uniformly chosen enabled object. With prob `p_wedge`,
  focus = midpoint between two enabled objects (drives pairwise contact
  sampling — important for RQ2).
- Sample uniform action, scale, clip to a square neighborhood around
  the focus. Same `dist_constraint` mechanic as the existing weak
  policy.

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

## 6. Branch & file layout

```
git checkout -b pusht-multi

stable_worldmodel/envs/pusht_multi/
    __init__.py              # PushTMulti, MultiObjectWeakPolicy, OBJECT_LIBRARY
    env.py                   # PushTMulti gym.Env
    objects.py               # canonical identity registry
    expert_policy.py         # MultiObjectWeakPolicy
stable_worldmodel/envs/__init__.py    # register swm/PushTMulti-v1
scripts/data/collect_pusht_multi.py
scripts/data/config/pusht_multi.yaml
tests/test_pusht_multi.py
docs/compositional_pusht_plan.md  # this file
```

`swm/PushT-v1` and `WeakPolicy` are untouched, so cjepa keeps building.

---

## 7. Implementation order

1. `objects.py` library + per-identity mass/friction plumbed into
   existing `add_*` constructors (extracted into a shared helper).
2. `env.py` skeleton with single-object parity (set
   `enabled_objects=('A',)` and verify it matches `swm/PushT-v1` for
   shared variations).
3. Multi-object reset with non-overlap rejection sampling.
4. `_render_segmentation()` and `_set_goal_state` extended to N bodies.
5. Per-object success / reward; `info` schema with NaN sentinels.
6. `MultiObjectWeakPolicy` (with wedge mode).
7. `collect_pusht_multi.py` + dataset configs (RQ1 + RQ2 + SC).
8. Smoke test: 5 episodes per split, assert label stability (B has
   label 3 in `{A,B}`, `{B,C}`, and `{A,B,C}`), disabled objects have
   no seg pixels, info-key schema identical across splits.

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
  regime. Mitigation: wedge mode + check contact-frequency stats per
  shard before training.
- **Goal-frame variation overrides**: the existing PushT
  `_set_goal_state` is delicate (angle-before-position, reindex). The
  N-body extension must preserve this.
- **Render cost**: extra segmentation pass per step. Acceptable for
  N≤6; gate behind a flag if it becomes a bottleneck.
- **Damping change**: `space.damping = 0.85` is a deliberate divergence
  from `swm/PushT-v1`. Don't backport.
