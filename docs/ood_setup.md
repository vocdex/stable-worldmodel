# OOD Planning Evaluation Setup

How we measure the robustness of pre-trained world models (DINO-WM, PreJEPA/LeWM
checkpoints) to visual out-of-distribution shifts at *test time*: the WM is frozen,
the planning task is identical to the in-distribution baseline, and only the
appearance of the scene is perturbed via the env's variation space.

Entry point: `scripts/plan/eval_wm.py` → `swm.World.evaluate_from_dataset`.
Launch scripts (one SLURM array task per OOD cell):
`scripts/cluster/plan_ood/cube.sh`, `scripts/cluster/plan_ood/pusht.sh`.

## End-to-end flow

### 1. Task sampling (identical across all cells)

`eval_wm.py` samples `eval.num_eval` (50) `(episode, start_step)` pairs from the
expert H5 dataset (`pusht_expert_train` / `cube_single_expert`) with a fixed
`seed` (42). The goal for each pair is **the same episode `goal_offset_steps − 1`
(24) env-steps later** — `load_chunk(ep, start, start + goal_offset_steps)`
slices with an exclusive end, so the goal row is `start + goal_offset_steps − 1`
(semantic pinned by `tests/eval/test_ood_trust_chain.py::goal_row`). Because
episode selection only depends on `(seed,
num_eval, goal_offset_steps, dataset)`, every cell of the OOD matrix evaluates
the exact same 50 windows.

### 2. Env reset under variation

Each OOD cell passes a hydra override, e.g.

```
+eval.variation_overrides={variation:[cube.color],variation_values:{cube.color:[[0.0,0.0,1.0]]}}
```

`evaluate_from_dataset` turns this into per-env reset options
`{variation: [...], variation_values: {...}, render_goal: True}`. Note this
**replaces** any variation values recorded in the dataset rather than merging
(benign today: the recorded default variations are positional and get restored
numerically in step 3). On reset, `reset_variation_space` pins the listed
variation entries to the given values and the env bakes them into itself:

- **PushT** rebuilds the pymunk scene in `_setup()` (colors, scales, shapes,
  distractor).
- **Cube** edits the MJCF and recompiles in `modify_mjcf_model()` (cube
  color/size, arm color, floor texture, camera pose, light intensity).

The variation persists for the whole episode: the eval loop never resets again
(vector autoreset is disabled).

### 3. State pinning via callables

The env's random reset state is overwritten from the H5 (configured under
`eval.callables` in `scripts/plan/config/{pusht,cube}.yaml`):

| | start state | success target | goal frame |
|---|---|---|---|
| PushT | `_set_state(state@t0)` | `_set_goal_state(state@t0+25)` sets `goal_state` | same call re-renders `self._goal` under the active variation |
| Cube | `set_state(qpos,qvel@t0)` | `set_target_pos(privileged_block_0_pos/quat@t0+25)` sets the mocap target | `render_goal_scene(goal_qpos, goal_qvel)` renders `_cur_goal_rendered` under the active variation |

The `goal_*` columns are produced by `evaluate_from_dataset` itself: it loads
the chunk `[t0, t0+25]` and exposes the **end-step** row of every column with a
`goal_` prefix (`pixels` → `goal`).

> **Do not remove the cube `render_goal_scene` callable.** Without it,
> `env._goal` holds whatever the env rendered at reset — a *randomly sampled
> predefined task goal* (unseeded `np.random` task selection in
> `ManipSpaceEnv.reset`) — and the planner targets a goal image showing the
> cube at the wrong position in every OOD cell. Cube OOD matrices produced
> before this callable existed are invalid.

### 4. Planner input frames

The H5 frames were rendered *without* the variation at collection time, so
under an override `evaluate_from_dataset` replaces them:

- **Goal frame**: re-pulled from `env.unwrapped._goal`, which step 3 just
  re-rendered at the H5 goal state under the variation.
- **Start frame** (t=0 observation): re-rendered live via
  `env.unwrapped.render()` — the env is at the H5 start pose with the
  variation-modified model. A failure here raises instead of silently feeding
  an unperturbed start frame.

Both replacements only fire when `variation_overrides` is set, so baseline /
in-distribution runs are byte-identical to the legacy behavior.

**All later observations are perturbed for free**: every `world.step()` renders
the live (variation-carrying) scene through `AddPixelsWrapper` into
`infos['pixels']`. Only the t=0 frame and the goal frame need the special
handling above, because only they originate from the H5.

### 5. CEM + MPC loop

Per env step, `world.step()` calls `WorldModelPolicy.get_action(infos)`:

- Inputs are normalized first (`_prepare_info`): images → float / Normalize /
  Resize 224 (half-stats for external DINO-WM, ImageNet for SWM-native);
  proprio & actions → `StandardScaler` fit on dataset stats (overridden by the
  ckpt's own constants for external DINO-WM).
- If the action buffer is non-empty: pop the next action, inverse-transform to
  env units, return. **Open loop — no model call; the fresh frame is ignored.**
- If empty, run CEM (`solver/cem.py`, N=300 samples, 30 iterations, topk=30,
  fresh variance each solve, seeded identically in every cell):
  1. Sample action sequences of shape `(horizon=5, action_dim × action_block=5)`
     in normalized action space from `N(mean, var)`; the first sample is forced
     to the current mean.
  2. `model.get_cost`: encode the goal frame once → goal latent; encode the
     current frame (+proprio); unroll the latent dynamics under each candidate
     (one model step consumes one 5-action block = 5 env steps); cost = latent
     distance of the predicted final state to the goal latent.
  3. Refit mean/var on the top-30 elites; after 30 iterations the mean is the plan.
- The first `receding_horizon` (5) blocks = **25 env actions** go into the
  buffer and execute open-loop. `warm_start` is a no-op here since
  `receding_horizon == horizon`.

With `eval_budget=50` there are exactly **two planning rounds**: at t=0 (from
the re-rendered perturbed start + goal) and at t=25 (from a live perturbed
render). The goal frame is re-injected into `infos` every step, so the t=25
replan targets the same perturbed goal.

### 6. Success metric (appearance-independent)

The env checks success from ground-truth state at *every* step, regardless of
what the planner sees: PushT `eval_state` (block pos diff < 20 px and angle
diff < π/9), cube `_compute_successes` (block within 4 cm of the mocap target).
An episode counts as success if it terminates successfully at any point within
the budget. Results land in `ood_matrix.csv` per cell (plus per-cell rollout
videos: live rollout / dataset replay / goal panel).

## History conditioning

**Current configs condition the world model on a single current frame — there
is no real observation history at plan time.**

- The eval `World` is built with `history_size=1, frame_skip=1`
  (`scripts/plan/config/{pusht,cube}.yaml`), so `infos['pixels']` has time
  dimension 1, and that is exactly what the WM receives as context.
- **PreJEPA** checkpoints are *trained* with `history_size=3` on
  frameskip-aligned features (3 context frames spaced 5 env-steps apart). At
  planning time, `rollout()` takes `n_obs = pixels.shape[time]` real frames —
  here 1 — and the trained 3-frame context only materializes *inside the
  latent rollout*: each `predict` call conditions on the last
  `history_size` latent states (`z_flat[:, -history_size:]`), which after the
  first two predictions are the model's own predicted latents, never extra
  rendered frames. The very first prediction therefore runs with 1-frame
  context (a deliberate train/inference mismatch).
- **External DINO-WM** does the same by design: the adapter feeds exactly 1
  init frame regardless of the ckpt's `num_hist`, matching the native
  `dino_wm/planning/cem.py` protocol (frame-repeat padding to `num_hist` was
  tried and removed — it fed an OOD identical-frame window *and* misaligned
  the rollout horizon; see comment in `wm/dino_wm_external/adapter.py::get_cost`).
- The goal is likewise a single frame (the wrappers replicate it along the
  history axis; the cost functions collapse it back to one frame).

**Are the conditioning frames perturbed?** Yes, all of them: the t=0 frame is
the variation-re-rendered start frame, the t=25 replan frame is a live render
of the variation-carrying env, and the goal frame is the variation-re-rendered
goal. The intermediate "frames" of the planning rollout are latent predictions,
not renders, so perturbation doesn't apply to them.

**If you ever set `history_size > 1`, this eval is not history-correct:**

- `evaluate_from_dataset` broadcasts the *single* start frame across the
  history axis instead of fetching the true previous H5 frames (see the TODO
  above the broadcast), so the t=0 "history" would be the same frame repeated.
- The frame-stacking buffer (`StackedWrapper`) is filled at reset time with a
  render of the env's *pre-callable* random pose; the callables change the
  state but don't refresh the buffer. The stale frames flush out after
  `history_size × frame_skip` steps, which is fine for the t=25 replan but
  wrong for any replan earlier than that.
- Matching the PreJEPA training distribution would additionally require
  `frame_skip=5` so stacked frames are spaced like the training windows.

## What changes vs. what doesn't (per OOD cell)

| changed | unchanged |
|---|---|
| env appearance (model/scene rebuild at reset) | the 50 evaluated (episode, start) windows |
| planner's start frame, goal frame, and all live observations | start/goal *numerical* states, success thresholds |
| — | CEM hyperparameters and seed, MPC schedule, normalization stats |

Caveats:

- `block.scale` (PushT) and `cube.size` (cube) change **physics**, not just
  appearance — those cells also measure dynamics-OOD, and frames re-rendered
  at recorded states have slightly inconsistent contacts.
- Camera, light, and all color cells are purely visual.
- The cube random-policy floor is ~46–52 % SR (many 25-step expert windows
  barely move the block, so the start state already satisfies the goal). SR
  differences between cells are compressed against that floor.
- `agent.color` (cube) and `block.scale` (PushT) were silent no-ops before
  commit `a1437f7`; matrices run before it are stale for those cells.

## Current cube results (2026-06-12, fixed eval, seeds 0/1/2)

Figures: `cjepa/experiments/presentations/ood_cube.{png,pdf}` (absolute SR,
3-way) and `ood_cube_deltas.{png,pdf}` (drop vs each model's own baseline).
Data: `cjepa/.../cube_ood_cem300_train/runs/dinowm_cube_fixed_seeds012_2026-06-12.csv`;
plot scripts `plot_ood_3way_models.py` / `plot_ood_deltas.py` in the same dir.

| factor | SC-WM | DINO-WM | LeWM |  | ΔSC | ΔDINO | ΔLeWM |
|---|---|---|---|---|---|---|---|
| in-distribution | 72.7 | 74.0 | 66.7 | | — | — | — |
| cube.color | 70.9 | 64.7 | 64.7 | | −1.8 | −9.3 | −2.0 |
| cube.scale | 68.0 | 63.6 | 63.7 | | −4.7 | −8.7 | −3.0 |
| agent.color | 69.3 | 68.7 | 61.7 | | −3.3 | −5.3 | −5.0 |
| background.color | 61.7 | 63.3 | 42.7 | | −11.0 | −10.7 | −24.0 |
| camera.angle_delta | 60.9 | 62.0 | 48.0 | | −11.8 | −12.0 | −18.7 |
| light.intensity | 70.7 | 66.7 | 59.3 | | −2.0 | −7.3 | −7.3 |

Reading: absolute SRs tie between SC and DINO-WM (DINO's higher baseline
masks its larger relative drops, mean −8.9pp vs SC's −5.8pp). The deltas are
where the architectures differ: SC is near-flat on object-level appearance
(scale-separation) but takes the same scene-level hit as DINO-WM; LeWM
collapses on scene-level. Earlier DINO-WM cube numbers (uniform 28–40pp
drops) were an artifact of the goal-frame bug fixed in `1f552ef`; seed 42
alone also inflated DINO-WM by 3–9pp vs the seeds-0/1/2 protocol. Known
gaps: seed-0 `cube_size_small` (SLURM-killed; `SEED=0 sbatch --array=4`)
and the `floor_natural_static` cell (not yet run).

## Trust verification

Every link of the chain above is pinned by atomic tests in
`tests/eval/test_ood_trust_chain.py` (goal/start frame correctness vs
independent re-renders, success-target invariance, baseline byte-identity, a
no-op sweep over every variation leaf, live-frame tracking — plus regression
detectors that re-create the missing-goal-callable bug class). At run time,
`SWM_EVAL_TRUST_CHECKS=1` (exported by the cluster launchers) re-verifies the
planner inputs inside `evaluate_from_dataset` and fails loudly instead of
producing plausible-but-wrong success rates. The newer dynamic-distractor
cells are documented in `dynamic_ood_cells.md`.

## Running

```bash
sbatch scripts/cluster/plan_ood/cube.sh        # 15 cells, idempotent per-cell
sbatch scripts/cluster/plan_ood/pusht.sh       # 10 cells
EPOCH=20 SEED=43 sbatch --array=5,8 scripts/cluster/plan_ood/cube.sh
```

Each cell writes `cells/<label>/{eval.log,gpu.log,done.flag}` and appends to a
flock-protected `ood_matrix.csv` under the checkpoint's results dir.
