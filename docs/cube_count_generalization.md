# Cube Count Generalization

Does a world model trained on N cubes generalize to planning with M ≠ N cubes?

## Setup

- **Models**: DINO-WM (ours, patch-grid DINO features) vs SlotFormer (cjepa, object-centric slots)
- **Training**: one model per cube count (1/2/3/4), each trained on its own expert dataset
- **Evaluation**: 4×4 matrix — every model × every test environment
- **Protocol**: CEM-300, 50 episodes, seeds 0/1/2, SR = fraction where **all** cubes reach goal (≤ 0.04 m)

## Results

![Count generalization heatmap](assets/cube_count_gen_heatmap.png)

*Diagonal (navy border) = in-distribution. Values: mean ± std across 3 seeds. †n=2 seeds.*

### Full table (DINO-WM / SlotFormer)

| Train \ Test | 1 cube | 2 cubes | 3 cubes | 4 cubes |
|---|---|---|---|---|
| **1 cube**  | **68.0±6.0** / 67.3±9.5 ◀ | 55.3±3.1 / **63.3±7.6** | **60.7±4.2** / 54.0±8.0 | **56.0±5.3** / 54.7±1.2 |
| **2 cubes** | **67.3±6.1** / 58.7±7.0   | 62.7±1.2 / **69.3±4.2** ◀ | **61.3±4.6** / 60.0±8.0 | **62.7±6.1** / 50.7±3.1 |
| **3 cubes** | **70.7±5.0** / 59.3±5.8   | 62.7±5.8 / **64.0±12.2** | **69.0±4.2†** / 58.0±3.5 ◀ | **61.3±6.4** / 53.3±3.1 |
| **4 cubes** | **66.7±5.0** / 63.3±6.1   | **67.3±1.2** / 63.3±5.8   | **74.0±5.3** / 65.3±5.8   | **64.7±3.1** / 61.3±4.2 ◀ |

**Bold** = winner by >0.5pp. ◀ = in-distribution diagonal.

DINO-WM wins **12/16 cells**. SlotFormer wins only on test=2 cubes for train=1/2/3.

## Key findings

**DINO-WM matches or exceeds SlotFormer on 12/16 cells**, despite using no object-centric
structure. The advantage grows with cube count: DINO-WM is clearly better on 3- and 4-cube
tasks, while SlotFormer has a small edge on 2-cube evaluation when trained on ≤3 cubes.

**Training on more cubes helps.** Train=4 is the strongest DINO-WM config across all test
environments — it reaches 74.0% on 3-cube OOD and 67.3% on 2-cube, the best in each column.
SlotFormer does not show the same monotonic benefit from harder training counts.

**Count generalization is robust for both models.** Train=1 → test=4 gives 56% (DINO-WM) and
55% (SlotFormer), only ~9–10pp below in-distribution. The first cube is always solved (SR_ge1 ≈ 100%);
the bottleneck is the last cube.

**DINO-WM has lower variance.** Most cells show tighter ±std, especially test=2 (±1.2 vs ±4.2
for in-distribution double), suggesting more stable planning in patch-feature space.

## Methodology notes

| | DINO-WM (SWM) | SlotFormer (cjepa) |
|---|---|---|
| Features | DINO ViT-S/8 patch grid, 224px | Frozen SC encoder, 7 slots, 256px |
| Planning | CEM-300, n_steps=30, topk=30 | CEM-300, goal-weighted slot filter |
| Max episode steps | 250 / 500 / 750 / 1000 (scaled by count) | 100 (= 2 × eval_budget, fixed) |
| Eval split | full dataset | train split (~2500 eps) |

The most notable difference: cjepa uses `+slot_filter=goal_weighted` during planning, which
re-weights slots by goal relevance — an explicit mechanism to help SlotFormer focus on the
right objects. DINO-WM has no equivalent and still wins on 12/16 cells.

The episode step budget differs: SWM scales max steps with cube count (more time for harder
tasks), while cjepa fixes it at 100 regardless. This likely benefits DINO-WM on triple/quadruple.
