"""Profile SC-WM planning time breakdown: encoder vs predictor vs CEM overhead.

Run in the cjepa conda env:
    cd /home/nazirjon/Desktop/stable-worldmodel
    PYTHONPATH=/home/nazirjon/Desktop/cjepa:/home/nazirjon/Desktop/vdinosaur \
    MUJOCO_GL=egl \
    /home/nazirjon/miniconda3/envs/cjepa/bin/python scripts/plan/profile_scwm.py
"""

import os, sys, time, statistics
sys.path.insert(0, '/home/nazirjon/Desktop/cjepa')
sys.path.insert(0, '/home/nazirjon/Desktop/vdinosaur')
os.environ.setdefault('MUJOCO_GL', 'egl')

import torch
from omegaconf import OmegaConf

_orig = OmegaConf.register_new_resolver
def _safe(name, resolver, **kw):
    kw['replace'] = True
    return _orig(name, resolver, **kw)
OmegaConf.register_new_resolver = staticmethod(_safe)

CKPT = ('/home/nazirjon/Desktop/cjepa/.cache/'
        'pusht_wm_sf_s0_sc_dinov3_256_nms0_noproprio/'
        'pusht_wm_sf_s0_sc_dinov3_256_nms0_noproprio_object.ckpt')

N     = 300  # CEM candidates
H     = 5    # horizon
ITERS = 30   # CEM iterations
WARMUP = 5
REPS   = 20


def timed(fn, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return {'mean': statistics.mean(ts), 'std': statistics.pstdev(ts)}


def fmt(d, unit='ms'):
    s = 1e3 if unit == 'ms' else 1
    return f"{d['mean']*s:.1f} ± {d['std']*s:.1f} {unit}"


def main():
    print("Loading SC-WM from .ckpt ...")
    wrapper = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = wrapper.model.cuda().eval()
    model.requires_grad_(False)

    K = model.slot_attention.num_slots if hasattr(model.slot_attention, 'num_slots') else 4
    print(f"  encoder:    {sum(p.numel() for p in model.encoder.parameters())/1e6:.1f}M params")
    print(f"  slot_attn:  {sum(p.numel() for p in model.slot_attention.parameters())/1e6:.2f}M params")
    print(f"  predictor:  {sum(p.numel() for p in model.predictor.parameters())/1e6:.2f}M params")

    # encoder wraps via MapOverTime → expects (B, T, C, H, W)
    IMG_SIZE = 256  # training input_size for DINOv3-256
    img  = torch.randn(1, 1, 3, IMG_SIZE, IMG_SIZE, device='cuda')  # (B, T, C, H, W)

    # --- 1. Encoder: image → slots (full encode() call as used at runtime) ---
    print(f"\n[1] Encoder (image {IMG_SIZE}px → patch features → slots), B=1 ...")
    info_enc = {'pixels': img}
    def run_encoder():
        with torch.inference_mode():
            model.encode(info_enc.copy(), target='embed', pixels_key='pixels')
    t_enc = timed(run_encoder)
    print(f"    {fmt(t_enc)}")

    # get real slot shape from encode output
    with torch.inference_mode():
        enc_out = model.encode(info_enc.copy(), target='embed', pixels_key='pixels')
    slots = enc_out['embed'][:, -1]  # (1, K, D) — last timestep
    K, D = slots.shape[1], slots.shape[2]
    print(f"    slots: (1, {K}, {D})")

    # --- 2. Full rollout for N=300 candidates (= 1 CEM iteration) via rollout() ---
    # action_dim for pusht: env_action_dim=2, action_block=5 → 10
    action_dim = 10
    # action_sequence: (B=1, N=300, H+1, action_dim)  — +1 because rollout needs history action too
    action_seq = torch.randn(1, N, H + 1, action_dim, device='cuda')
    # rollout() expects pre-expanded (B, N, ...) tensors — same as CEM pre-expansion
    img_exp  = img.unsqueeze(1).expand(1, N, -1, -1, -1, -1)  # (1, N, 1, 3, 256, 256)
    info_rollout = {
        'pixels':   img_exp,
        'goal':     img_exp,
        'action':   torch.zeros(1, N, 1, action_dim, device='cuda'),
        'id':       torch.zeros(1, N, dtype=torch.long, device='cuda'),
        'step_idx': torch.zeros(1, N, dtype=torch.long, device='cuda'),
    }

    print(f"\n[2] Full H={H} rollout for N={N} candidates (= 1 CEM iteration) ...")
    def run_rollout():
        with torch.inference_mode():
            model.rollout(info_rollout.copy(), action_seq)
    t_rollout = timed(run_rollout)
    print(f"    {fmt(t_rollout)}")
    # single step estimate
    t_pred = {'mean': t_rollout['mean'] / H, 'std': t_rollout['std'] / H}
    print(f"    → per step: {fmt(t_pred)}")

    # --- 4. Breakdown ---
    t_predictor_total = ITERS * t_rollout['mean']
    t_total = t_enc['mean'] + t_predictor_total
    print(f"\n[4] Reconstructed CEM solve ({ITERS} iters × H={H}, N={N}):")
    print(f"    encoder (once):  {t_enc['mean']*1e3:6.1f} ms  ({t_enc['mean']/t_total*100:.1f}%)")
    print(f"    predictor (all): {t_predictor_total*1e3:6.0f} ms  ({t_predictor_total/t_total*100:.1f}%)")
    print(f"    total:           {t_total:.2f} s")

    # --- 5. Config sweep ---
    print(f"\n[5] Estimated solve time vs CEM config "
          f"(encoder={t_enc['mean']*1e3:.0f}ms fixed, pred_per_step={t_pred['mean']*1e3:.1f}ms):")
    print(f"    {'N':>6}  {'iters':>6}  {'est. solve':>12}  {'vs baseline':>12}")
    baseline = t_total
    for n_cands in [300, 200, 100]:
        scale = n_cands / N
        for n_iters in [30, 20, 10]:
            t = t_enc['mean'] + n_iters * H * t_pred['mean'] * scale
            print(f"    {n_cands:>6}  {n_iters:>6}  {t:>10.2f} s  {t/baseline*100:>10.0f}%")


if __name__ == '__main__':
    main()
