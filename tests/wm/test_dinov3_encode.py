"""Tests for backbone prefix-token handling in PreJEPA._encode_image.

DINOv3 backbones prepend register tokens after CLS
(last_hidden_state = [CLS, reg*4, patches]), so PreJEPA must strip
`num_prefix_tokens` leading tokens (5 for DINOv3, 1 for CLS-only backbones
like DINOv2) — otherwise register tokens silently masquerade as patches.
"""

import torch
import torch.nn as nn

from stable_worldmodel.wm.prejepa import PreJEPA


class _HFStyleOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeViT(nn.Module):
    """Mimics an HF ViT: returns [prefix tokens, patch tokens] per frame."""

    def __init__(self, num_prefix: int, num_patches: int, dim: int = 8):
        super().__init__()
        self.num_prefix = num_prefix
        self.num_patches = num_patches
        self.dim = dim
        self.saw_interpolate_kwarg = False

    def forward(self, x, **kwargs):
        self.saw_interpolate_kwarg = 'interpolate_pos_encoding' in kwargs
        B = x.shape[0]
        tokens = torch.arange(self.num_prefix + self.num_patches).float()
        out = tokens.view(1, -1, 1).expand(B, -1, self.dim)
        return _HFStyleOutput(out)


def _make_model(num_prefix_tokens, backbone, interpolate_pos_encoding=True):
    return PreJEPA(
        encoder=backbone,
        predictor=nn.Identity(),
        history_size=3,
        num_pred=1,
        interpolate_pos_encoding=interpolate_pos_encoding,
        num_prefix_tokens=num_prefix_tokens,
    )


def test_default_strips_cls_only():
    backbone = FakeViT(num_prefix=1, num_patches=256)
    model = _make_model(1, backbone)
    out = model._encode_image(torch.zeros(2, 3, 3, 224, 224))
    assert out.shape == (2, 3, 256, 8)
    # first patch token is index 1 (CLS at 0 stripped)
    assert out[0, 0, 0, 0].item() == 1.0


def test_dinov3_strips_cls_and_registers():
    backbone = FakeViT(num_prefix=5, num_patches=196)
    model = _make_model(5, backbone, interpolate_pos_encoding=False)
    out = model._encode_image(torch.zeros(2, 3, 3, 224, 224))
    assert out.shape == (2, 3, 196, 8)
    # first patch token is index 5 (CLS + 4 registers stripped)
    assert out[0, 0, 0, 0].item() == 5.0


def test_interpolate_kwarg_gated_by_flag():
    backbone = FakeViT(num_prefix=5, num_patches=196)
    model = _make_model(5, backbone, interpolate_pos_encoding=False)
    model._encode_image(torch.zeros(1, 1, 3, 224, 224))
    assert not backbone.saw_interpolate_kwarg

    backbone2 = FakeViT(num_prefix=1, num_patches=256)
    model2 = _make_model(1, backbone2, interpolate_pos_encoding=True)
    model2._encode_image(torch.zeros(1, 1, 3, 224, 224))
    assert backbone2.saw_interpolate_kwarg


def test_cached_features_passthrough_skips_encoder():
    backbone = FakeViT(num_prefix=5, num_patches=196)
    model = _make_model(5, backbone)
    feats = torch.randn(2, 4, 196, 8)
    out = model._encode_image(feats)
    assert torch.equal(out, feats.float())
