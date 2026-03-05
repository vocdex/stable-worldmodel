"""Tests for colored noise generation."""

import torch

from stable_worldmodel.solver.colored_noise import powerlaw_psd_gaussian


def test_shape():
    """Output shape matches requested shape."""
    shape = (4, 8, 16)
    result = powerlaw_psd_gaussian(2.0, shape)
    assert result.shape == shape


def test_white_noise_unit_variance():
    """Beta=0 (white noise) produces approximately unit variance."""
    torch.manual_seed(0)
    result = powerlaw_psd_gaussian(0.0, (1000, 64))
    assert abs(result.var().item() - 1.0) < 0.15


def test_temporal_correlation():
    """Higher beta produces more temporally correlated (smoother) noise."""
    torch.manual_seed(42)
    shape = (100, 256)

    white = powerlaw_psd_gaussian(0.0, shape)
    brown = powerlaw_psd_gaussian(2.0, shape)

    white_diff = (white[:, 1:] - white[:, :-1]).abs().mean()
    brown_diff = (brown[:, 1:] - brown[:, :-1]).abs().mean()

    assert brown_diff < white_diff


def test_horizon_one():
    """Edge case: temporal dimension of 1 falls back to standard randn."""
    result = powerlaw_psd_gaussian(2.0, (5, 3, 1))
    assert result.shape == (5, 3, 1)


def test_generator_reproducibility():
    """Same generator seed produces same output."""
    gen1 = torch.Generator().manual_seed(123)
    gen2 = torch.Generator().manual_seed(123)
    a = powerlaw_psd_gaussian(1.5, (4, 16), generator=gen1)
    b = powerlaw_psd_gaussian(1.5, (4, 16), generator=gen2)
    assert torch.allclose(a, b)
