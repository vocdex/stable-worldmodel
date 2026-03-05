"""Power-law (1/f^β) colored noise generation in PyTorch."""

import torch


def powerlaw_psd_gaussian(
    beta: float,
    shape: tuple[int, ...],
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate Gaussian noise with a power-law spectral density.

    The last dimension of `shape` is treated as the temporal axis.

    Args:
        beta: Spectral exponent. 0 = white, 1 = pink, 2 = brown/red.
        shape: Shape of the output tensor; last dim is the temporal axis.
        generator: Optional torch Generator for reproducibility.
        device: Device for the output tensor.

    Returns:
        Tensor of shape `shape` with unit variance and 1/f^β spectral profile.
    """
    temporal_len = shape[-1]

    if temporal_len <= 1:
        return torch.randn(shape, generator=generator, device=device)

    # White noise in frequency domain
    freqs = torch.fft.rfftfreq(temporal_len, device=device)
    # Avoid division by zero at DC component
    freqs[0] = 1.0

    # Power-law scaling: S(f) ~ 1/f^β  →  amplitude ~ 1/f^(β/2)
    scale = freqs.pow(-beta / 2)
    scale[0] = scale[1]  # set DC to same scale as lowest non-zero freq

    # Generate white noise and transform
    white = torch.randn(shape, generator=generator, device=device)
    fft = torch.fft.rfft(white, dim=-1)
    colored_fft = fft * scale
    colored = torch.fft.irfft(colored_fft, n=temporal_len, dim=-1)

    # Normalize to unit variance per-sample
    std = colored.std(dim=-1, keepdim=True)
    std = std.clamp(min=1e-8)
    return colored / std
