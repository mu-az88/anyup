
import torch
import torch.nn.functional as F


def random_brightness_contrast(
    frames: torch.Tensor,          # (B, T, H, W, 3) — RGB guidance video, float in [0, 1]
    brightness_range: tuple = (0.85, 1.15),  # ↑ multiplicative brightness factor per frame
    contrast_range:   tuple = (0.85, 1.15),  # ↑ contrast scale per frame around mean
) -> torch.Tensor:
    """
    Apply independent per-frame brightness and contrast jitter.
    All ops stay on the same device as `frames`.

    Returns:
        Augmented frames, same shape and dtype as input, clamped to [0, 1].
    """
    B, T, H, W, C = frames.shape   # ↑ T frames processed independently

    lo_b, hi_b = brightness_range
    lo_c, hi_c = contrast_range

    # Sample one scalar per (B, T) pair — shape (B, T, 1, 1, 1) for broadcasting
    # ↓ each frame gets its own factor — independent across time
    brightness = torch.empty(B, T, 1, 1, 1, device=frames.device).uniform_(lo_b, hi_b)
    contrast   = torch.empty(B, T, 1, 1, 1, device=frames.device).uniform_(lo_c, hi_c)

    # Per-frame mean for contrast anchoring — shape (B, T, 1, 1, 1)
    # ↓ mean computed over H, W, C — cheap relative to attention
    frame_mean = frames.mean(dim=(2, 3, 4), keepdim=True)

    augmented = (frames - frame_mean) * contrast + frame_mean  # contrast
    augmented = augmented * brightness                          # brightness
    return augmented.clamp(0.0, 1.0)


def random_gaussian_noise(
    frames: torch.Tensor,   # (B, T, H, W, 3) — RGB guidance video, float in [0, 1]
    std: float = 0.02,      # ↑ noise std; 0.02 ≈ subtle, 0.05 starts to be visible
) -> torch.Tensor:
    """
    Add independent per-frame Gaussian noise.

    Returns:
        Noisy frames, same shape, clamped to [0, 1].
    """
    # ↓ noise tensor same shape as frames — (B, T, H, W, 3)
    # ↓ H*W*3 floats per frame — memory proportional to resolution; reduce H,W if OOM
    noise = torch.randn_like(frames) * std
    return (frames + noise).clamp(0.0, 1.0)


def augment_guidance_video(
    frames: torch.Tensor,                   # (B, T, H, W, 3) float in [0, 1]
    brightness_range: tuple = (0.85, 1.15), # ↑ see random_brightness_contrast
    contrast_range:   tuple = (0.85, 1.15), # ↑ see random_brightness_contrast
    noise_std: float = 0.02,                # ↑ see random_gaussian_noise
) -> torch.Tensor:
    """
    Full augmentation pipeline for L_self-consistency.
    Apply brightness/contrast jitter then Gaussian noise.
    Order matters: noise after rescaling avoids noise being scaled up by brightness.

    Returns:
        Augmented guidance video, same shape as input.
    """
    frames = random_brightness_contrast(frames, brightness_range, contrast_range)
    frames = random_gaussian_noise(frames, noise_std)
    return frames