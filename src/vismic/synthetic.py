"""Deterministic sub-pixel vibration benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticClip:
    frames: FloatArray
    displacement: FloatArray
    fps: float


def make_texture(shape: tuple[int, int], seed: int = 7) -> FloatArray:
    """Generate a repeatable, multi-scale texture with strong local phase."""

    rng = np.random.default_rng(seed)
    fine = gaussian_filter(rng.standard_normal(shape), 0.7)
    coarse = gaussian_filter(rng.standard_normal(shape), 3.0)
    yy, xx = np.indices(shape)
    directional = 0.25 * np.sin(2 * np.pi * (0.13 * xx + 0.07 * yy))
    texture = fine + 0.7 * coarse + directional
    texture -= texture.min()
    texture /= texture.max()
    return texture


def fourier_shift(image: FloatArray, dx: float, dy: float = 0.0) -> FloatArray:
    """Translate an image by a fractional pixel using the Fourier shift theorem."""

    height, width = image.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    ramp = np.exp(-2j * np.pi * (fx * dx + fy * dy))
    return np.fft.ifft2(np.fft.fft2(image) * ramp).real


def make_synthetic_clip(
    *,
    shape: tuple[int, int] = (96, 96),
    fps: float = 1000.0,
    duration: float = 0.5,
    frequency: float = 37.0,
    amplitude_pixels: float = 0.01,
    noise_std: float = 0.001,
    seed: int = 7,
) -> SyntheticClip:
    """Shift a texture with a known horizontal sinusoid."""

    if fps <= 2 * frequency:
        raise ValueError("fps must exceed twice the vibration frequency")
    if duration <= 0 or amplitude_pixels <= 0:
        raise ValueError("duration and amplitude_pixels must be positive")
    count = int(round(fps * duration))
    time = np.arange(count, dtype=np.float64) / fps
    displacement = amplitude_pixels * np.sin(2 * np.pi * frequency * time)
    texture = make_texture(shape, seed)
    rng = np.random.default_rng(seed + 1)
    frames = np.stack(
        [
            fourier_shift(texture, float(dx)) + rng.normal(0.0, noise_std, shape)
            for dx in displacement
        ]
    )
    return SyntheticClip(frames, displacement, fps)
