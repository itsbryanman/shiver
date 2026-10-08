"""Small Fourier-domain complex pyramid used by the prototype.

This is an intentionally compact, inspectable filter bank.  Each filter is a
log-Gaussian radial band multiplied by a raised-cosine orientation window and
an analytic half-plane mask.  Its complex response has a local phase whose
change tracks tiny translations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class PyramidBand:
    scale: int
    orientation: int
    center_frequency: float
    angle_radians: float
    transfer: ComplexArray


def _angle_distance(angle: FloatArray, center: float) -> FloatArray:
    """Smallest signed orientation difference, modulo pi."""

    return (angle - center + np.pi / 2.0) % np.pi - np.pi / 2.0


def build_filter_bank(
    shape: tuple[int, int],
    *,
    scales: int = 3,
    orientations: int = 4,
    base_frequency: float = 0.24,
    radial_bandwidth_octaves: float = 0.8,
) -> list[PyramidBand]:
    """Build complex oriented band-pass filters for a frame shape.

    Frequencies are in cycles per pixel.  The finest band's center frequency
    is ``base_frequency`` and each following scale is one octave lower.
    """

    if scales < 1 or orientations < 1:
        raise ValueError("scales and orientations must be positive")
    if not 0.0 < base_frequency < 0.5:
        raise ValueError("base_frequency must be between 0 and Nyquist (0.5)")

    height, width = shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.hypot(fx, fy)
    angle = np.arctan2(fy, fx)
    safe_radius = np.maximum(radius, np.finfo(float).tiny)
    bands: list[PyramidBand] = []

    for scale in range(scales):
        center_frequency = base_frequency / (2**scale)
        log_distance = np.log2(safe_radius / center_frequency)
        radial = np.exp(-0.5 * (log_distance / radial_bandwidth_octaves) ** 2)
        radial[radius == 0] = 0.0

        for orientation in range(orientations):
            theta = orientation * np.pi / orientations
            delta = _angle_distance(angle, theta)
            half_width = np.pi / orientations
            angular = np.zeros(shape, dtype=np.float64)
            inside = np.abs(delta) <= half_width
            angular[inside] = np.cos(delta[inside] * np.pi / (2.0 * half_width)) ** 2

            # Retain one frequency half-plane.  The inverse transform is then
            # complex, giving a quadrature pair and a meaningful local phase.
            projection = fx * np.cos(theta) + fy * np.sin(theta)
            analytic = np.where(projection > 0.0, 2.0, 0.0)
            transfer = (radial * angular * analytic).astype(np.complex128)
            bands.append(PyramidBand(scale, orientation, center_frequency, theta, transfer))

    return bands


def decompose(frame: FloatArray, filters: list[PyramidBand]) -> list[ComplexArray]:
    """Return the complex response of every pyramid band."""

    spectrum = np.fft.fft2(np.asarray(frame, dtype=np.float64))
    return [np.fft.ifft2(spectrum * band.transfer) for band in filters]
