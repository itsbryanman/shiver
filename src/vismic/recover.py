"""Global phase-motion extraction and band alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .pyramid import build_filter_bank, decompose


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RecoveryConfig:
    scales: int = 3
    orientations: int = 4
    base_frequency: float = 0.24
    radial_bandwidth_octaves: float = 0.8
    max_alignment_lag: int = 3
    amplitude_floor_quantile: float = 0.25


@dataclass(frozen=True)
class RecoveryResult:
    signal: FloatArray
    band_signals: FloatArray
    band_weights: FloatArray
    lags: NDArray[np.int64]
    correlations: FloatArray


def _frames(video: ArrayLike | Iterable[ArrayLike]) -> tuple[Iterator[FloatArray], FloatArray]:
    iterator = iter(video)
    try:
        first = np.asarray(next(iterator), dtype=np.float64)
    except StopIteration as exc:
        raise ValueError("video must contain at least one frame") from exc
    if first.ndim == 3:
        first = first[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    if first.ndim != 2:
        raise ValueError("frames must be 2D grayscale or 3D RGB arrays")

    def normalized() -> Iterator[FloatArray]:
        yield first
        for raw in iterator:
            frame = np.asarray(raw, dtype=np.float64)
            if frame.ndim == 3:
                frame = frame[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
            if frame.shape != first.shape:
                raise ValueError(f"all frames must have shape {first.shape}, got {frame.shape}")
            yield frame

    return normalized(), first


def _best_lag(reference: FloatArray, candidate: FloatArray, max_lag: int) -> tuple[int, float]:
    best_lag = 0
    best_corr = 0.0
    ref = reference - np.mean(reference)
    cand = candidate - np.mean(candidate)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = ref[-lag:], cand[:lag]
        elif lag > 0:
            left, right = ref[:-lag], cand[lag:]
        else:
            left, right = ref, cand
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        corr = float(np.dot(left, right) / denom) if denom > 0 else 0.0
        if abs(corr) > abs(best_corr):
            best_lag, best_corr = lag, corr
    return best_lag, best_corr


def _shift_with_edges(signal: FloatArray, lag: int) -> FloatArray:
    if lag == 0:
        return signal.copy()
    shifted = np.empty_like(signal)
    if lag > 0:
        shifted[:-lag] = signal[lag:]
        shifted[-lag:] = signal[-1]
    else:
        shifted[-lag:] = signal[:lag]
        shifted[:-lag] = signal[0]
    return shifted


def align_and_combine(
    band_signals: FloatArray,
    band_weights: FloatArray,
    *,
    max_lag: int,
) -> tuple[FloatArray, NDArray[np.int64], FloatArray]:
    """Cross-correlate band traces, correct polarity/lag, and average."""

    if band_signals.ndim != 2:
        raise ValueError("band_signals must have shape (bands, samples)")
    temporal_power = np.std(band_signals, axis=1)
    anchor = int(np.argmax(temporal_power * band_weights))
    aligned = np.empty_like(band_signals)
    lags = np.zeros(len(band_signals), dtype=np.int64)
    correlations = np.ones(len(band_signals), dtype=np.float64)
    aligned[anchor] = band_signals[anchor]

    for index, trace in enumerate(band_signals):
        if index == anchor:
            continue
        lag, correlation = _best_lag(band_signals[anchor], trace, max_lag)
        lags[index] = lag
        correlations[index] = correlation
        aligned[index] = _shift_with_edges(trace, lag) * (1.0 if correlation >= 0 else -1.0)

    confidence = band_weights * np.maximum(np.abs(correlations), 0.05)
    if not np.any(confidence > 0):
        raise ValueError("no textured pyramid bands were found")
    combined = np.average(aligned, axis=0, weights=confidence)
    combined -= np.mean(combined)
    peak = np.max(np.abs(combined))
    if peak > 0:
        combined /= peak
    return combined, lags, correlations


def recover_signal(
    video: ArrayLike | Iterable[ArrayLike],
    config: RecoveryConfig | None = None,
) -> RecoveryResult:
    """Recover one normalized vibration sample per video frame.

    This is the high-speed/global-shutter path: the output sample rate equals
    the input frame rate.  It does not manufacture audio-band samples from a
    low-frame-rate clip.
    """

    config = config or RecoveryConfig()
    frames, first = _frames(video)
    filters = build_filter_bank(
        first.shape,
        scales=config.scales,
        orientations=config.orientations,
        base_frequency=config.base_frequency,
        radial_bandwidth_octaves=config.radial_bandwidth_octaves,
    )
    reference = decompose(first, filters)
    amplitude2 = [np.abs(response) ** 2 for response in reference]
    masks = []
    for weights in amplitude2:
        threshold = np.quantile(weights, config.amplitude_floor_quantile)
        masks.append(weights > threshold)

    samples: list[list[float]] = [[] for _ in filters]
    for frame in frames:
        responses = decompose(frame, filters)
        for index, response in enumerate(responses):
            # angle(z * conj(z_ref)) is the wrapped phase change and avoids
            # subtracting two independently wrapped angles.
            phase_delta = np.angle(response * np.conj(reference[index]))
            weights = amplitude2[index] * masks[index]
            weight_sum = float(np.sum(weights))
            sample = float(np.sum(weights * phase_delta) / weight_sum) if weight_sum else 0.0
            samples[index].append(sample)

    band_signals = np.asarray(samples, dtype=np.float64)
    band_weights = np.asarray(
        [
            float(np.mean(weights[mask])) if np.any(mask) else 0.0
            for weights, mask in zip(amplitude2, masks)
        ]
    )
    signal, lags, correlations = align_and_combine(
        band_signals,
        band_weights,
        max_lag=config.max_alignment_lag,
    )
    return RecoveryResult(signal, band_signals, band_weights, lags, correlations)
