"""Per-row phase recovery and timing for rolling-shutter video."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_toeplitz

from .recover import RecoveryConfig, _frames, align_and_combine


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RollingShutterTiming:
    frame_rate: float
    line_delay_seconds: float
    rows: int

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.line_delay_seconds

    @property
    def readout_seconds(self) -> float:
        return self.rows * self.line_delay_seconds

    @property
    def frame_period_seconds(self) -> float:
        return 1.0 / self.frame_rate

    def validate(self) -> None:
        if self.frame_rate <= 0 or self.line_delay_seconds <= 0 or self.rows < 1:
            raise ValueError("frame_rate, line_delay_seconds, and rows must be positive")
        if self.readout_seconds > self.frame_period_seconds * 1.001:
            raise ValueError("row readout cannot be longer than the frame period")


@dataclass(frozen=True)
class RollingShutterSamples:
    signal: FloatArray
    sample_rate: float
    observed: NDArray[np.bool_]


@dataclass(frozen=True)
class RollingRecoveryResult:
    signal: FloatArray
    sample_rate: float
    observed: NDArray[np.bool_]
    row_signals: FloatArray
    band_signals: FloatArray


def _horizontal_filters(
    width: int,
    scales: int,
    base_frequency: float,
    radial_bandwidth_octaves: float,
) -> list[NDArray[np.complex128]]:
    """Build analytic 1D bands without mixing rows (and therefore times)."""

    frequency = np.fft.fftfreq(width)
    radius = np.maximum(np.abs(frequency), np.finfo(float).tiny)
    analytic = np.where(frequency > 0.0, 2.0, 0.0)
    filters = []
    for scale in range(scales):
        center = base_frequency / (2**scale)
        log_distance = np.log2(radius / center)
        radial = np.exp(-0.5 * (log_distance / radial_bandwidth_octaves) ** 2)
        radial[frequency == 0] = 0.0
        filters.append((radial * analytic).astype(np.complex128))
    return filters


def _decompose_rows(
    frame: FloatArray, filters: list[NDArray[np.complex128]]
) -> list[NDArray[np.complex128]]:
    spectrum = np.fft.fft(frame, axis=1)
    return [np.fft.ifft(spectrum * transfer[None, :], axis=1) for transfer in filters]


def resample_row_signals(
    row_signals: ArrayLike,
    timing: RollingShutterTiming,
    *,
    interpolation: str = "ar",
    max_frequency_hz: float | None = None,
) -> RollingShutterSamples:
    """Map (frame, row) measurements to a uniform line-rate timebase.

    Missing inter-frame samples are linearly interpolated for audio export and
    marked ``False`` in ``observed`` so callers can visualize or replace them.
    """

    timing.validate()
    values = np.asarray(row_signals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != timing.rows:
        raise ValueError(f"row_signals must have shape (frames, {timing.rows})")
    frame_times = np.arange(values.shape[0])[:, None] / timing.frame_rate
    row_times = np.arange(timing.rows)[None, :] * timing.line_delay_seconds
    times = (frame_times + row_times).ravel()
    samples = values.ravel()
    # Convert to the integer line clock before allocating/marking.  Doing the
    # indexing through floating-point search can collapse e.g. 0.11 and 0.12
    # seconds onto the same bin because neither is exactly representable.
    measured_indices = np.rint(times * timing.sample_rate).astype(np.int64)
    uniform_times = np.arange(int(measured_indices[-1]) + 1) / timing.sample_rate
    observed = np.zeros(len(uniform_times), dtype=bool)
    observed[measured_indices] = True
    interpolated = np.interp(uniform_times, times, samples)
    if interpolation == "ar":
        interpolated = _interpolate_gaps_ar(interpolated, observed)
    elif interpolation == "bandlimited":
        if max_frequency_hz is None or max_frequency_hz <= 0:
            raise ValueError("bandlimited interpolation needs max_frequency_hz")
        interpolated = _interpolate_gaps_bandlimited(
            interpolated,
            observed,
            timing.sample_rate,
            max_frequency_hz,
        )
    elif interpolation != "linear":
        raise ValueError("interpolation must be 'ar', 'bandlimited', or 'linear'")
    return RollingShutterSamples(interpolated, timing.sample_rate, observed)


def _ar_coefficients(context: FloatArray, order: int) -> FloatArray | None:
    centered = context - np.mean(context)
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    correlation /= max(len(centered), 1)
    if correlation[0] <= np.finfo(float).eps:
        return None
    correlation[0] *= 1.0001
    try:
        return solve_toeplitz(correlation[:order], correlation[1 : order + 1])
    except np.linalg.LinAlgError:
        return None


def _predict_ar(context: FloatArray, count: int, order: int) -> FloatArray | None:
    coefficients = _ar_coefficients(context, order)
    if coefficients is None:
        return None
    mean = float(np.mean(context))
    history = list(context - mean)
    limit = max(5.0 * float(np.std(context)), np.finfo(float).eps)
    prediction = np.empty(count, dtype=np.float64)
    for index in range(count):
        value = float(np.dot(coefficients, history[-order:][::-1]))
        value = float(np.clip(value, -limit, limit))
        history.append(value)
        prediction[index] = value + mean
    return prediction


def _interpolate_gaps_ar(linear: FloatArray, observed: NDArray[np.bool_]) -> FloatArray:
    """Bidirectionally predict periodic gaps with local AR models."""

    output = linear.copy()
    padded = np.r_[True, observed, True]
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(transitions[::2], transitions[1::2]):
        gap_length = stop - start
        left_observed = np.flatnonzero(observed[:start])
        right_observed = np.flatnonzero(observed[stop:]) + stop
        if not len(left_observed) or not len(right_observed):
            continue
        left_end = int(left_observed[-1]) + 1
        left_start = left_end
        while left_start > 0 and observed[left_start - 1]:
            left_start -= 1
        right_start = int(right_observed[0])
        right_end = right_start
        while right_end < len(observed) and observed[right_end]:
            right_end += 1
        left = output[max(left_start, left_end - 512) : left_end]
        right = output[right_start : min(right_end, right_start + 512)]
        order = min(64, len(left) // 3, len(right) // 3)
        if order < 4:
            continue
        forward = _predict_ar(left, gap_length, order)
        backward = _predict_ar(right[::-1], gap_length, order)
        if forward is None or backward is None:
            continue
        weight = np.linspace(0.0, 1.0, gap_length + 2)[1:-1]
        output[start:stop] = (1.0 - weight) * forward + weight * backward[::-1]
    return output


def _interpolate_gaps_bandlimited(
    initial: FloatArray,
    observed: NDArray[np.bool_],
    sample_rate: float,
    max_frequency_hz: float,
    iterations: int = 24,
) -> FloatArray:
    """Reconstruct missing samples by alternating band and data projections."""

    if max_frequency_hz >= sample_rate / 2.0:
        raise ValueError("max_frequency_hz must be below Nyquist")
    output = initial.copy()
    known = output[observed].copy()
    frequencies = np.fft.rfftfreq(len(output), 1.0 / sample_rate)
    # A short cosine transition avoids the ringing of a brick-wall projector.
    transition_start = max_frequency_hz * 0.9
    spectral_mask = np.ones_like(frequencies)
    transition = (frequencies > transition_start) & (frequencies < max_frequency_hz)
    spectral_mask[frequencies >= max_frequency_hz] = 0.0
    spectral_mask[transition] = 0.5 * (
        1.0
        + np.cos(
            np.pi
            * (frequencies[transition] - transition_start)
            / (max_frequency_hz - transition_start)
        )
    )
    for _ in range(iterations):
        output = np.fft.irfft(np.fft.rfft(output) * spectral_mask, n=len(output))
        output[observed] = known
    return output


def recover_rolling_shutter(
    video: ArrayLike,
    timing: RollingShutterTiming,
    config: RecoveryConfig | None = None,
    *,
    max_frequency_hz: float | None = None,
) -> RollingRecoveryResult:
    """Recover one horizontal phase-motion sample per sensor row.

    This follows the coherent horizontal-motion model from section 6 of the
    paper. Keep the complete sensor-height ROI: cropping rows changes their
    absolute capture times and needs a corresponding timing offset.
    """

    timing.validate()
    config = config or RecoveryConfig(scales=2, orientations=2)
    frames, first = _frames(video)
    if first.shape[0] != timing.rows:
        raise ValueError(f"timing describes {timing.rows} rows but frames contain {first.shape[0]}")

    filters = _horizontal_filters(
        first.shape[1],
        config.scales,
        config.base_frequency,
        config.radial_bandwidth_octaves,
    )
    reference = _decompose_rows(first, filters)
    amplitude2 = [np.abs(response) ** 2 for response in reference]
    masks: list[NDArray[np.bool_]] = []
    for weights in amplitude2:
        threshold = np.quantile(weights, config.amplitude_floor_quantile)
        masks.append(weights > threshold)

    per_band: list[list[FloatArray]] = [[] for _ in filters]
    for frame in frames:
        responses = _decompose_rows(frame, filters)
        for index, response in enumerate(responses):
            phase_delta = np.angle(response * np.conj(reference[index]))
            weights = amplitude2[index] * masks[index]
            denominator = np.sum(weights, axis=1)
            numerator = np.sum(weights * phase_delta, axis=1)
            row_trace = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0,
            )
            per_band[index].append(row_trace)

    band_rows = np.asarray(per_band, dtype=np.float64)
    # Comparing with one rolling reference frame adds its fixed row waveform
    # to every frame. Remove that periodic reference term before flattening.
    band_rows -= np.mean(band_rows, axis=1, keepdims=True)
    # Tripod/camera drift contributes a different constant horizontal phase
    # to each frame. It becomes a strong comb at the frame rate after row
    # concatenation, while audio above the frame rate varies within a frame.
    band_rows -= np.mean(band_rows, axis=2, keepdims=True)
    band_signals = band_rows.reshape(len(filters), -1)
    band_weights = np.asarray([float(np.mean(weights)) for weights in amplitude2])
    combined, _, _ = align_and_combine(
        band_signals,
        band_weights,
        max_lag=config.max_alignment_lag,
    )
    row_signals = combined.reshape(band_rows.shape[1:])
    sampled = resample_row_signals(
        row_signals,
        timing,
        interpolation="bandlimited" if max_frequency_hz else "ar",
        max_frequency_hz=max_frequency_hz,
    )
    return RollingRecoveryResult(
        sampled.signal,
        sampled.sample_rate,
        sampled.observed,
        row_signals,
        band_signals,
    )
