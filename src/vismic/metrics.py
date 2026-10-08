"""Lag-, scale-, and polarity-tolerant WAV comparison for benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from math import gcd
import json
from pathlib import Path
import wave

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate, correlation_lags, resample_poly, stft, welch


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AudioComparison:
    reference_rate: int
    lag_samples: int
    lag_seconds: float
    overlap_seconds: float
    absolute_correlation: float
    si_sdr_db: float
    log_spectral_correlation: float
    log_spectrogram_correlation: float


def read_wav(path: str | Path) -> tuple[FloatArray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("comparison currently supports mono 16-bit PCM WAV files")
        rate = source.getframerate()
        data = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return data.astype(np.float64) / 32768.0, rate


def _resample(signal: FloatArray, source_rate: int, target_rate: int) -> FloatArray:
    divisor = gcd(source_rate, target_rate)
    return resample_poly(signal, target_rate // divisor, source_rate // divisor)


def compare_signals(
    estimate: FloatArray,
    estimate_rate: int,
    reference: FloatArray,
    reference_rate: int,
    *,
    max_lag_seconds: float = 2.0,
    frequency_min: float = 50.0,
    frequency_max: float | None = 2000.0,
) -> AudioComparison:
    """Align two signals and report waveform and spectrum agreement."""

    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate_rate != reference_rate:
        estimate = _resample(estimate, estimate_rate, reference_rate)
    estimate -= np.mean(estimate)
    reference -= np.mean(reference)

    cross_correlation = correlate(estimate, reference, method="fft")
    lags = correlation_lags(len(estimate), len(reference))
    allowed = np.abs(lags) <= round(max_lag_seconds * reference_rate)
    if not np.any(allowed):
        raise ValueError("no lags fall inside max_lag_seconds")
    lag = int(lags[allowed][np.argmax(np.abs(cross_correlation[allowed]))])
    if lag >= 0:
        estimate, reference = estimate[lag:], reference
    else:
        estimate, reference = estimate, reference[-lag:]
    overlap = min(len(estimate), len(reference))
    if overlap < 16:
        raise ValueError("aligned signals have too little overlap")
    estimate, reference = estimate[:overlap], reference[:overlap]

    absolute_correlation = abs(float(np.corrcoef(estimate, reference)[0, 1]))
    projection = np.dot(estimate, reference) / max(
        float(np.dot(reference, reference)), np.finfo(float).eps
    )
    target = projection * reference
    residual = estimate - target
    si_sdr = 10.0 * np.log10(
        max(float(np.dot(target, target)), np.finfo(float).eps)
        / max(float(np.dot(residual, residual)), np.finfo(float).eps)
    )

    segment = min(4096, overlap)
    frequencies, estimate_psd = welch(estimate, reference_rate, nperseg=segment)
    _, reference_psd = welch(reference, reference_rate, nperseg=segment)
    maximum = frequency_max or reference_rate / 2.0
    selected = (frequencies >= frequency_min) & (frequencies <= maximum)
    if np.count_nonzero(selected) < 2:
        raise ValueError("frequency range contains too few spectral bins")
    log_spectral_correlation = float(
        np.corrcoef(
            np.log10(estimate_psd[selected] + np.finfo(float).tiny),
            np.log10(reference_psd[selected] + np.finfo(float).tiny),
        )[0, 1]
    )

    time_segment = min(overlap, max(32, round(reference_rate * 0.12)))
    time_overlap = time_segment * 3 // 4
    time_frequencies, _, estimate_stft = stft(
        estimate,
        reference_rate,
        nperseg=time_segment,
        noverlap=time_overlap,
    )
    _, _, reference_stft = stft(
        reference,
        reference_rate,
        nperseg=time_segment,
        noverlap=time_overlap,
    )
    time_selected = (time_frequencies >= frequency_min) & (time_frequencies <= maximum)
    estimate_magnitude = np.abs(estimate_stft[time_selected])
    reference_magnitude = np.abs(reference_stft[time_selected])
    estimate_floor = max(float(np.max(estimate_magnitude)) * 1e-5, np.finfo(float).tiny)
    reference_floor = max(float(np.max(reference_magnitude)) * 1e-5, np.finfo(float).tiny)
    log_spectrogram_correlation = float(
        np.corrcoef(
            np.log10(estimate_magnitude + estimate_floor).ravel(),
            np.log10(reference_magnitude + reference_floor).ravel(),
        )[0, 1]
    )
    return AudioComparison(
        reference_rate,
        lag,
        lag / reference_rate,
        overlap / reference_rate,
        absolute_correlation,
        float(si_sdr),
        log_spectral_correlation,
        log_spectrogram_correlation,
    )


def compare_wav(estimate: str | Path, reference: str | Path, **kwargs) -> AudioComparison:
    estimate_signal, estimate_rate = read_wav(estimate)
    reference_signal, reference_rate = read_wav(reference)
    return compare_signals(
        estimate_signal,
        estimate_rate,
        reference_signal,
        reference_rate,
        **kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a recovered WAV with ground truth")
    parser.add_argument("estimate")
    parser.add_argument("reference")
    parser.add_argument("--max-lag", type=float, default=2.0, metavar="SECONDS")
    parser.add_argument("--fmin", type=float, default=50.0, metavar="HZ")
    parser.add_argument("--fmax", type=float, default=2000.0, metavar="HZ")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = compare_wav(
        args.estimate,
        args.reference,
        max_lag_seconds=args.max_lag,
        frequency_min=args.fmin,
        frequency_max=args.fmax,
    )
    values = asdict(result)
    if args.json:
        print(json.dumps(values, indent=2))
    else:
        for name, value in values.items():
            print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
