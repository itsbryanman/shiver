"""Audio cleanup and standard-library WAV output."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, iirnotch, istft, sosfiltfilt, stft, tf2sos


FloatArray = NDArray[np.float64]


def clean_signal(
    signal: ArrayLike,
    sample_rate: float,
    *,
    highpass_hz: float = 40.0,
    lowpass_hz: float | None = None,
    frame_harmonics_hz: float | None = None,
    spectral_subtraction: float = 1.0,
) -> FloatArray:
    """Remove drift and apply conservative magnitude spectral subtraction."""

    data = np.asarray(signal, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    data = data - np.mean(data)
    if highpass_hz > 0 and len(data) >= 16:
        cutoff = min(highpass_hz, sample_rate * 0.45)
        sos = butter(3, cutoff, btype="highpass", fs=sample_rate, output="sos")
        data = sosfiltfilt(sos, data)
    if lowpass_hz is not None and lowpass_hz > 0 and len(data) >= 16:
        cutoff = min(lowpass_hz, sample_rate * 0.45)
        sos = butter(5, cutoff, btype="lowpass", fs=sample_rate, output="sos")
        data = sosfiltfilt(sos, data)
    if frame_harmonics_hz is not None and frame_harmonics_hz > 0 and len(data) >= 16:
        maximum = lowpass_hz or sample_rate * 0.45
        for harmonic in range(1, int(maximum / frame_harmonics_hz) + 1):
            frequency = harmonic * frame_harmonics_hz
            if frequency >= sample_rate / 2.0:
                break
            numerator, denominator = iirnotch(frequency, 50.0, fs=sample_rate)
            data = sosfiltfilt(tf2sos(numerator, denominator), data)

    if spectral_subtraction > 0 and len(data) >= 64:
        segment = min(512, max(32, 2 ** int(np.floor(np.log2(len(data) / 4)))))
        _, _, spectrum = stft(data, fs=sample_rate, nperseg=segment, noverlap=segment // 2)
        noise_frames = max(1, spectrum.shape[1] // 10)
        noise = np.median(np.abs(spectrum[:, :noise_frames]), axis=1, keepdims=True)
        magnitude = np.maximum(
            np.abs(spectrum) - spectral_subtraction * noise, 0.05 * np.abs(spectrum)
        )
        cleaned_spectrum = magnitude * np.exp(1j * np.angle(spectrum))
        _, data = istft(cleaned_spectrum, fs=sample_rate, nperseg=segment, noverlap=segment // 2)
        data = data[: len(signal)]

    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    return data / peak if peak > 0 else data


def write_wav(path: str | Path, signal: ArrayLike, sample_rate: float) -> None:
    """Write normalized mono 16-bit PCM without an extra audio dependency."""

    data = np.asarray(signal, dtype=np.float64)
    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    if peak > 1.0:
        data = data / peak
    pcm = np.round(np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(round(sample_rate))
        output.writeframes(pcm.tobytes())
