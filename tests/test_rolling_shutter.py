import numpy as np

from vismic.recover import RecoveryConfig
from vismic.rolling_shutter import (
    RollingShutterTiming,
    recover_rolling_shutter,
    resample_row_signals,
)
from vismic.synthetic import make_texture


def test_maps_rows_to_line_rate_and_marks_frame_gaps() -> None:
    timing = RollingShutterTiming(frame_rate=10.0, line_delay_seconds=0.01, rows=4)
    rows = np.arange(12, dtype=float).reshape(3, 4)
    result = resample_row_signals(rows, timing)

    assert result.sample_rate == 100.0
    assert np.count_nonzero(result.observed) == rows.size
    assert np.any(~result.observed)
    np.testing.assert_allclose(result.signal[:4], rows[0])


def test_recovers_synthetic_rolling_shutter_tone() -> None:
    frame_rate = 20.0
    line_rate = 800.0
    frequency = 73.0
    frame_count, height, width = 30, 24, 48
    texture = make_texture((height, width))
    horizontal_frequency = np.fft.fftfreq(width)[None, :]
    texture_spectrum = np.fft.fft(texture, axis=1)
    frames = []
    for frame in range(frame_count):
        row_times = frame / frame_rate + np.arange(height) / line_rate
        displacement = 0.01 * np.sin(2 * np.pi * frequency * row_times)
        ramp = np.exp(-2j * np.pi * displacement[:, None] * horizontal_frequency)
        frames.append(np.fft.ifft(texture_spectrum * ramp, axis=1).real)

    timing = RollingShutterTiming(frame_rate, 1.0 / line_rate, height)
    recovered = recover_rolling_shutter(
        np.asarray(frames),
        timing,
        RecoveryConfig(scales=2, orientations=2),
    )
    sample_times = np.arange(len(recovered.signal)) / line_rate
    expected = np.sin(2 * np.pi * frequency * sample_times)
    correlation = abs(float(np.corrcoef(recovered.signal, expected)[0, 1]))

    assert correlation > 0.85
    assert np.any(~recovered.observed)
