import numpy as np

from vismic.recover import RecoveryConfig, recover_signal
from vismic.synthetic import make_synthetic_clip


def test_recovers_thousandth_pixel_sine() -> None:
    clip = make_synthetic_clip(
        shape=(64, 64),
        fps=500,
        duration=0.4,
        frequency=31,
        amplitude_pixels=0.001,
        noise_std=0.001,
    )
    result = recover_signal(clip.frames, RecoveryConfig(scales=2, orientations=4))

    correlation = abs(float(np.corrcoef(result.signal, clip.displacement)[0, 1]))
    assert correlation > 0.95
    assert result.band_signals.shape == (8, len(clip.frames))


def test_rejects_temporal_aliasing_in_synthetic_generator() -> None:
    try:
        make_synthetic_clip(fps=100, frequency=51)
    except ValueError as error:
        assert "twice" in str(error)
    else:
        raise AssertionError("expected Nyquist validation")
