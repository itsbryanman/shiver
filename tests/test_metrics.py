import numpy as np

from vismic.metrics import compare_signals


def test_comparison_handles_rate_lag_scale_and_polarity() -> None:
    reference_rate = 1000
    time = np.arange(1000) / reference_rate
    reference = np.sin(2 * np.pi * 73 * time) + 0.3 * np.sin(2 * np.pi * 131 * time)
    estimate = -2.5 * np.r_[np.zeros(100), reference]

    result = compare_signals(
        estimate,
        reference_rate,
        reference,
        reference_rate,
        max_lag_seconds=0.2,
        frequency_max=300,
    )

    assert result.lag_samples == 100
    assert result.absolute_correlation > 0.999
    assert result.si_sdr_db > 50
    assert result.log_spectral_correlation > 0.999
    assert result.log_spectrogram_correlation > 0.999
