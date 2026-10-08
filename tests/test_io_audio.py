import wave

import numpy as np
import pytest

from vismic.audio import write_wav
from vismic.io import crop, parse_roi


def test_roi_parser_and_crop() -> None:
    frame = np.arange(30).reshape(5, 6)
    np.testing.assert_array_equal(crop(frame, parse_roi("2,1,3,2")), frame[1:3, 2:5])
    with pytest.raises(ValueError):
        parse_roi("1,2,3")
    with pytest.raises(ValueError):
        crop(frame, (5, 4, 2, 2))


def test_write_wav(tmp_path) -> None:
    destination = tmp_path / "tone.wav"
    write_wav(destination, np.sin(np.linspace(0, 4 * np.pi, 200)), 1000)
    with wave.open(str(destination), "rb") as source:
        assert source.getframerate() == 1000
        assert source.getnframes() == 200
        assert source.getnchannels() == 1
