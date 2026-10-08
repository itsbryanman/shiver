"""Video and NumPy input helpers."""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
Roi = tuple[int, int, int, int]


def parse_roi(value: str) -> Roi:
    try:
        x, y, width, height = (int(part) for part in value.split(","))
    except (TypeError, ValueError) as exc:
        raise ValueError("ROI must be x,y,w,h") from exc
    if min(x, y) < 0 or width <= 0 or height <= 0:
        raise ValueError("ROI origin must be non-negative and size must be positive")
    return x, y, width, height


def crop(frame: NDArray, roi: Roi | None) -> NDArray:
    if roi is None:
        return frame
    x, y, width, height = roi
    if y + height > frame.shape[0] or x + width > frame.shape[1]:
        raise ValueError(f"ROI {roi} exceeds frame shape {frame.shape[:2]}")
    return frame[y : y + height, x : x + width]


def read_frames(path: str | Path, roi: Roi | None = None) -> tuple[Iterator[NDArray], float | None]:
    """Read .npy frame arrays or videos supported by imageio/FFmpeg."""

    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r")
        if array.ndim not in (3, 4):
            raise ValueError(".npy input must have shape (time, height, width[, channels])")
        return (crop(frame, roi) for frame in array), None
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            return _read_frames_ffmpeg(path, roi)
        raise RuntimeError(
            "video input needs FFmpeg on PATH or `pip install 'vismic[video]'`"
        ) from exc
    metadata = iio.immeta(path)
    fps = float(metadata["fps"]) if "fps" in metadata else None
    return (crop(frame, roi) for frame in iio.imiter(path)), fps


def _read_frames_ffmpeg(path: Path, roi: Roi | None) -> tuple[Iterator[NDArray], float | None]:
    """Stream grayscale frames through the system FFmpeg executable."""

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    source_width, source_height = int(stream["width"]), int(stream["height"])
    if roi is None:
        width, height = source_width, source_height
        video_filter = "format=gray"
    else:
        x, y, width, height = roi
        if x + width > source_width or y + height > source_height:
            raise ValueError(f"ROI {roi} exceeds frame shape {(source_height, source_width)}")
        video_filter = f"crop={width}:{height}:{x}:{y},format=gray"

    numerator, denominator = stream.get("avg_frame_rate", "0/1").split("/")
    fps = float(numerator) / float(denominator) if float(denominator) else None
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]

    def decoded() -> Iterator[NDArray]:
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        assert process.stdout is not None
        frame_bytes = width * height
        try:
            while True:
                raw = process.stdout.read(frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise RuntimeError("FFmpeg returned a truncated video frame")
                yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        finally:
            process.stdout.close()
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"FFmpeg exited with status {return_code}")

    return decoded(), fps
