#!/usr/bin/env python3
"""Fetch official Visual Microphone validation assets without vendoring them."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen


DATA_BASE = "https://data.csail.mit.edu/vidmag/VisualMic"
CODE_URL = "https://people.csail.mit.edu/abedavis/research/VisMic/VMSlim.zip"

REFERENCE_AUDIO = [
    "Results/Chips2-2200Hz-Mary_MIDI-input.wav",
    "Results/Chips2-2200Hz-Mary_MIDI-recovered.wav",
    "Results/Plant-2200Hz-Mary_MIDI-input.wav",
    "Results/Plant-2200Hz-Mary_MIDI-recovered.wav",
    "Results/Chips1-2200Hz-Mary_Had-input_resampled_to_video_rate.wav",
    "Results/Chips1-2200Hz-Mary_Had-recovered.wav",
    "Results/Chips1-20000Hz-Mary_Had-input_resampled_to_video_rate.wav",
    "Results/Chips1-20000Hz-Mary_Had-recovered.wav",
    "RollingShutter/KitKat-60Hz-RollingShutter-Raven-input.wav",
    "RollingShutter/KitKat-60Hz-RollingShutter-Raven-recovered.wav",
    "RollingShutter/KitKat-60Hz-RollingShutter-Mary_MIDI-input.wav",
    "RollingShutter/KitKat-60Hz-RollingShutter-Mary_MIDI-recovered.wav",
]

VIDEOS = {
    "raven": "RollingShutter/KitKat-60Hz-RollingShutter-Raven-input.avi",
    "rolling-midi": "RollingShutter/KitKat-60Hz-RollingShutter-Mary_MIDI-input.avi",
    "chips-midi": "Results/Chips2-2200Hz-Mary_MIDI-input.avi",
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url) as response, partial.open("wb") as output:
        size = int(response.headers.get("Content-Length", 0))
        if destination.exists() and size and destination.stat().st_size == size:
            print(f"already present: {destination}")
            partial.unlink(missing_ok=True)
            return
        print(f"downloading {url} ({size / 1_000_000:.1f} MB)")
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=["references", "matlab", *VIDEOS],
        help="small reference WAVs, MATLAB code, or one input video",
    )
    parser.add_argument("--output", type=Path, default=Path("data/mit"))
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="required for the 10.8 GB high-speed chips MIDI video",
    )
    args = parser.parse_args()

    print("Upstream data/code remain MIT's assets; review their page and license before reuse.")
    if args.dataset == "references":
        for relative in REFERENCE_AUDIO:
            download(f"{DATA_BASE}/{relative}", args.output / relative)
    elif args.dataset == "matlab":
        download(CODE_URL, args.output / "VMSlim.zip")
    else:
        if args.dataset == "chips-midi" and not args.allow_large:
            parser.error("chips-midi is 10.8 GB; repeat with --allow-large to download it")
        relative = VIDEOS[args.dataset]
        download(f"{DATA_BASE}/{relative}", args.output / relative)
    return 0


if __name__ == "__main__":
    sys.exit(main())
