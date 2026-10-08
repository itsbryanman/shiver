"""Command-line entry point."""

from __future__ import annotations

import argparse
from itertools import chain
from pathlib import Path

import numpy as np

from .audio import clean_signal, write_wav
from .io import parse_roi, read_frames
from .recover import RecoveryConfig, recover_signal
from .rolling_shutter import RollingShutterTiming, recover_rolling_shutter
from .synthetic import make_synthetic_clip


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vismic",
        description="Recover a vibration waveform from phase changes in video.",
    )
    parser.add_argument("input", nargs="?", help="video or (time,height,width) .npy file")
    parser.add_argument("--roi", metavar="X,Y,W,H", help="crop before processing")
    parser.add_argument("-o", "--output", default="out.wav", help="output mono WAV")
    parser.add_argument("--fps", type=float, help="override/declare capture frame rate")
    parser.add_argument("--scales", type=int, default=3)
    parser.add_argument("--orientations", type=int, default=4)
    parser.add_argument("--highpass", type=float, default=40.0, metavar="HZ")
    parser.add_argument("--lowpass", type=float, metavar="HZ")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument(
        "--rolling-shutter",
        action="store_true",
        help="recover one sample per sensor row instead of per frame",
    )
    parser.add_argument(
        "--line-rate",
        type=float,
        metavar="HZ",
        help="calibrated rolling-shutter row rate (for example 61920)",
    )
    parser.add_argument("--synthetic", action="store_true", help="run the built-in benchmark")
    parser.add_argument("--frequency", type=float, default=73.0, help="synthetic tone frequency")
    parser.add_argument("--amplitude", type=float, default=0.001, help="synthetic shift in pixels")
    parser.add_argument("--duration", type=float, default=0.5, help="synthetic duration in seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    expected = None
    if args.synthetic:
        fps = args.fps or 1000.0
        clip = make_synthetic_clip(
            fps=fps,
            duration=args.duration,
            frequency=args.frequency,
            amplitude_pixels=args.amplitude,
        )
        frames = iter(clip.frames)
        expected = clip.displacement
    else:
        if not args.input:
            _parser().error("INPUT is required unless --synthetic is used")
        roi = parse_roi(args.roi) if args.roi else None
        frames, detected_fps = read_frames(args.input, roi)
        fps = args.fps or detected_fps
        if fps is None:
            _parser().error("--fps is required when the input has no frame-rate metadata")

    config = RecoveryConfig(scales=args.scales, orientations=args.orientations)
    output_rate = fps
    if args.rolling_shutter:
        if args.synthetic:
            _parser().error("--rolling-shutter cannot be combined with --synthetic")
        if not args.line_rate or args.line_rate <= 0:
            _parser().error("--line-rate is required for rolling-shutter recovery")
        frame_iterator = iter(frames)
        try:
            first = next(frame_iterator)
        except StopIteration:
            _parser().error("the input contains no video frames")
        frames = chain([first], frame_iterator)
        timing = RollingShutterTiming(
            frame_rate=fps,
            line_delay_seconds=1.0 / args.line_rate,
            rows=first.shape[0],
        )
        result = recover_rolling_shutter(
            frames,
            timing,
            config,
            max_frequency_hz=args.lowpass,
        )
        output_rate = result.sample_rate
    else:
        result = recover_signal(frames, config)
    cleaned = clean_signal(
        result.signal,
        output_rate,
        highpass_hz=args.highpass,
        lowpass_hz=args.lowpass,
        frame_harmonics_hz=fps if args.rolling_shutter else None,
        spectral_subtraction=0.0 if args.no_denoise else 1.0,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output, cleaned, output_rate)
    print(f"wrote {len(cleaned)} samples at {output_rate:g} Hz to {output}")
    if expected is not None:
        correlation = abs(float(np.corrcoef(result.signal, expected)[0, 1]))
        print(f"synthetic displacement |correlation|: {correlation:.5f} (polarity is arbitrary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
