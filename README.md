# vismic

## The demo this repo is building toward

| 1. Film | 2. Show the silent clip | 3. Play recovered sound |
| --- | --- | --- |
| A crinkled chip bag beside a speaker, camera locked to a tripod | The ROI barely appears to move | The sweep—and eventually speech—comes back from the pixels |

> [!WARNING]
> **Current vibe: the pixels are talking, but we do not speak fluent chip bag yet.** Synthetic sub-pixel recovery works, and the Raven benchmark produces a related signal. The MIDI note-sequence test still fails, so this is an active experiment—not a magic silent-video-to-audio button. Bring curiosity, not courtroom evidence.

More precisely: the phase pipeline recovers a known **0.001-pixel** sinusoidal shift from a noisy synthetic texture with greater than 0.95 absolute correlation. It also runs end-to-end on MIT's public 60 fps rolling-shutter Raven clip, reaching 0.626 waveform and 0.840 log-spectrum correlation with MIT's recovered reference after lag/rate alignment. The rolling-shutter MIDI note-sequence gate still fails; see the [validation log](docs/validation.md). A real chip-bag demo belongs above this paragraph as soon as it exists; there is intentionally no polished fake demo in its place.

`vismic` is a small, inspectable Python implementation of the core idea in Davis et al., [*The Visual Microphone: Passive Recovery of Sound from Video*](https://people.csail.mit.edu/mrub/VisualMic/) (SIGGRAPH 2014): tiny object vibrations become measurable as phase changes in complex spatial filters.

> [!IMPORTANT]
> Sample rate is physics, not a software option. In the high-speed/global-shutter path, one video frame produces one signal sample. A 240 fps clip cannot yield intelligible speech. Audio-band recovery from ordinary frame-rate video needs calibrated rolling-shutter line timing; `vismic` will not guess it.

## Run the first milestone

Python 3.10+ is required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'

vismic --synthetic --fps 1000 --frequency 73 --amplitude 0.001 -o out.wav
pytest
```

Expected terminal output includes a displacement `|correlation|` close to 1. Phase polarity is arbitrary, so the benchmark correctly compares absolute correlation.

For a high-speed video (system FFmpeg or the optional `imageio` dependency is required):

```bash
pip install -e '.[video]'
vismic in.mp4 --roi 120,80,256,256 --fps 2200 -o recovered.wav
```

Use the **capture** rate for `--fps`. Some high-speed cameras store misleading playback metadata in the container. You can also feed a NumPy array shaped `(time, height, width[, channels])`:

```bash
vismic frames.npy --fps 2200 -o recovered.wav
```

## What is implemented

The current CPU prototype is deliberately narrow:

1. Build a multi-scale, multi-orientation complex Fourier filter bank.
2. Measure local motion as phase change against the first frame.
3. Suppress flat/noisy regions using reference amplitude squared.
4. Cross-correlate band traces, correct lag and arbitrary polarity, then combine them.
5. High-pass, perform conservative spectral subtraction, and write mono PCM WAV.

The filter bank is an original compact log-Gabor/analytic prototype, not a byte-for-byte port of the authors' MATLAB complex steerable pyramid. That keeps the synthetic milestone easy to audit while preserving the relevant phase behavior. A Riesz-pyramid backend and GPU acceleration are planned once real-video correctness is measured.

Rolling-shutter mode extracts horizontal motion per sensor row, maps `(frame, row)` measurements onto the calibrated line clock, fills inter-frame holes with bidirectional autoregressive prediction, and retains an `observed` mask. It follows the paper's coherent horizontal-motion model, so the object/camera geometry still matters. General camera timing calibration remains open work.

## Reproduce the official rolling-shutter check

MIT's data page includes two rolling-shutter videos in addition to the much larger high-speed captures. The smaller Raven video is 32 MB, 1280×720 Motion JPEG at 59.94 fps. The paper reports about 16 μs line delay, about 5 ms frame delay, an effective 61,920 row samples/s, 1/2000 s exposure, and a roughly 2 kHz recoverable ceiling for this camera.

Third-party assets are downloaded to the ignored `data/mit/` directory and are never redistributed by this repository:

```bash
python scripts/fetch_mit_data.py references
python scripts/fetch_mit_data.py raven

vismic data/mit/RollingShutter/KitKat-60Hz-RollingShutter-Raven-input.avi \
  --roi 100,0,1000,720 \
  --rolling-shutter --line-rate 61920 --fps 59.94 \
  --scales 2 --orientations 2 --highpass 40 --lowpass 2000 \
  -o artifacts/raven-vismic.wav

vismic-compare artifacts/raven-vismic.wav \
  data/mit/RollingShutter/KitKat-60Hz-RollingShutter-Raven-input.wav \
  --fmax 2000 --json
```

The comparator resamples, finds the best lag, tolerates arbitrary phase polarity and gain, and reports absolute waveform correlation, SI-SDR, log-power-spectrum correlation, and time-varying log-spectrogram correlation. Use the source WAV and MIT's recovered WAV as separate references; they answer different questions.

The 2200 Hz chips/MIDI input is 10.8 GB. Its download requires an explicit size acknowledgement:

```bash
python scripts/fetch_mit_data.py chips-midi --allow-large
```

## Roadmap and gates

- [x] Deterministic sub-pixel synthetic generator
- [x] Phase recovery benchmark and tests
- [x] Streaming video/NumPy CLI, ROI crop, cleanup, WAV export
- [x] Per-row rolling-shutter phase extraction and explicit gap mask
- [x] Autoregressive inter-frame gap reconstruction
- [x] Provisional reproduction of the public MIT rolling-shutter Raven benchmark
- [ ] Pass the public rolling-shutter MIDI note-sequence gate
- [ ] Validate a filmed tone sweep and publish raw/recovered artifacts
- [ ] Calibrate line delay from camera metadata or a flicker target
- [ ] Recover speech
- [ ] Add the visual spectrogram and vibration-heatmap viewer
- [ ] Profile, then move the measured hot loop to CuPy/Rust/wgpu

## Capture checklist for the tone-sweep gate

- Lock exposure, focus, white balance, and stabilization; use a solid tripod.
- Fill the ROI with a textured, specular object such as a chip bag.
- Use bright continuous lighting and the shortest practical exposure.
- Record the emitted sweep separately as ground truth.
- Record the actual capture fps. For rolling shutter, also measure line delay and readout direction.
- Keep raw footage. Compression and rescaling can destroy the phase signal.

## Layout

```text
src/vismic/
  pyramid.py          complex oriented filter bank
  recover.py          phase extraction, weighting, alignment
  synthetic.py        known fractional-pixel benchmark
  rolling_shutter.py  per-row extraction, line clock, AR gap filling
  metrics.py          lag/rate-tolerant reference comparison
  audio.py            cleanup and WAV export
  io.py / cli.py      inputs, ROI, command line
tests/                 synthetic, timing, I/O tests
docs/validation.md     pass/fail benchmark record
```

## Scope, consent, and provenance

This is an experimental measurement tool, not a promise that arbitrary silent footage contains recoverable audio. Only process recordings you are authorized to analyze and follow local privacy and recording laws.

The algorithm is based on the paper and its public examples; this repository contains an independent implementation and no MIT sample footage or MATLAB code. The fetcher only downloads assets from their original host into a gitignored directory. See the [project page and samples](https://people.csail.mit.edu/mrub/VisualMic/), [official data and results](https://data.csail.mit.edu/vidmag/VisualMic/), and the [MIT open-access manuscript](https://dspace.mit.edu/handle/1721.1/100023).

## Citation

If this project is useful in research, cite the original work:

```bibtex
@article{Davis2014VisualMic,
  author  = {Abe Davis and Michael Rubinstein and Neal Wadhwa and
             Gautham J. Mysore and Fr\'{e}do Durand and William T. Freeman},
  title   = {The Visual Microphone: Passive Recovery of Sound from Video},
  journal = {ACM Transactions on Graphics (Proc. SIGGRAPH)},
  volume  = {33},
  number  = {4},
  pages   = {79:1--79:10},
  year    = {2014}
}
```

Code in this repository is MIT licensed. The linked manuscript and third-party data have their own terms.
