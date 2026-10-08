# Validation log

This file records gates, including failures. Metrics are only comparable when the reference and frequency range match.

## Synthetic fractional-pixel motion — pass

- Signal: 31 Hz horizontal sinusoid, 0.001 pixel amplitude
- Capture: 500 samples/s, 64×64 deterministic texture, additive noise σ=0.001
- Gate: absolute correlation greater than 0.95
- Automated by `tests/test_synthetic.py`

## MIT Pentax rolling shutter, “The Raven” — provisional pass

- Official input: 60 fps (59.94 container rate), 1280×720 Motion JPEG, 1/2000 s exposure
- Calibration from the paper: 61,920 row samples/s, about 5 ms frame gap
- Processing ROI: `100,0,1000,720`; two horizontal analytic scales; 40–2000 Hz; frame-harmonic notches
- Against MIT recovered WAV after rate/lag alignment:
  - absolute waveform correlation: **0.6261**
  - SI-SDR: **−1.91 dB**
  - log-spectrum correlation, 50–2000 Hz: **0.8398**
  - log-spectrogram correlation, 50–2000 Hz: **0.7447**
- Against source WAV:
  - absolute waveform correlation: **0.4253**
  - log-spectrum correlation, 50–2000 Hz: **0.7793**
  - log-spectrogram correlation, 50–2000 Hz: **0.5712**

This proves that the calibrated row path extracts related signal. It does not establish speech intelligibility.

## MIT Pentax rolling shutter, MIDI — fail

The 173 MB public rolling-shutter MIDI clip was run because the 10.8 GB high-speed chips/MIDI clip is intentionally download-gated. Against the source WAV from 50–1000 Hz, absolute waveform correlation is 0.0792, aggregate log-spectrum correlation is 0.5182, and log-spectrogram correlation is only **0.2063**. Frame-harmonic removal can make the aggregate spectrum resemble MIT's result, but time–frequency agreement with the source remains poor: the recovered note sequence is not yet reliable. This gate stays failed until the log-spectrogram metric and visible note track improve, regardless of aggregate spectral correlation.

Likely next investigations:

1. Match the authors' row-motion estimator and missing-sample interpolation more closely.
2. Measure whether the published 61,920 Hz rate is exact for this encoded clip.
3. Test readout direction and the reported frame-delay boundary explicitly.
4. Compare several texture ROIs instead of summing most of the bag.

## Provenance

No MIT media or MATLAB source is committed. `scripts/fetch_mit_data.py` downloads assets from the [official data page](https://data.csail.mit.edu/vidmag/VisualMic/) into `data/mit/`, which is ignored by Git. Review upstream terms before redistribution.
