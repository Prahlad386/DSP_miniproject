
import numpy as np
from scipy.signal import get_window


def estimate_pitch(
    audio,
    sample_rate,
    frame_length=2048,
    hop_length=512,
    n_fft=4096,
    fmin=27.5,
    fmax=None,
    harmonics=5,
    minimum_rms=1e-4,
):
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("Frame length and hop length must be positive")

    if n_fft < frame_length:
        raise ValueError("n_fft must be at least frame_length")

    if harmonics < 1:
        raise ValueError("harmonics must be at least 1")

    nyquist_frequency = sample_rate / 2

    if fmax is None:
        fmax = min(4186.0, nyquist_frequency * 0.99)

    if not 0 < fmin < fmax < nyquist_frequency:
        raise ValueError(
            "Require 0 < fmin < fmax < sample_rate / 2. "
            "Check sample_rate, fmin, and fmax."
        )

    if audio.size == 0:
        return np.empty(0, dtype=np.float64)

    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))

    frame_starts = np.arange(
        0,
        audio.size - frame_length + 1,
        hop_length,
    )

    window = get_window("hann", frame_length, fftbins=True)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    candidate_step = sample_rate / n_fft / 4

    candidate_frequencies = np.arange(
        fmin,
        fmax,
        candidate_step,
    )

    if candidate_frequencies.size == 0:
        return np.zeros(len(frame_starts), dtype=np.float64)

    maximum_harmonics = max(
        1,
        min(harmonics, int(nyquist_frequency // fmin)),
    )

    harmonic_weights = np.array(
        [1.0 / harmonic for harmonic in range(1, maximum_harmonics + 1)]
    )

    estimates = []

    for frame_start in frame_starts:
        frame = audio[frame_start:frame_start + frame_length]

        if np.sqrt(np.mean(frame ** 2)) < minimum_rms:
            estimates.append(0.0)
            continue

        frame = frame - np.mean(frame)
        spectrum = np.abs(np.fft.rfft(frame * window, n=n_fft))
        spectrum /= np.max(spectrum) + 1e-12

        score_sums = np.zeros(candidate_frequencies.size)
        weight_sums = np.zeros(candidate_frequencies.size)

        for harmonic, weight in enumerate(harmonic_weights, start=1):
            harmonic_frequencies = candidate_frequencies * harmonic
            valid = harmonic_frequencies < nyquist_frequency

            magnitudes = np.zeros(candidate_frequencies.size)

            magnitudes[valid] = np.interp(
                harmonic_frequencies[valid],
                frequencies,
                spectrum,
            )

            score_sums[valid] += weight * np.log(
                magnitudes[valid] + 1e-12
            )
            weight_sums[valid] += weight

        scores = score_sums / np.maximum(weight_sums, 1e-12)
        estimates.append(float(candidate_frequencies[np.argmax(scores)]))

    return np.asarray(estimates, dtype=np.float64)
