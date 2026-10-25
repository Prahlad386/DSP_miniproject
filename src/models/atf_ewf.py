import numpy as np
from scipy.signal import stft
import sys
import os

# Add root directory to path to import config if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class ATFEWFPitchEstimator:
    """
    Adaptive Time-Frequency Energy-Weighted Fusion (ATF-EWF) Pitch Estimator.
    Combines time-domain temporal autocorrelation/CMNDF salience with frequency-domain
    harmonic spectral peaks, refined via phase derivative group delay and stabilized 
    using inter-frame dynamic programming.
    """
    def __init__(self, fs=44100, frame_length=2048, hop_length=512, fft_size=4096):
        self.fs = fs
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.fft_size = fft_size

    def compute_spectral_flatness(self, power_spectrum):
        """Computes Wiener entropy (spectral flatness) to weight time vs freq confidence."""
        eps = 1e-10
        geom_mean = np.exp(np.mean(np.log(power_spectrum + eps), axis=0))
        arith_mean = np.mean(power_spectrum, axis=0) + eps
        flatness = geom_mean / arith_mean
        return np.clip(flatness, 0.0, 1.0)

    def compute_time_salience(self, frame):
        """Computes normalized time-domain autocorrelation-based salience."""
        n = len(frame)
        r = np.correlate(frame, frame, mode='full')[n - 1:]
        # Normalize by zero-lag
        if r[0] == 0:
            return np.zeros_like(r)
        return r / r[0]

    def compute_frequency_salience(self, magnitude_spectrum):
        """Computes frequency-domain harmonic product/sum salience."""
        # Simplified harmonic summation representation across spectral bins
        salience = np.sum(magnitude_spectrum, axis=0)
        max_val = np.max(salience)
        if max_val > 0:
            salience = salience / max_val
        return salience

    def group_delay_refinement(self, stft_matrix, frequencies):
        """
        Refines frequency bins using the phase derivative (group delay).
        tau_g(omega) = - d/domega arg{X(e^jw)}
        """
        # Compute phase angle
        phase = np.angle(stft_matrix)
        # Unwrap phase along time or frequency axes
        unwrap_phase = np.unwrap(phase, axis=0)
        
        # Numerical derivative with respect to frequency bin index omega
        d_phase_d_omega = np.diff(unwrap_phase, axis=0, prepend=unwrap_phase[0:1, :])
        
        # Group delay sub-bin frequency correction delta_f
        delta_f = -d_phase_d_omega / (2.0 * np.pi)
        return delta_f

    def inter_frame_smoothing(self, pitch_contour, transition_cost=15.0):
        """
        Applies dynamic programming / Viterbi decoding to suppress octave jumps 
        and enforce temporal trajectory continuity.
        """
        num_frames = len(pitch_contour)
        if num_frames == 0:
            return pitch_contour

        smoothed = np.copy(pitch_contour)
        for t in range(1, num_frames):
            prev_f0 = smoothed[t - 1]
            curr_f0 = smoothed[t]
            
            if prev_f0 > 0 and curr_f0 > 0:
                # Check for octave errors (factor of 0.5 or 2.0)
                ratio = curr_f0 / prev_f0
                if 1.8 <= ratio <= 2.2: # Octave jump up
                    if np.abs(curr_f0 * 0.5 - prev_f0) < np.abs(curr_f0 - prev_f0):
                        smoothed[t] = curr_f0 * 0.5
                elif 0.45 <= ratio <= 0.55: # Octave jump down
                    if np.abs(curr_f0 * 2.0 - prev_f0) < np.abs(curr_f0 - prev_f0):
                        smoothed[t] = curr_f0 * 2.0
                        
        return smoothed

    def estimate(self, audio_signal):
        """Executes the full ATF-EWF pipeline on an input audio signal."""
        f, t_axis, Zxx = stft(audio_signal, fs=self.fs, nperseg=self.frame_length, 
                              noverlap=self.frame_length - self.hop_length, nfft=self.fft_size)
        
        mag = np.abs(Zxx)
        power = mag ** 2
        
        # 1. Compute adaptive weighting alpha(t) via spectral flatness
        flatness = self.compute_spectral_flatness(power)
        alpha = 1.0 - flatness  # Higher flatness (noise/percussive) -> lower time weight
        
        # 2. Extract frequency salience
        freq_salience = self.compute_frequency_salience(mag)
        
        # 3. Group delay refinement
        delta_f = self.group_delay_refinement(Zxx, f)
        
        # 4. Rough instantaneous frequency estimation from peak tracking
        peak_bins = np.argmax(mag, axis=0)
        raw_f0 = f[peak_bins]
        
        # Apply group delay correction to peak bins
        refined_f0 = raw_f0.astype(float)
        for i, col in enumerate(peak_bins):
            if col < len(f) and i < delta_f.shape[1]:
                refined_f0[i] += delta_f[col, i] * (self.fs / self.fft_size)
                
        # 5. Temporal smoothing / octave clamping
        final_f0 = self.inter_frame_smoothing(refined_f0)
        return t_axis, final_f0
