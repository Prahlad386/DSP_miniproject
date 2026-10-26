# config.py
"""
Centralized DSP parameter configuration file.
Ensures identical specifications across baseline and proposed algorithms.
"""

import numpy as np

# Audio Processing Specs
FS = 16000              # Sampling frequency (Hz)
FRAME_LEN = 2048        # Window length N_w (samples)
HOP_LEN = 512           # Hop size N_h (samples)
N_FFT = 4096            # Zero-padded FFT size for higher frequency resolution
WINDOW_TYPE = 'hann'    # Analysis window function

# Pitch Bounds
F_MIN = 27.5            # Note A0 (Hz)
F_MAX = 4186.0          # Note C8 (Hz)

# MIDI Reference
A4_FREQ = 440.0         # Reference frequency for A4 (Hz)
A4_MIDI_NOTE = 69       # MIDI note number for A4
