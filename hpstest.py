
import numpy as np
from src.models.hps import estimate_pitch

sample_rate = 1010
duration = 0.5
time = np.arange(int(sample_rate * duration)) / sample_rate
audio = np.sin(2 * np.pi * 440 * time)

estimates = estimate_pitch(
    audio,
    sample_rate=sample_rate,
    fmin=100,
)

valid_estimates = estimates[estimates > 0]

print("Expected frequency:", 440.0)
print(
    "Estimated frequency:",
    np.median(valid_estimates) if valid_estimates.size else "No pitch detected",
)
