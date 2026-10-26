# src/utils/pitch_utils.py
import numpy as np
import sys
sys.path.append('.')
import config

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def hz_to_midi(f0: float) -> tuple[int, float]:
    """
    Converts physical frequency (Hz) to MIDI note number and cent deviation.
    
    Returns:
        midi_note (int): Nearest integer MIDI note index [1..128]
        cents_dev (float): Deviation from exact semitone pitch in cents [-50, +50]
    """
    if f0 <= 0 or np.isnan(f0):
        return 0, 0.0
    
    # Exact fractional MIDI note calculation
    midi_exact = 12.0 * np.log2(f0 / config.A4_FREQ) + config.A4_MIDI_NOTE
    midi_note = int(np.round(midi_exact))
    
    # Frequency corresponding to exact nearest MIDI note
    f_exact = config.A4_FREQ * (2.0 ** ((midi_note - config.A4_MIDI_NOTE) / 12.0))
    
    # Cent deviation formula: Δc = 1200 * log2(f0 / f_exact)
    cents_dev = 1200.0 * np.log2(f0 / f_exact)
    
    return midi_note, cents_dev

def midi_to_note_name(midi_note: int) -> str:
    """Converts a MIDI note number to musical note name string (e.g., 69 -> 'A4')."""
    if midi_note < 12 or midi_note > 127:
        return "N/A"
    note_idx = midi_note % 12
    octave = (midi_note // 12) - 1
    return f"{NOTE_NAMES[note_idx]}{octave}"

if __name__ == "__main__":
    # Sanity check
    f_test = 440.0
    note, cents = hz_to_midi(f_test)
    print(f"Test Frequency {f_test} Hz -> MIDI Note: {note} ({midi_to_note_name(note)}), Cent Deviation: {cents:.2f} cents")
