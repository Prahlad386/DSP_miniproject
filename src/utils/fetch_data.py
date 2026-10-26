import os
import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

def verify_and_fetch_data():
    # Ensure data/raw directory exists
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data root directory verified at: {DATA_RAW_DIR}")
    
    # Check for existing audio files
    wav_files = list(DATA_RAW_DIR.glob("*.wav"))
    print(f"Found {len(wav_files)} raw .wav files in data/raw/")
    
    if len(wav_files) == 0:
        print("\n[NOTE] No .wav files found yet in data/raw/.")
        print("Please download your dataset files (e.g., NSynth / Bach10) and drop the .wav files into:")
        print(f" -> {DATA_RAW_DIR}\n")
    else:
        print("\nSample files ready for team processing:")
        for f in wav_files[:5]:  # Display up to first 5 files
            print(f" - {f.name}")

if __name__ == "__main__":
    verify_and_fetch_data()
