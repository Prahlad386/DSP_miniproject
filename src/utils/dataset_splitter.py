
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"

DATASET_SPLITS = {
    "train": "train_val",
    # "valid": "train_val",
    "test": "test",
}


def load_split(split_name, output_split):
    split_directory = RAW_DIRECTORY / f"nsynth-{split_name}"
    metadata_path = split_directory / "examples.json"
    audio_directory = split_directory / "audio"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    if not audio_directory.is_dir():
        raise FileNotFoundError(f"Missing audio directory: {audio_directory}")

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        examples = json.load(metadata_file)

    records = []

    for example_id, metadata in examples.items():
        audio_path = audio_directory / f"{example_id}.wav"

        if not audio_path.is_file():
            continue

        midi_pitch = int(metadata["pitch"])
        reference_f0 = 440.0 * 2 ** ((midi_pitch - 69) / 12)

        records.append({
            "audio_path": audio_path.relative_to(PROJECT_ROOT).as_posix(),
            "dataset": f"nsynth-{split_name}",
            "instrument_id": metadata["instrument_str"],
            "instrument_family": metadata["instrument_family_str"],
            "midi_pitch": midi_pitch,
            "reference_f0": reference_f0,
            "sample_rate": metadata["sample_rate"],
            "split": output_split,
        })

    print(
        f"Loaded {len(records)} recordings from nsynth-{split_name}; "
        f"{len(examples) - len(records)} missing audio files"
    )

    return records


def main():
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)

    development_records = []
    test_records = []

    for source_split, output_split in DATASET_SPLITS.items():
        records = load_split(source_split, output_split)

        if output_split == "test":
            test_records.extend(records)
        else:
            development_records.extend(records)

    development_instruments = {
        record["instrument_id"] for record in development_records
    }

    test_instruments = {
        record["instrument_id"] for record in test_records
    }

    overlapping_instruments = development_instruments & test_instruments

    if overlapping_instruments:
        raise ValueError(
            f"Instrument overlap detected: {sorted(overlapping_instruments)}"
        )

    if not development_records or not test_records:
        raise ValueError("Development and test datasets must both be non-empty.")

    fieldnames = [
        "audio_path",
        "dataset",
        "instrument_id",
        "instrument_family",
        "midi_pitch",
        "reference_f0",
        "sample_rate",
        "split",
    ]

    manifests = {
        "train_val_manifest.csv": development_records,
        "test_manifest.csv": test_records,
    }

    for filename, records in manifests.items():
        output_path = PROCESSED_DIRECTORY / filename

        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"Saved {len(records)} records to {output_path}")

    print("Instrument overlap check passed.")


if __name__ == "__main__":
    main()
