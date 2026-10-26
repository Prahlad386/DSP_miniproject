
import json
from pathlib import Path

raw_directory = Path("data/raw")

split_names = ["train", "valid", "test"]
instrument_sets = {}

for split_name in split_names:
    metadata_path = raw_directory / f"nsynth-{split_name}" / "examples.json"

    if not metadata_path.exists():
        print(f"{split_name}: metadata file not found")
        continue

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        examples = json.load(metadata_file)

    instrument_sets[split_name] = {
        metadata["instrument_str"]
        for metadata in examples.values()
    }

    print(
        f"{split_name}: {len(examples)} recordings, "
        f"{len(instrument_sets[split_name])} instruments"
    )

for first_split, second_split in [
    ("train", "valid"),
    ("train", "test"),
    ("valid", "test"),
]:
    if first_split in instrument_sets and second_split in instrument_sets:
        overlap = (
            instrument_sets[first_split]
            & instrument_sets[second_split]
        )

        print(
            f"{first_split} vs {second_split}: "
            f"{len(overlap)} shared instruments"
        )

        if overlap:
            print("Examples:", sorted(overlap)[:10])
