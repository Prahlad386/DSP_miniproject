
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hps import estimate_pitch
from src.models.yin import YINEstimator


FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_FFT = 4096
MINIMUM_FREQUENCY = 27.5
MAXIMUM_FREQUENCY = 4186.0
YIN_THRESHOLD = 0.10
GROSS_ERROR_THRESHOLD_CENTS = 50.0
OCTAVE_TOLERANCE_CENTS = 50.0


def resolve_path(path_value):
    supplied_path = Path(str(path_value))

    candidates = [
        supplied_path,
        PROJECT_ROOT / supplied_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Audio file not found: {path_value}")


def load_audio(audio_path):
    sample_rate, audio = wavfile.read(audio_path)

    if audio.ndim > 1:
        audio = audio.astype(np.float64).mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        integer_limits = np.iinfo(audio.dtype)
        audio = audio.astype(np.float64)

        if integer_limits.min < 0:
            audio /= max(abs(integer_limits.min), integer_limits.max)
        else:
            midpoint = (integer_limits.max + 1) / 2
            audio = (audio - midpoint) / midpoint
    else:
        audio = audio.astype(np.float64)

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    return int(sample_rate), audio


def calculate_metrics(predictions, reference_frequency):
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)

    valid_predictions = np.isfinite(predictions) & (predictions > 0)
    valid_frequencies = predictions[valid_predictions]

    metrics = {
        "total_frames": int(predictions.size),
        "valid_pitch_frames": int(valid_predictions.sum()),
        "pitch_coverage_pct": (
            100.0 * valid_predictions.mean()
            if predictions.size
            else np.nan
        ),
        "gross_pitch_error_pct": np.nan,
        "octave_error_pct": np.nan,
        "fine_pitch_rmse_cents": np.nan,
        "median_abs_error_cents": np.nan,
    }

    if valid_frequencies.size == 0 or reference_frequency <= 0:
        return metrics

    cents_errors = 1200.0 * np.log2(
        valid_frequencies / reference_frequency
    )
    absolute_errors = np.abs(cents_errors)

    octave_multiples = np.rint(absolute_errors / 1200.0)

    octave_errors = (
        (octave_multiples >= 1)
        & (
            np.abs(
                absolute_errors - octave_multiples * 1200.0
            )
            <= OCTAVE_TOLERANCE_CENTS
        )
    )

    correctly_estimated = (
        absolute_errors <= GROSS_ERROR_THRESHOLD_CENTS
    )

    metrics["gross_pitch_error_pct"] = float(
        100.0
        * np.mean(
            absolute_errors > GROSS_ERROR_THRESHOLD_CENTS
        )
    )

    metrics["octave_error_pct"] = float(
        100.0 * np.mean(octave_errors)
    )

    if np.any(correctly_estimated):
        metrics["fine_pitch_rmse_cents"] = float(
            np.sqrt(
                np.mean(
                    cents_errors[correctly_estimated] ** 2
                )
            )
        )

    metrics["median_abs_error_cents"] = float(
        np.median(absolute_errors)
    )

    return metrics


def prepare_analysis_audio(audio):
    if audio.size < FRAME_LENGTH:
        return np.pad(
            audio,
            (0, FRAME_LENGTH - audio.size),
        )

    return audio


def evaluate_yin(audio, sample_rate, maximum_frequency):
    estimator = YINEstimator(
        fs=sample_rate,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        fmin=MINIMUM_FREQUENCY,
        fmax=maximum_frequency,
        threshold=YIN_THRESHOLD,
    )

    predictions, voiced_flags, confidence_scores = estimator.predict(audio)

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    ).reshape(-1)

    voiced_flags = np.asarray(
        voiced_flags,
        dtype=bool,
    ).reshape(-1)

    confidence_scores = np.asarray(
        confidence_scores,
        dtype=np.float64,
    ).reshape(-1)

    if predictions.size != voiced_flags.size:
        raise ValueError(
            "YIN frequency and voicing arrays have different lengths"
        )

    predictions[~voiced_flags] = 0.0
    predictions[
        ~np.isfinite(predictions) | (predictions <= 0)
    ] = 0.0

    finite_confidence = confidence_scores[
        np.isfinite(confidence_scores)
    ]

    mean_confidence = (
        float(np.mean(finite_confidence))
        if finite_confidence.size
        else np.nan
    )

    return predictions, mean_confidence


def evaluate_hps(audio, sample_rate, maximum_frequency):
    predictions = estimate_pitch(
        audio,
        sample_rate=sample_rate,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        fmin=MINIMUM_FREQUENCY,
        fmax=maximum_frequency,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    ).reshape(-1)

    predictions[
        ~np.isfinite(predictions) | (predictions <= 0)
    ] = 0.0

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/processed/test_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/evaluation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )
    arguments = parser.parse_args()

    manifest_path = Path(arguments.manifest)

    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    output_directory = Path(arguments.output_dir)

    if not output_directory.is_absolute():
        output_directory = PROJECT_ROOT / output_directory

    output_directory.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)

    required_columns = {"audio_path", "reference_f0"}

    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )

    if arguments.limit > 0:
        manifest = manifest.head(arguments.limit)

    recording_results = []
    error_results = []

    pooled_predictions = {
        "YIN": [],
        "HPS": [],
    }

    pooled_references = {
        "YIN": [],
        "HPS": [],
    }

    total_recordings = len(manifest)

    for recording_number, (_, recording) in enumerate(
        manifest.iterrows(),
        start=1,
    ):
        base_metadata = recording.to_dict()
        audio_path_value = recording["audio_path"]

        try:
            audio_path = resolve_path(audio_path_value)
            sample_rate, audio = load_audio(audio_path)
            audio = prepare_analysis_audio(audio)

            reference_frequency = float(recording["reference_f0"])

            if not np.isfinite(reference_frequency) or reference_frequency <= 0:
                raise ValueError(
                    f"Invalid reference frequency: {reference_frequency}"
                )

            maximum_frequency = min(
                MAXIMUM_FREQUENCY,
                sample_rate * 0.495,
            )

            if not MINIMUM_FREQUENCY < maximum_frequency:
                raise ValueError(
                    f"Sample rate is too low for the configured pitch range: "
                    f"{sample_rate} Hz"
                )

            model_outputs = {}

            try:
                yin_predictions, mean_confidence = evaluate_yin(
                    audio,
                    sample_rate,
                    maximum_frequency,
                )

                model_outputs["YIN"] = (
                    yin_predictions,
                    {"mean_confidence": mean_confidence},
                )

            except Exception as error:
                error_results.append({
                    "audio_path": str(audio_path_value),
                    "method": "YIN",
                    "error": repr(error),
                })

            try:
                hps_predictions = evaluate_hps(
                    audio,
                    sample_rate,
                    maximum_frequency,
                )

                model_outputs["HPS"] = (
                    hps_predictions,
                    {},
                )

            except Exception as error:
                error_results.append({
                    "audio_path": str(audio_path_value),
                    "method": "HPS",
                    "error": repr(error),
                })

            for method, (predictions, extra_metrics) in model_outputs.items():
                metrics = calculate_metrics(
                    predictions,
                    reference_frequency,
                )

                result = {
                    **base_metadata,
                    "actual_sample_rate": sample_rate,
                    "method": method,
                    "status": "success",
                    **metrics,
                    **extra_metrics,
                }

                recording_results.append(result)

                pooled_predictions[method].append(predictions)
                pooled_references[method].append(
                    np.full(
                        predictions.size,
                        reference_frequency,
                        dtype=np.float64,
                    )
                )

        except Exception as error:
            error_results.append({
                "audio_path": str(audio_path_value),
                "method": "ALL",
                "error": repr(error),
            })

        if (
            recording_number % 100 == 0
            or recording_number == total_recordings
        ):
            print(
                f"Processed {recording_number}/{total_recordings} recordings"
            )

    recording_results_path = (
        output_directory / "evaluation_per_recording.csv"
    )

    error_results_path = (
        output_directory / "evaluation_errors.csv"
    )

    summary_results_path = (
        output_directory / "evaluation_summary.csv"
    )

    recording_results_frame = pd.DataFrame(recording_results)
    error_results_frame = pd.DataFrame(error_results)

    recording_results_frame.to_csv(
        recording_results_path,
        index=False,
    )

    error_results_frame.to_csv(
        error_results_path,
        index=False,
    )

    summary_results = []

    for method in ("YIN", "HPS"):
        if not pooled_predictions[method]:
            summary_results.append({
                "method": method,
                "recordings_evaluated": 0,
                "status": "no successful evaluations",
            })
            continue

        all_predictions = np.concatenate(
            pooled_predictions[method]
        )

        all_references = np.concatenate(
            pooled_references[method]
        )

        pooled_metrics = {}

        for reference_frequency in np.unique(all_references):
            reference_mask = all_references == reference_frequency

            reference_metrics = calculate_metrics(
                all_predictions[reference_mask],
                reference_frequency,
            )

            for metric_name, metric_value in reference_metrics.items():
                pooled_metrics.setdefault(metric_name, []).append(
                    (reference_mask.sum(), metric_value)
                )

        method_recordings = recording_results_frame[
            (recording_results_frame["method"] == method)
            & (recording_results_frame["status"] == "success")
        ]

        summary = {
            "method": method,
            "status": "success",
            "recordings_evaluated": int(
                method_recordings["audio_path"].nunique()
            ),
            "total_frames": int(all_predictions.size),
        }

        valid_all = (
            np.isfinite(all_predictions)
            & (all_predictions > 0)
            & np.isfinite(all_references)
            & (all_references > 0)
        )

        frame_metrics = []

        for reference_frequency in np.unique(all_references):
            reference_mask = all_references == reference_frequency

            frame_metrics.append(
                calculate_metrics(
                    all_predictions[reference_mask],
                    reference_frequency,
                )
            )

        frame_metrics_frame = pd.DataFrame(frame_metrics)

        for metric_name in (
            "pitch_coverage_pct",
            "gross_pitch_error_pct",
            "octave_error_pct",
            "fine_pitch_rmse_cents",
            "median_abs_error_cents",
        ):
            summary[f"mean_{metric_name}"] = float(
                method_recordings[metric_name].mean()
            )

        summary.update(
            calculate_metrics(
                all_predictions,
                float(np.median(all_references)),
            )
        )

        summary["pooled_valid_pitch_frames"] = int(valid_all.sum())

        summary_results.append(summary)

    summary_results_frame = pd.DataFrame(summary_results)

    summary_results_frame.to_csv(
        summary_results_path,
        index=False,
    )

    print("\nEvaluation complete.")
    print(f"Per-recording results: {recording_results_path}")
    print(f"Summary results: {summary_results_path}")
    print(f"Errors: {error_results_path}")

    if not summary_results_frame.empty:
        columns_to_show = [
            column
            for column in (
                "method",
                "recordings_evaluated",
                "mean_pitch_coverage_pct",
                "mean_gross_pitch_error_pct",
                "mean_octave_error_pct",
                "mean_fine_pitch_rmse_cents",
            )
            if column in summary_results_frame.columns
        ]

        print("\nSummary:")
        print(
            summary_results_frame[columns_to_show].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
