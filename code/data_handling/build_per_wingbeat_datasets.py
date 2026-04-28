import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from scipy.signal import find_peaks

from process_data import (
    PROCESSED_FILE_SUFFIX,
    PREDICTION_DATASETS_DIR,
    PROCESSED_PREDICTION_FLIGHT_DATA_DIR,
    PROCESSED_TRAIN_FLIGHT_DATA_DIR,
    TRAIN_DATASETS_DIR,
    _extract_features_and_targets,
)

WINGBEAT_DATASET_DIR = "data/wingbeat_datasets"

def _detect_wingbeat_intervals(
    wing_matrix: np.ndarray,
    min_peak_distance: int,
    min_peak_prominence: Optional[float],
    split_by: str = 'min',
) -> List[Tuple[int, int]]:
    """Detect [start, end) frame intervals for wingbeats based on split_by mode.

    If split_by == 'min': detect boundaries at troughs (backwardmost wing points).
    If split_by == 'max': detect boundaries at peaks (forwardmost wing points).
    """
    if wing_matrix.ndim != 2 or wing_matrix.shape[1] != 6:
        raise ValueError(f"Expected wing_matrix shape [T, 6], got {wing_matrix.shape}")

    left_phi = wing_matrix[:, 0]
    right_phi = wing_matrix[:, 3]
    stroke_signal = 0.5 * (left_phi + right_phi)

    if min_peak_prominence is None:
        signal_std = float(np.std(stroke_signal))
        min_peak_prominence = max(1e-6, 0.05 * signal_std)

    # Choose what to find based on split_by mode
    if split_by == 'max':
        # Find peaks (local maxima) - boundaries at forwardmost points
        signal_to_search = stroke_signal
    else:  # split_by == 'min' (default)
        # Find troughs (local minima) - boundaries at backwardmost points
        # Negate the signal so troughs become peaks
        signal_to_search = -stroke_signal

    peaks, _ = find_peaks(
        signal_to_search,
        distance=max(1, int(min_peak_distance)),
        prominence=min_peak_prominence,
    )

    if peaks.size < 2:
        return []

    intervals: List[Tuple[int, int]] = []
    for i in range(len(peaks) - 1):
        start = int(peaks[i])
        end = int(peaks[i + 1])
        if end > start:
            intervals.append((start, end))
    return intervals


def _sample_wingbeat_segment(
    segment: np.ndarray,
    n_samples: int,
    split_by: str = 'min',
) -> np.ndarray:
    """
    Convert a [Z, 6] wingbeat segment to [n_samples, 6] using time-proportional sampling.

    The opposite extreme (peak) from the boundary definition is dynamically located
    and explicitly included in the output at its natural time-proportional position.
    Two-phase interpolation is used: first phase from start to peak, second from peak to end.

    Args:
        segment: Wing angle data [Z, 6] representing one wingbeat interval
        n_samples: Target number of samples in output
        split_by: 'min' (boundaries at troughs) or 'max' (boundaries at peaks).
                  Determines which extreme to search for within the segment.

    Returns:
        Resampled segment [n_samples, 6]
    """
    if segment.ndim != 2 or segment.shape[1] != 6:
        raise ValueError(f"Expected segment shape [Z, 6], got {segment.shape}")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    z = segment.shape[0]
    if z <= 0:
        raise ValueError("Cannot sample an empty wingbeat segment")

    # Special case: single frame
    if z == 1:
        return np.repeat(segment, repeats=n_samples, axis=0)

    # Compute stroke signal to find the opposite extreme
    left_phi = segment[:, 0]
    right_phi = segment[:, 3]
    stroke_signal = 0.5 * (left_phi + right_phi)

    # Find the opposite extreme within this segment
    if split_by == 'max':
        # Boundaries are at maxima, so find minimum as the opposite extreme
        peak_idx = np.argmin(stroke_signal)
    else:  # split_by == 'min' (default)
        # Boundaries are at minima, so find maximum as the opposite extreme
        peak_idx = np.argmax(stroke_signal)

    # Calculate time-proportional allocation
    proportion = peak_idx / max(1, z - 1) if z > 1 else 0.5
    n_first_phase = max(1, int(round(proportion * n_samples)))
    # Add 1 to n_second_phase to account for the duplicate peak we'll remove during concatenation
    n_second_phase = n_samples - n_first_phase + 1

    # Helper function to interpolate a segment
    def interpolate_segment(seg: np.ndarray, n: int) -> np.ndarray:
        """Linearly interpolate segment to n samples."""
        seg_len = seg.shape[0]
        if seg_len == 1:
            return np.repeat(seg, repeats=n, axis=0)

        x_old = np.arange(seg_len, dtype=np.float64)
        x_new = np.linspace(0.0, float(seg_len - 1), num=n, endpoint=True)

        result = np.empty((n, 6), dtype=seg.dtype)
        for col in range(6):
            result[:, col] = np.interp(x_new, x_old, seg[:, col])
        return result

    # Phase 1: Interpolate from start to peak (inclusive)
    phase1_segment = segment[: peak_idx + 1]
    phase1_samples = interpolate_segment(phase1_segment, n_first_phase)

    # Phase 2: Interpolate from peak to end (inclusive)
    phase2_segment = segment[peak_idx:]
    phase2_samples = interpolate_segment(phase2_segment, n_second_phase)

    # Concatenate phases, removing duplicate peak sample from phase2
    result = np.vstack([phase1_samples, phase2_samples[1:]])

    # With the +1 adjustment to n_second_phase, we should have exactly n_samples
    assert result.shape[0] == n_samples, f"Expected {n_samples} samples, got {result.shape[0]}"
    return result


def _build_per_wingbeat_for_file(
    h5_path: str,
    forces_indication_vector: str,
    n_samples_per_wingbeat: int,
    use_radians: bool,
    min_peak_distance: int,
    min_peak_prominence: Optional[float],
    split_by: str = 'min',
) -> Tuple[np.ndarray, np.ndarray, dict]:
    body_matrix, wing_matrix = _extract_features_and_targets(
        h5_path,
        forces_indication_vector,
        use_radians=use_radians,
    )

    intervals = _detect_wingbeat_intervals(
        wing_matrix,
        min_peak_distance=min_peak_distance,
        min_peak_prominence=min_peak_prominence,
        split_by=split_by,
    )

    beat_body: List[np.ndarray] = []
    beat_wings: List[np.ndarray] = []

    for start, end in intervals:
        body_segment = body_matrix[start:end]
        wing_segment = wing_matrix[start:end]

        if body_segment.shape[0] <= 0 or wing_segment.shape[0] <= 0:
            continue

        body_mean = np.mean(body_segment, axis=0)
        wing_sampled = _sample_wingbeat_segment(wing_segment, n_samples=n_samples_per_wingbeat, split_by=split_by)

        beat_body.append(body_mean)
        beat_wings.append(wing_sampled)

    if beat_body:
        body_out = np.stack(beat_body, axis=0).astype(np.float32)
        wings_out = np.stack(beat_wings, axis=0).astype(np.float32)
    else:
        body_out = np.empty((0, body_matrix.shape[1]), dtype=np.float32)
        wings_out = np.empty((0, n_samples_per_wingbeat, 6), dtype=np.float32)

    if np.isnan(body_out).any() or np.isnan(wings_out).any():
        raise ValueError(f"NaNs found after wingbeat conversion for {h5_path}")

    metadata = {
        "file": os.path.basename(h5_path),
        "frames": int(body_matrix.shape[0]),
        "wingbeats": int(body_out.shape[0]),
        "body_dim": int(body_out.shape[1]) if body_out.ndim == 2 else 0,
        "output_shape": tuple(wings_out.shape),
    }
    return body_out, wings_out, metadata


def _build_train_dataset(
    forces_indication_vector: str,
    n_samples_per_wingbeat: int,
    use_radians: bool,
    min_peak_distance: int,
    min_peak_prominence: Optional[float],
    save_suffix: str,
    split_by: str = 'min',
) -> Tuple[str, str]:
    os.makedirs(TRAIN_DATASETS_DIR, exist_ok=True)

    all_body_inputs: List[torch.Tensor] = []
    all_wing_targets: List[torch.Tensor] = []

    files = sorted(f for f in os.listdir(PROCESSED_TRAIN_FLIGHT_DATA_DIR) if f.endswith(".h5"))
    for file_name in files:
        path = os.path.join(PROCESSED_TRAIN_FLIGHT_DATA_DIR, file_name)
        body_out, wings_out, metadata = _build_per_wingbeat_for_file(
            path,
            forces_indication_vector=forces_indication_vector,
            n_samples_per_wingbeat=n_samples_per_wingbeat,
            use_radians=use_radians,
            min_peak_distance=min_peak_distance,
            min_peak_prominence=min_peak_prominence,
            split_by=split_by,
        )
        all_body_inputs.append(torch.from_numpy(body_out))
        all_wing_targets.append(torch.from_numpy(wings_out))
        logger.info(
            f"train {metadata['file']}: frames={metadata['frames']}, "
            f"wingbeats={metadata['wingbeats']}, body_dim={metadata['body_dim']}, "
            f"wing_shape={metadata['output_shape']}"
        )

    input_path = os.path.join(TRAIN_DATASETS_DIR, f"train_input_forces_{save_suffix}.pt")
    target_path = os.path.join(TRAIN_DATASETS_DIR, f"train_output_kinematics_{save_suffix}.pt")
    torch.save(all_body_inputs, input_path)
    torch.save(all_wing_targets, target_path)

    logger.info(
        f"Saved per-wingbeat train dataset with {len(all_body_inputs)} sequences: "
        f"{input_path}, {target_path}"
    )
    return input_path, target_path


def _build_prediction_datasets(
    forces_indication_vector: str,
    n_samples_per_wingbeat: int,
    use_radians: bool,
    min_peak_distance: int,
    min_peak_prominence: Optional[float],
    save_suffix: str,
    split_by: str = 'min',
) -> int:
    os.makedirs(PREDICTION_DATASETS_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(PROCESSED_PREDICTION_FLIGHT_DATA_DIR) if f.endswith(".h5"))
    count = 0

    for file_name in files:
        path = os.path.join(PROCESSED_PREDICTION_FLIGHT_DATA_DIR, file_name)
        body_out, wings_out, metadata = _build_per_wingbeat_for_file(
            path,
            forces_indication_vector=forces_indication_vector,
            n_samples_per_wingbeat=n_samples_per_wingbeat,
            use_radians=use_radians,
            min_peak_distance=min_peak_distance,
            min_peak_prominence=min_peak_prominence,
            split_by=split_by,
        )

        base_name = file_name.replace(PROCESSED_FILE_SUFFIX, "")
        # Remove any remaining .h5 extension and _condensed_data to ensure clean base names
        if base_name.endswith(".h5"):
            base_name = base_name[:-3]
        base_name = base_name.replace("_condensed_data", "")

        input_path = os.path.join(PREDICTION_DATASETS_DIR, f"pred_inputs_{base_name}_{save_suffix}.pt")
        target_path = os.path.join(PREDICTION_DATASETS_DIR, f"pred_targets_{base_name}_{save_suffix}.pt")

        torch.save([torch.from_numpy(body_out)], input_path)
        torch.save([torch.from_numpy(wings_out)], target_path)

        logger.info(
            f"pred {metadata['file']}: frames={metadata['frames']}, "
            f"wingbeats={metadata['wingbeats']}, body_dim={metadata['body_dim']}, "
            f"wing_shape={metadata['output_shape']}"
        )
        count += 1

    logger.info(f"Saved {count} per-wingbeat prediction datasets in {PREDICTION_DATASETS_DIR}")
    return count


def _validate_forces_vector(forces: str) -> None:
    if len(forces) != 4 or any(ch not in {"0", "1"} for ch in forces):
        raise ValueError("forces must be a 4-bit string, e.g. 1111")


def run_per_wingbeat_builder(
    n_samples_per_wingbeat: int,
    forces_indication_vector: str = "1111",
    use_radians: bool = True,
    min_peak_distance: int = 20,
    min_peak_prominence: Optional[float] = None,
    save_suffix: Optional[str] = None,
    split_by: str = 'min',
) -> None:
    _validate_forces_vector(forces_indication_vector)

    if n_samples_per_wingbeat <= 0:
        raise ValueError("n_samples_per_wingbeat must be positive")

    if split_by not in ('min', 'max'):
        raise ValueError("split_by must be 'min' or 'max'")

    suffix = save_suffix or f"wingbeat_n{n_samples_per_wingbeat}_split{split_by}"

    train_input_path, train_target_path = _build_train_dataset(
        forces_indication_vector=forces_indication_vector,
        n_samples_per_wingbeat=n_samples_per_wingbeat,
        use_radians=use_radians,
        min_peak_distance=min_peak_distance,
        min_peak_prominence=min_peak_prominence,
        save_suffix=suffix,
        split_by=split_by,
    )

    pred_count = _build_prediction_datasets(
        forces_indication_vector=forces_indication_vector,
        n_samples_per_wingbeat=n_samples_per_wingbeat,
        use_radians=use_radians,
        min_peak_distance=min_peak_distance,
        min_peak_prominence=min_peak_prominence,
        save_suffix=suffix,
        split_by=split_by,
    )

    logger.info("Per-wingbeat dataset build complete.")
    logger.info(f"Train inputs: {train_input_path}")
    logger.info(f"Train targets: {train_target_path}")
    logger.info(f"Prediction datasets created: {pred_count}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-wingbeat datasets from processed H5 files. "
            "Inputs: one averaged kinematics vector per wingbeat. "
            "Targets: n sampled wing-angle vectors per wingbeat."
        )
    )
    parser.add_argument(
        "--n_samples_per_wingbeat",
        type=int,
        required=True,
        help="Number of target 6D wing-angle samples per wingbeat",
    )
    parser.add_argument(
        "--forces",
        type=str,
        default="1111",
        help="4-bit indication string for 12D body components (default: 1111)",
    )
    parser.add_argument(
        "--radians",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert target wing angles to radians (default: True)",
    )
    parser.add_argument(
        "--min_peak_distance",
        type=int,
        default=20,
        help="Minimum frame distance between consecutive backwardmost points",
    )
    parser.add_argument(
        "--min_peak_prominence",
        type=float,
        default=None,
        help="Optional peak prominence threshold for backwardmost-point detection",
    )
    parser.add_argument(
        "--save_suffix",
        type=str,
        default=None,
        help="Optional suffix used in saved .pt file names",
    )
    parser.add_argument(
        "--split_by",
        type=str,
        choices=['min', 'max'],
        default='min',
        help="Wingbeat boundary definition: 'min' for backwardmost points, 'max' for forwardmost points (default: min)",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_per_wingbeat_builder(
        n_samples_per_wingbeat=args.n_samples_per_wingbeat,
        forces_indication_vector=args.forces,
        use_radians=args.radians,
        min_peak_distance=args.min_peak_distance,
        min_peak_prominence=args.min_peak_prominence,
        save_suffix=args.save_suffix,
        split_by=args.split_by,
    )
