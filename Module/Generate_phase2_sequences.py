import os
import numpy as np
import random
from typing import Dict, List, Optional, Tuple


def is_zero_frame(frame: np.ndarray, atol: float = 1e-8) -> bool:
    """
    Return True if the entire frame vector is numerically close to zeros.

    Works regardless of feature dimensionality (e.g. 128 with masks at end).
    """
    return np.all(np.isclose(frame, 0.0, atol=atol))


def trim_leading_zeros(seq: np.ndarray, atol: float = 1e-8) -> Optional[np.ndarray]:
    """
    Remove leading frames that are entirely zeros. If sequence contains only zeros,
    return None.
    """
    if seq.size == 0:
        return None

    # Find first non-zero frame index
    first_idx = None
    for i in range(seq.shape[0]):
        if not is_zero_frame(seq[i], atol=atol):
            first_idx = i
            break

    if first_idx is None:
        # all-zero clip
        return None

    return seq[first_idx:]


def load_and_clean_sequence(path: str, atol: float = 1e-8) -> Optional[np.ndarray]:
    """
    Load a keypoint sequence (.npy) and trim leading all-zero frames.
    Return None if the sequence is empty or fully zero after trimming.
    """
    try:
        seq = np.load(path)
    except Exception:
        return None

    if not isinstance(seq, np.ndarray) or seq.ndim != 2:
        return None

    seq = trim_leading_zeros(seq, atol=atol)
    return seq


def collect_isolated_files(root: str, split: str) -> Dict[str, List[str]]:
    """
    Collect all .npy keypoint files under Data_Keypoints-like structure:
    {root}/{split}/{label}/*.npy
    Returns dict: label -> list of file paths
    """
    split_dir = os.path.join(root, split)
    label_to_files: Dict[str, List[str]] = {}
    if not os.path.isdir(split_dir):
        return label_to_files

    for label in os.listdir(split_dir):
        label_dir = os.path.join(split_dir, label)
        if not os.path.isdir(label_dir):
            continue
        files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npy')]
        if files:
            label_to_files[label] = files

    return label_to_files


def build_label_map(labels: List[str]) -> Dict[str, int]:
    """Map label name -> integer id (sorted for stability)."""
    return {lbl: idx for idx, lbl in enumerate(sorted(labels))}


def generate_sequence_from_labels(
    label_to_files: Dict[str, List[str]],
    label_map: Dict[str, int],
    k: int,
    min_frames_per_sign: int = 3,
    blank_frames_between: int = 0,
    atol: float = 1e-8,
) -> Optional[Tuple[np.ndarray, List[int], List[str], List[int]]]:
    """
    Generate one concatenated sequence by sampling k labels, one file per label,
    cleaning leading zeros, and concatenating. Optionally insert blank zero frames
    as separators.

    Returns:
      - concat_seq: (T, D)
      - label_ids: [k]
      - label_names: [k]
      - part_lengths: list of lengths of each sign segment (after cleaning)
    """
    if not label_to_files:
        return None

    available_labels = list(label_to_files.keys())
    if len(available_labels) < k:
        return None

    chosen_labels = random.sample(available_labels, k=k)

    cleaned_parts: List[np.ndarray] = []
    part_lengths: List[int] = []
    for lbl in chosen_labels:
        candidates = label_to_files[lbl]
        if not candidates:
            return None
        # pick a random file for this label
        p = random.choice(candidates)
        seq = load_and_clean_sequence(p, atol=atol)
        if seq is None:
            return None
        if seq.shape[0] < min_frames_per_sign:
            return None
        cleaned_parts.append(seq)
        part_lengths.append(int(seq.shape[0]))

    # All feature dims must match
    feat_dim = cleaned_parts[0].shape[1]
    if any(part.shape[1] != feat_dim for part in cleaned_parts):
        return None

    # Optional blank separators
    parts_with_blank: List[np.ndarray] = []
    for i, part in enumerate(cleaned_parts):
        parts_with_blank.append(part)
        if blank_frames_between > 0 and i < len(cleaned_parts) - 1:
            parts_with_blank.append(np.zeros((blank_frames_between, feat_dim), dtype=part.dtype))

    concat_seq = np.concatenate(parts_with_blank, axis=0)
    label_ids = [label_map[lbl] for lbl in chosen_labels]

    return concat_seq, label_ids, chosen_labels, part_lengths


def generate_phase2_dataset(
    input_root: str,
    output_root: str,
    split: str = "train",
    num_sequences: int = 1000,
    min_signs: int = 2,
    max_signs: int = 3,
    min_frames_per_sign: int = 3,
    blank_frames_between: int = 0,
    seed: Optional[int] = None,
) -> str:
    """
    Create phase2 concatenated keypoint sequences from isolated sign keypoints.

    - Trims leading all-zero frames per component sequence.
    - Skips clips that remain too short after trimming.
    - Optionally inserts blank separators of all-zero frames between signs.

    Saves .npz files to `{output_root}/{split}` with arrays:
      - keypoints: (T, D)
      - labels: (K,) int ids
      - label_names: (K,) list of strings
      - part_lengths: (K,) lengths after cleaning
    Also writes a small `manifest.txt` summarizing outputs.
    Returns the path to the manifest file.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(output_root, exist_ok=True)
    out_split_dir = os.path.join(output_root, split)
    os.makedirs(out_split_dir, exist_ok=True)

    label_to_files = collect_isolated_files(input_root, split)
    if not label_to_files:
        raise RuntimeError(f"No keypoint files found under {input_root}/{split}")

    label_map = build_label_map(list(label_to_files.keys()))

    written = 0
    manifest_lines: List[str] = []
    attempt = 0
    max_attempts = num_sequences * 20  # allow retries if a sample fails due to cleaning
    while written < num_sequences and attempt < max_attempts:
        attempt += 1
        k = random.randint(min_signs, max_signs)
        sample = generate_sequence_from_labels(
            label_to_files,
            label_map,
            k=k,
            min_frames_per_sign=min_frames_per_sign,
            blank_frames_between=blank_frames_between,
        )
        if sample is None:
            continue

        concat_seq, label_ids, label_names, part_lengths = sample
        save_path = os.path.join(out_split_dir, f"seq_{written:06d}.npz")
        np.savez_compressed(
            save_path,
            keypoints=concat_seq.astype(np.float32),
            labels=np.array(label_ids, dtype=np.int32),
            label_names=np.array(label_names, dtype=object),
            part_lengths=np.array(part_lengths, dtype=np.int32),
        )
        manifest_lines.append(
            f"seq_{written:06d}.npz\tK={len(label_ids)}\tlabels={label_names}\tT={concat_seq.shape[0]}\tD={concat_seq.shape[1]}"
        )
        written += 1

    manifest_path = os.path.join(out_split_dir, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines))

    return manifest_path


if __name__ == "__main__":
    # Example usage with defaults tailored to this repo structure
    import argparse

    parser = argparse.ArgumentParser(description="Generate Phase2 concatenated keypoint sequences (CTC-ready)")
    parser.add_argument("--input_root", type=str, default="Data_Keypoints", help="Root of isolated keypoints")
    parser.add_argument("--output_root", type=str, default="Data_Keypoints_Seq", help="Output directory for sequences")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Data split to use")
    parser.add_argument("--num_sequences", type=int, default=100, help="How many sequences to generate")
    parser.add_argument("--min_signs", type=int, default=2, help="Min signs per sequence")
    parser.add_argument("--max_signs", type=int, default=3, help="Max signs per sequence")
    parser.add_argument("--min_frames_per_sign", type=int, default=3, help="Min frames per sign after trimming")
    parser.add_argument("--blank_frames_between", type=int, default=0, help="Blank zero frames between signs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    manifest = generate_phase2_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        split=args.split,
        num_sequences=args.num_sequences,
        min_signs=args.min_signs,
        max_signs=args.max_signs,
        min_frames_per_sign=args.min_frames_per_sign,
        blank_frames_between=args.blank_frames_between,
        seed=args.seed,
    )
    print(f"Wrote manifest: {manifest}")

