#!/usr/bin/env python3
"""Independent, read-only OscGrid v6 CSV audit. No classifier training.

Reimplements the public 2026 validation/train.py block split (commit 4cbdf81)
and checks source-recording overlap, class concentration, and channel coverage.
No dataset bytes or private source files are distributed with this code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_SHA256 = "fa3c891163c27880296c240b77c1d23d2ce41be36858ca233f774be9e86f1854"
AUTHOR_COMMIT = "4cbdf81f9a05f646b5e4ee508e273a7b8e92b056"
FEATURES = ["IA", "IC", "UA BB", "UB BB", "UC BB"]
CL_CHANNELS = ["UA CL", "UB CL", "UC CL"]
SIZE, STRIDE, SEED = 32, 10, 443
RATIOS = {"train": 0.6, "val": 0.1, "test": 0.3}
PROBE_RECORDING = "a3164c73c305b027d2ba382e7840bbd2"
DOMINANT_RECORDING = "8f49afcdd88bfa4e880c3ac108b24f07"
UNSEEN_RECORDING = "a066a9c201a217aed8ab79a9c7a8e512"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_partition(csv_path: Path):
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    ml = [c for c in columns if c.startswith(("ML_1", "ML_2", "ML_3"))]
    assert all(any(c.startswith(p) for c in ml) for p in ("ML_1", "ML_2", "ML_3")), (
        "Expected ML label family missing"
    )
    required = ["sample", "file_name"] + FEATURES + CL_CHANNELS
    assert set(required).issubset(columns), "Required CSV columns missing"
    usecols = list(dict.fromkeys(required + ml))
    dtypes = {c: "float32" for c in FEATURES + CL_CHANNELS + ml}
    raw = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes, low_memory=False)
    total_rows = len(raw)
    filtered = raw.dropna(subset=FEATURES).reset_index(drop=True)
    del raw

    flags = [
        filtered[[c for c in ml if c.startswith(p)]].eq(1).any(axis=1)
        for p in ("ML_1", "ML_2", "ML_3")
    ]
    m1, m2, m3 = flags
    target = np.select(
        [~(m1 | m2 | m3), m1 & ~m2 & ~m3, m2 & ~m3, m3],
        [0, 1, 2, 3],
        default=-1,
    ).astype("int8")
    assert (target >= 0).all(), "Unsupported label combination"
    filtered["target"] = target
    filtered["recording_id"] = filtered["file_name"].str.extract(
        r"^([0-9a-f]{32})", expand=False
    )
    assert filtered["recording_id"].notna().all(), "Recording ID does not match CSV convention"

    # This reproduces the author's contiguous (file_name, target) block definition.
    filtered["block_id"] = (
        filtered["file_name"].ne(filtered["file_name"].shift())
        | filtered["target"].ne(filtered["target"].shift())
    ).cumsum()
    blocks = filtered.groupby("block_id", sort=True).agg(
        file_name=("file_name", "first"),
        recording_id=("recording_id", "first"),
        target=("target", "first"),
        length=("target", "size"),
    )
    eligible = blocks.loc[blocks.length >= SIZE].copy()
    shuffled = eligible.sample(frac=1, random_state=SEED)
    assignment = {}
    for _, group in shuffled.groupby("target"):
        limits = {key: group.length.sum() * ratio for key, ratio in RATIOS.items()}
        used = {key: 0 for key in RATIOS}
        for block_id, row in group.iterrows():
            for split in ("val", "test", "train"):
                if used[split] + row.length <= limits[split] or split == "train":
                    assignment[block_id] = split
                    used[split] += row.length
                    break
    eligible["split"] = eligible.index.map(assignment)
    assert eligible["split"].notna().all()
    return total_rows, filtered, blocks, eligible


def audit(csv_path: Path) -> dict:
    digest = sha256_file(csv_path)
    if digest != DATASET_SHA256:
        raise ValueError("Dataset SHA-256 does not match pinned Figshare v6 labeled.csv: " + digest)
    total_rows, df, blocks, eligible = load_and_partition(csv_path)
    train_sources = set(eligible.loc[eligible.split == "train", "recording_id"])
    test_sources = set(eligible.loc[eligible.split == "test", "recording_id"])
    seen_test = eligible.loc[eligible.split == "test"]
    seen_test_rows = int(
        seen_test.loc[seen_test.recording_id.isin(train_sources), "length"].sum()
    )
    n_test_rows = int(seen_test.length.sum())

    # Model windows cannot cross a constant-label block boundary.
    eligible["n_windows"] = ((eligible.length - SIZE) // STRIDE + 1).astype(int)
    test3 = eligible.loc[(eligible.split == "test") & (eligible.target == 3)]
    train_class3_sources = set(
        eligible.loc[(eligible.split == "train") & (eligible.target == 3), "recording_id"]
    )
    per_test_source3 = test3.groupby("recording_id").n_windows.sum()
    class3_test_windows = int(test3.n_windows.sum())
    known3 = per_test_source3.loc[per_test_source3.index.isin(train_class3_sources)]
    unknown3 = per_test_source3.loc[~per_test_source3.index.isin(train_class3_sources)]

    # Class-3 concentration after filtering for the five input features.
    valid3 = df.loc[df.target == 3].groupby("recording_id").size().sort_values(ascending=False)
    eligible3 = eligible.loc[eligible.target == 3].groupby("recording_id").length.sum()
    eligible3 = eligible3.sort_values(ascending=False)

    # This is a channel-coverage observation, not a diagnosis of the original event.
    record = df.loc[(df.recording_id == PROBE_RECORDING) & (df.target == 3)]
    chunks = record[FEATURES].to_numpy(dtype=np.float64)
    chunk_count = len(chunks) // SIZE
    chunk_std = chunks[:chunk_count * SIZE].reshape(chunk_count, SIZE, 5).std(axis=1)
    chunks_near_flat = int((chunk_std.max(axis=1) < 0.02).sum())
    channel_std = record[FEATURES + CL_CHANNELS].std(ddof=1)

    return {
        "dataset": {"doi": "10.6084/m9.figshare.28465427.v6",
                    "csv_sha256": digest, "csv_rows": total_rows,
                    "rows_with_five_inputs": int(len(df)),
                    "author_commit": AUTHOR_COMMIT},
        "split": {
            "author_seed": SEED,
            "frame_size": SIZE,
            "stride": STRIDE,
            "n_eligible_blocks": int(len(eligible)),
            "n_source_recordings": int(eligible.recording_id.nunique()),
            "train_rows": int(eligible.loc[eligible.split == "train", "length"].sum()),
            "validation_rows": int(eligible.loc[eligible.split == "val", "length"].sum()),
            "test_rows": n_test_rows,
            "source_recordings_shared_train_test": len(train_sources & test_sources),
            "test_rows_from_train_seen_recordings": seen_test_rows,
            "test_rows_from_train_seen_recordings_fraction": round(seen_test_rows/n_test_rows, 6),
        },
        "class3": {
            "valid_rows": int((df.target == 3).sum()),
            "eligible_rows": int(eligible3.sum()),
            "top_five_valid_rows": int(valid3.head(5).sum()),
            "top_five_valid_rows_fraction": round(valid3.head(5).sum()/valid3.sum(), 6),
            "top_five_eligible_rows_fraction": round(eligible3.head(5).sum()/eligible3.sum(), 6),
            "test_windows": class3_test_windows,
            "seen_same_class_recordings": int(len(known3)),
            "unseen_same_class_recordings": int(len(unknown3)),
            "seen_same_class_windows": int(known3.sum()),
            "unseen_same_class_windows": int(unknown3.sum()),
            "dominant_recording_test_windows": int(per_test_source3.get(DOMINANT_RECORDING, 0)),
            "unseen_a066_test_windows": int(per_test_source3.get(UNSEEN_RECORDING, 0)),
        },
        "channel_probe": {
            "recording_id": PROBE_RECORDING,
            "true_class3_rows": int(len(record)),
            "nonoverlapping_32_sample_chunks": chunk_count,
            "chunks_max_five_input_std_below_0_02": chunks_near_flat,
            "fraction_chunks_max_five_input_std_below_0_02":
                round(chunks_near_flat/chunk_count, 6),
            "whole_record_class3_std_by_channel": {
                key: round(float(value), 6) for key, value in channel_std.items()
            },
        },
        "limitations": [
            "This audit reproduces split structure and channel statistics, not trained-model accuracy.",
            "Shared source IDs do not quantify any F1 inflation or prove distinct physical events.",
            "The selected five feature columns are not the entire set of physical measurements.",
            "CL and BB abbreviations are preserved from dataset channel names; hardware topology is not verified.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="Extracted original Figshare v6 labeled.csv")
    ap.add_argument("--output", type=Path, help="Optional JSON report path")
    args = ap.parse_args()
    result = audit(args.csv)
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(data, encoding="utf-8")
    print(data)


if __name__ == "__main__":
    main()
