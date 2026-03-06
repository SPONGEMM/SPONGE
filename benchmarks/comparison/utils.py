import json
from functools import lru_cache
from pathlib import Path
import re

import numpy as np


def _ensure_file_exists(file_path, label, suite_label):
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {suite_label} {label}: {path}")
    return path


def get_comparison_root_from_statics(statics_path):
    return Path(statics_path).resolve().parents[2]


def get_reference_root(statics_path, suite_name):
    return (
        get_comparison_root_from_statics(statics_path)
        / "reference"
        / str(suite_name)
    )


def get_reference_json_path(statics_path, suite_name):
    return get_reference_root(statics_path, suite_name) / "reference.json"


def get_reference_statics_case_dir(statics_path, suite_name, case_name):
    return get_reference_root(statics_path, suite_name) / "statics" / case_name


@lru_cache(maxsize=12)
def load_reference_entries(reference_json_path_str, suite_label):
    reference_json_path = _ensure_file_exists(
        reference_json_path_str, "reference file", suite_label
    )
    payload = json.loads(reference_json_path.read_text())
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(
            f"Invalid {suite_label} reference format in {reference_json_path}: "
            "'entries' must be a list."
        )

    index = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid {suite_label} reference entry #{idx}: expected object"
            )
        try:
            case_name = str(entry["case_name"])
            iteration = int(entry["iteration"])
        except KeyError as exc:
            raise ValueError(
                f"Invalid {suite_label} reference entry #{idx}: missing {exc}"
            ) from exc
        key = (case_name, iteration)
        if key in index:
            raise ValueError(
                f"Duplicate {suite_label} reference entry for case={case_name}, "
                f"iteration={iteration}"
            )
        index[key] = entry
    return payload, index


def load_reference_entry(
    reference_json_path, suite_label, case_name, iteration
):
    json_path = _ensure_file_exists(
        reference_json_path, "reference file", suite_label
    )
    _payload, index = load_reference_entries(str(json_path), suite_label)
    key = (str(case_name), int(iteration))
    if key not in index:
        raise KeyError(
            f"{suite_label} reference entry not found for "
            f"case={case_name}, iteration={iteration} in {json_path}"
        )
    return index[key]


def load_reference_npy(
    reference_root,
    entry,
    *,
    key,
    suite_label,
    dtype=np.float64,
    expected_ndim=None,
    expected_last_dim=None,
):
    rel_path = entry.get(key)
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError(
            f"{suite_label} reference entry missing '{key}': {entry}"
        )
    array_path = _ensure_file_exists(
        Path(reference_root) / rel_path,
        f"reference file for '{key}'",
        suite_label,
    )
    arr = np.asarray(np.load(array_path), dtype=dtype)
    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(
            f"Invalid {suite_label} reference array rank {arr.ndim} in "
            f"{array_path}; expected {expected_ndim}"
        )
    if expected_last_dim is not None and arr.shape[-1] != expected_last_dim:
        raise ValueError(
            f"Invalid {suite_label} reference array shape {arr.shape} in "
            f"{array_path}; expected last dim {expected_last_dim}"
        )
    return arr


def force_stats(reference, predicted):
    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if reference.shape != predicted.shape:
        raise ValueError(
            f"Force shape mismatch: ref={reference.shape}, pred={predicted.shape}"
        )
    diff = predicted - reference
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())
    rms = float(np.sqrt(np.mean(diff * diff)))

    ref_norm = np.linalg.norm(reference, axis=1)
    pred_norm = np.linalg.norm(predicted, axis=1)
    denom = ref_norm * pred_norm
    valid = denom > 1.0e-12
    if np.any(valid):
        cos = float(
            np.mean(
                np.sum(reference[valid] * predicted[valid], axis=1)
                / denom[valid]
            )
        )
    else:
        cos = 1.0
    return {
        "max_abs_diff": max_abs,
        "rms_diff": rms,
        "cosine_similarity": cos,
    }


def extract_sander_epot(sander_out_path):
    text = Path(sander_out_path).read_text()
    matches = re.findall(
        r"EPtot\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        text,
    )
    if not matches:
        raise ValueError(f"Failed to parse EPtot from {sander_out_path}")
    return float(matches[-1])


def extract_sander_forces_mdfrc(mdfrc_path):
    lines = Path(mdfrc_path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid mdfrc (too short): {mdfrc_path}")

    values = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        for j in range(0, len(line), 8):
            chunk = line[j : j + 8]
            if chunk.strip():
                values.append(float(chunk))
    if len(values) % 3 != 0:
        raise ValueError(
            f"Invalid mdfrc force length: {len(values)} in {mdfrc_path}"
        )
    return np.array(values, dtype=np.float64).reshape(-1, 3)
