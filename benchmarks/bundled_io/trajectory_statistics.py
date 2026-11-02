"""Semantic observable series for trajectory A/B comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def trajectory_observable_series(
    dataset: str,
    values: Sequence[float],
    shape: tuple[int, ...],
    *,
    atom_weights: Sequence[float] | None = None,
    box_values: Sequence[float] | None = None,
    box_shape: tuple[int, ...] | None = None,
    include_atom_features: bool = True,
    maximum_pair_samples: int = 4096,
) -> dict[str, list[float]]:
    """Convert a particle or box trajectory into physical frame observables."""

    if maximum_pair_samples < 1:
        raise AssertionError("maximum pair samples must be positive")

    if dataset.endswith("/box/edges/value") and len(shape) == 3:
        return _box_observables(values, shape)
    if len(shape) != 3 or shape[-1] != 3:
        return {}
    if atom_weights is not None and (
        len(atom_weights) != shape[1]
        or any(weight <= 0.0 for weight in atom_weights)
    ):
        raise AssertionError(
            "atom weights must be positive and match atom count"
        )
    if dataset.endswith("/position/value"):
        return _position_observables(
            values,
            shape,
            atom_weights,
            box_values=box_values,
            box_shape=box_shape,
            include_atom_features=include_atom_features,
            maximum_pair_samples=maximum_pair_samples,
        )
    if dataset.endswith("/velocity/value"):
        return _velocity_observables(values, shape, atom_weights)
    if dataset.endswith("/force/value"):
        return _force_observables(
            values, shape, include_atom_features=include_atom_features
        )
    return {}


def _vector_frames(
    values: Sequence[float], shape: tuple[int, ...]
) -> list[list[tuple[float, float, float]]]:
    frame_count, item_count, width = shape
    if frame_count <= 0 or item_count <= 0 or width != 3:
        raise AssertionError(f"invalid vector trajectory shape: {shape}")
    if len(values) != frame_count * item_count * width:
        raise AssertionError(
            f"trajectory shape/value mismatch: shape={shape}, values={len(values)}"
        )
    frames = []
    offset = 0
    for _ in range(frame_count):
        frame = []
        for _ in range(item_count):
            frame.append(
                (values[offset], values[offset + 1], values[offset + 2])
            )
            offset += 3
        frames.append(frame)
    return frames


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _position_observables(
    values: Sequence[float],
    shape: tuple[int, ...],
    atom_weights: Sequence[float] | None,
    *,
    box_values: Sequence[float] | None,
    box_shape: tuple[int, ...] | None,
    include_atom_features: bool,
    maximum_pair_samples: int,
) -> dict[str, list[float]]:
    frames = np.asarray(values, dtype=float).reshape(shape)
    boxes = _position_boxes(box_values, box_shape, shape[0])
    box_array = (
        np.asarray(boxes, dtype=float).reshape((-1, 3, 3))
        if boxes is not None
        else None
    )
    inverses = np.linalg.inv(box_array) if box_array is not None else None
    if boxes is not None:
        fractional = np.einsum("fai,fij->faj", frames, inverses)
        frames = np.einsum(
            "fai,fij->faj", fractional - np.floor(fractional), box_array
        )
    origin = frames[0]
    weights = np.asarray(
        atom_weights if atom_weights is not None else [1.0] * shape[1],
        dtype=float,
    )
    center_name = (
        "center_of_mass" if atom_weights is not None else "center_of_geometry"
    )
    total_weight = float(np.sum(weights))
    centers = np.einsum("a,fai->fi", weights, frames) / total_weight
    result = {
        f"{center_name}_component_{axis}": centers[:, axis].tolist()
        for axis in range(3)
    }
    if include_atom_features:
        displacements = frames - origin
        if box_array is not None:
            fractional = np.einsum("fai,fij->faj", displacements, inverses)
            displacements = np.einsum(
                "fai,fij->faj", fractional - np.rint(fractional), box_array
            )
        squared_displacements = np.sum(displacements * displacements, axis=2)
        for atom_index in range(shape[1]):
            result[f"atom_{atom_index}_squared_displacement"] = (
                squared_displacements[:, atom_index].tolist()
            )
    sampled_pairs = _sampled_atom_pairs(shape[1], maximum_pair_samples)
    if sampled_pairs:
        left, right = np.asarray(sampled_pairs, dtype=int).T
        pair_vectors = frames[:, right, :] - frames[:, left, :]
        if box_array is not None:
            fractional = np.einsum("fpi,fij->fpj", pair_vectors, inverses)
            pair_vectors = np.einsum(
                "fpi,fij->fpj", fractional - np.rint(fractional), box_array
            )
        pair_distances = np.linalg.norm(pair_vectors, axis=2)
    else:
        pair_distances = np.zeros((shape[0], 1), dtype=float)
    for name, fraction in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
        result[f"pair_distance_{name}"] = np.quantile(
            pair_distances, fraction, axis=1
        ).tolist()
    return result


def _position_boxes(
    values: Sequence[float] | None,
    shape: tuple[int, ...] | None,
    frame_count: int,
) -> list[tuple[float, ...]] | None:
    if values is None and shape is None:
        return None
    if values is None or shape is None:
        raise AssertionError(
            "position box values and shape must be provided together"
        )
    if (
        len(shape) != 3
        or shape[1:] != (3, 3)
        or shape[0] not in {1, frame_count}
    ):
        raise AssertionError(f"invalid position box shape: {shape}")
    if len(values) != shape[0] * 9:
        raise AssertionError("position box shape/value mismatch")
    matrices = [
        tuple(values[index * 9 : (index + 1) * 9]) for index in range(shape[0])
    ]
    if len(matrices) == 1:
        return matrices * frame_count
    return matrices


def _wrap_position(
    vector: tuple[float, float, float],
    box: tuple[float, ...],
    inverse: tuple[float, ...] | None = None,
) -> tuple[float, float, float]:
    inverse = inverse or _inverse_matrix(box)
    fractional = _row_vector_matrix_product(vector, inverse)
    wrapped = tuple(value - math.floor(value) for value in fractional)
    return _row_vector_matrix_product(wrapped, box)


def _minimum_image(
    vector: tuple[float, float, float],
    box: tuple[float, ...] | None,
    inverse: tuple[float, ...] | None = None,
) -> tuple[float, float, float]:
    if box is None:
        return vector
    inverse = inverse or _inverse_matrix(box)
    fractional = _row_vector_matrix_product(vector, inverse)
    wrapped = tuple(value - round(value) for value in fractional)
    return _row_vector_matrix_product(wrapped, box)


def _sampled_atom_pairs(
    atom_count: int, maximum_pairs: int = 4096
) -> list[tuple[int, int]]:
    total = atom_count * (atom_count - 1) // 2
    if total <= 0:
        return []
    sample_count = min(total, maximum_pairs)
    if sample_count == 1:
        flat_indices = [0]
    else:
        flat_indices = [
            round(index * (total - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]
    pairs = []
    left = 0
    prefix = 0
    for flat_index in flat_indices:
        while flat_index >= prefix + atom_count - left - 1:
            prefix += atom_count - left - 1
            left += 1
        right = left + 1 + (flat_index - prefix)
        pairs.append((left, right))
    return pairs


def _row_vector_matrix_product(
    vector: Sequence[float], matrix: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(
        sum(vector[row] * matrix[row * 3 + column] for row in range(3))
        for column in range(3)
    )


def _inverse_matrix(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != 9:
        raise AssertionError("box matrix must contain nine values")
    a, b, c, d, e, f, g, h, i = values
    determinant = _determinant(values)
    if not math.isfinite(determinant) or abs(determinant) <= 1.0e-20:
        raise AssertionError("box matrix must be finite and non-singular")
    scale = 1.0 / determinant
    return (
        (e * i - f * h) * scale,
        (c * h - b * i) * scale,
        (b * f - c * e) * scale,
        (f * g - d * i) * scale,
        (a * i - c * g) * scale,
        (c * d - a * f) * scale,
        (d * h - e * g) * scale,
        (b * g - a * h) * scale,
        (a * e - b * d) * scale,
    )


def _velocity_observables(
    values: Sequence[float],
    shape: tuple[int, ...],
    atom_weights: Sequence[float] | None,
) -> dict[str, list[float]]:
    frames = np.asarray(values, dtype=float).reshape(shape)
    result: dict[str, list[float]] = {}
    for axis in range(3):
        result[f"component_{axis}_mean"] = np.mean(
            frames[:, :, axis], axis=1
        ).tolist()
        result[f"component_{axis}_variance"] = np.var(
            frames[:, :, axis], axis=1
        ).tolist()
    squared_speeds = np.sum(frames * frames, axis=2)
    if atom_weights is None:
        result["mean_squared_speed"] = np.mean(squared_speeds, axis=1).tolist()
    else:
        weights = np.asarray(atom_weights, dtype=float)
        result["mass_weighted_mean_squared_speed"] = (
            np.einsum("a,fa->f", weights, squared_speeds) / np.sum(weights)
        ).tolist()
    return _add_numpy_norm_distribution(result, np.sqrt(squared_speeds))


def _force_observables(
    values: Sequence[float],
    shape: tuple[int, ...],
    *,
    include_atom_features: bool,
) -> dict[str, list[float]]:
    frames = np.asarray(values, dtype=float).reshape(shape)
    result: dict[str, list[float]] = {
        f"net_component_{axis}": np.sum(frames[:, :, axis], axis=1).tolist()
        for axis in range(3)
    }
    result["component_rms"] = np.sqrt(
        np.mean(frames * frames, axis=(1, 2))
    ).tolist()
    norms = np.linalg.norm(frames, axis=2)
    if include_atom_features:
        for atom_index in range(shape[1]):
            result[f"atom_{atom_index}_norm"] = norms[:, atom_index].tolist()
    return _add_numpy_norm_distribution(result, norms)


def _add_numpy_norm_distribution(
    result: dict[str, list[float]], norms: np.ndarray
) -> dict[str, list[float]]:
    for name, fraction in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
        result[f"norm_{name}"] = np.quantile(norms, fraction, axis=1).tolist()
    return result


def _add_norm_distribution(
    result: dict[str, list[float]],
    frames: Sequence[Sequence[tuple[float, float, float]]],
) -> dict[str, list[float]]:
    frame_norms = [[_norm(vector) for vector in frame] for frame in frames]
    for name, fraction in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
        result[f"norm_{name}"] = [
            _quantile(norms, fraction) for norms in frame_norms
        ]
    return result


def _box_observables(
    values: Sequence[float], shape: tuple[int, ...]
) -> dict[str, list[float]]:
    if shape[1:] != (3, 3) or len(values) != shape[0] * 9:
        raise AssertionError(f"invalid box trajectory shape: {shape}")
    matrices = [
        values[index * 9 : (index + 1) * 9] for index in range(shape[0])
    ]
    result = {
        f"matrix_{row}_{column}": [
            matrix[row * 3 + column] for matrix in matrices
        ]
        for row in range(3)
        for column in range(3)
    }
    vectors = [
        [
            tuple(matrix[row * 3 + column] for column in range(3))
            for row in range(3)
        ]
        for matrix in matrices
    ]
    for axis in range(3):
        result[f"length_{axis}"] = [_norm(matrix[axis]) for matrix in vectors]
    result["volume"] = [abs(_determinant(matrix)) for matrix in matrices]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        result[f"angle_{left}_{right}"] = [
            _angle(matrix[left], matrix[right]) for matrix in vectors
        ]
    return result


def _determinant(matrix: Sequence[float]) -> float:
    return (
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
    )


def _angle(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator == 0.0:
        raise AssertionError("box vectors must have non-zero length")
    cosine = sum(a * b for a, b in zip(left, right)) / denominator
    return math.acos(max(-1.0, min(1.0, cosine)))
