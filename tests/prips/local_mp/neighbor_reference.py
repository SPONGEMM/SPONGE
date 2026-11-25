"""Exhaustive periodic edge enumeration used as a test reference."""

from itertools import product
from math import ceil, floor, sqrt


def _inverse3(matrix):
    a, b, c = matrix
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    cofactors = (
        (
            b[1] * c[2] - b[2] * c[1],
            a[2] * c[1] - a[1] * c[2],
            a[1] * b[2] - a[2] * b[1],
        ),
        (
            b[2] * c[0] - b[0] * c[2],
            a[0] * c[2] - a[2] * c[0],
            a[2] * b[0] - a[0] * b[2],
        ),
        (
            b[0] * c[1] - b[1] * c[0],
            a[1] * c[0] - a[0] * c[1],
            a[0] * b[1] - a[1] * b[0],
        ),
    )
    return tuple(
        tuple(value / determinant for value in row) for row in cofactors
    )


def enumerate_periodic_edges(
    positions, global_ids, num_owned, cell, pbc, radius
):
    """Return all in-radius directed edges from owned targets to source images.

    ``cell`` contains the three lattice vectors as rows.  A shift translates
    the source by ``(sx, sy, sz) @ cell``.  The inverse-cell bounds enumerate
    every lattice image whose displacement could have norm at most ``radius``;
    the final Cartesian distance check decides inclusion.
    Alias slots with the same global ID and raw coordinates collapse to one
    physical edge tuple per target ID, source ID, and lattice shift.
    """
    points = tuple(
        tuple(float(value) for value in point) for point in positions
    )
    lattice = tuple(tuple(float(value) for value in row) for row in cell)
    inverse = _inverse3(lattice)
    identifiers = tuple(global_ids)
    periodic = tuple(bool(value) for value in pbc)
    radius = float(radius)
    inverse_column_norms = tuple(
        sqrt(sum(inverse[row][axis] ** 2 for row in range(3)))
        for axis in range(3)
    )

    edges = []
    for target in range(num_owned):
        target_position = points[target]
        for source, source_position in enumerate(points):
            delta = tuple(
                source_position[i] - target_position[i] for i in range(3)
            )
            fractional_delta = tuple(
                sum(delta[row] * inverse[row][axis] for row in range(3))
                for axis in range(3)
            )
            shift_ranges = []
            for axis in range(3):
                if periodic[axis]:
                    bound = radius * inverse_column_norms[axis]
                    shift_ranges.append(
                        range(
                            ceil(-fractional_delta[axis] - bound),
                            floor(-fractional_delta[axis] + bound) + 1,
                        )
                    )
                else:
                    shift_ranges.append((0,))

            for shift in product(*shift_ranges):
                if identifiers[source] == identifiers[target] and shift == (
                    0,
                    0,
                    0,
                ):
                    continue
                displacement = tuple(
                    delta[cartesian]
                    + sum(
                        shift[axis] * lattice[axis][cartesian]
                        for axis in range(3)
                    )
                    for cartesian in range(3)
                )
                if (
                    sum(value * value for value in displacement)
                    <= radius * radius
                ):
                    edges.append(
                        (identifiers[target], identifiers[source], *shift)
                    )

    return sorted(set(edges))
