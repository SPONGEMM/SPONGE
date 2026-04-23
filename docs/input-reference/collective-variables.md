# Collective Variables Guide

SPONGE loads collective variable (CV) definitions from `cv_in_file` in
`mdin.spg.toml`.

```toml
cv_in_file = "cv.toml"
```

If `cv_in_file` is not set, the CV controller is not initialized.

## Supported file formats

By default, CV files should be written in TOML.

The file referenced by `cv_in_file` can be either:

- a TOML file
- a legacy block-style text file

Both formats are flattened into the same internal command namespace, but TOML
should be used for new documentation and new input files.

### Recommended TOML format

```toml
[print]
CV = ["phi", "psi"]

[phi]
CV_type = "dihedral"
atom = [4, 6, 8, 14]

[psi]
CV_type = "dihedral"
atom = [6, 8, 14, 16]
```

### Legacy block format

The legacy block format is still supported for compatibility with existing
cases:

```text
print
{
    CV = phi psi
}
phi
{
    CV_type = dihedral
    atom = 4 6 8 14
}
psi
{
    CV_type = dihedral
    atom = 6 8 14 16
}
```

## Basic workflow

1. Set `cv_in_file` in `mdin.spg.toml`.
2. Define one or more CV modules in the CV file.
3. Reference those CVs from `print`, `meta`, `steer`, `restrain`, or other
   modules that consume CV names.
4. Run SPONGE and inspect the corresponding output columns in `mdout`.

This page documents CV-side `restrain`, which is different from the atom-
coordinate restrain module documented in [restrain.md](restrain.md).

## CV definitions

Each CV is defined in a named module. The module name is how other sections
refer to the CV.

```toml
[distance]
CV_type = "distance"
atom = [4, 6]
```

In this example:

- `distance` is the CV module name
- `CV_type = distance` selects the implementation
- `atom = 4 6` provides the required atom indices

## Built-in CV types

| `CV_type` | Meaning | Required parameters |
|-----------|---------|---------------------|
| `position_x`, `position_y`, `position_z` | Cartesian coordinate of one atom | `atom` (1 atom) |
| `scaled_position_x`, `scaled_position_y`, `scaled_position_z` | Fractional coordinate of one atom | `atom` (1 atom) |
| `box_length_x`, `box_length_y`, `box_length_z` | Box length along one cell axis | none |
| `distance` | Distance between two atoms | `atom` (2 atoms) |
| `displacement_x`, `displacement_y`, `displacement_z` | Periodic displacement component between two atoms | `atom` (2 atoms) |
| `angle` | Angle formed by three atoms | `atom` (3 atoms) |
| `dihedral` | Dihedral angle formed by four atoms | `atom` (4 atoms) |
| `combination` | Algebraic combination of other CVs | `CV`, `function` |
| `tabulated` | Tabulated mapping of another CV with 4th-order B-spline interpolation | `CV`, `min`, `max`, `parameter` |
| `rmsd` | RMSD to a reference structure | `atom` or `atom_in_file`; `coordinate` or `coordinate_in_file` |

## RMSD CV

`rmsd` evaluates the RMSD between a selected atom set and a reference
coordinate set.

The atom list and reference coordinates can be generated with
`Xponge maskgen`.

```toml
[rmsd_ala]
CV_type = "rmsd"
atom_in_file = "rmsd_atoms.txt"
coordinate_in_file = "rmsd_ref.txt"
rotate = true
```

`rotate = true` enables optimal rotational alignment before RMSD evaluation.

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `atom` | int list | Atom list |
| `atom_in_file` | file | Atom list file |
| `coordinate` | float list | Reference coordinates |
| `coordinate_in_file` | file | Reference coordinate file |
| `rotate` | bool | Whether to align the reference before RMSD evaluation |

## Combination CV

`combination` combines one or more CVs into a new value through a JIT-compiled
expression.

Example:

```toml
[lx]
CV_type = "box_length_x"

[ly]
CV_type = "box_length_y"

[lz]
CV_type = "box_length_z"

[example_CV]
CV_type = "combination"
CV = ["lx", "ly", "lz"]
function = "lx * ly * lz"
```

The names used inside `function` must match the CV names listed in `CV`.

When the expression contains floating-point literals, prefer single-precision
suffixes such as `1.0f`. If the expression is compiled as a floating-point
expression but integer literals are used directly, the JIT compiler may fail.

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `CV` | CV list | CVs used in the combination |
| `function` | string | Expression used to combine the CVs |

Supported operators and functions:

| Meaning | Syntax | Meaning | Syntax |
|---------|--------|---------|--------|
| `a + b` | `a + b` | `a - b` | `a - b` |
| `a * b` | `a * b` | `a / b` | `a / b` |
| `a` to the power `b` | `powf(a)(b)` | natural logarithm of `a` | `logf(a)` |
| `exp(a)` | `expf(a)` | complementary error function | `erfcf(a)` |
| square root of `a` | `sqrtf(a)` | cosine of `a` | `cosf(a)` |
| sine of `a` | `sinf(a)` | tangent of `a` | `tanf(a)` |
| arccosine of `a` | `acosf(a)` | arcsine of `a` | `asinf(a)` |
| arctangent of `a` | `atanf(a)` | absolute value of `a` | `fabsf(a)` |
| larger of `a` and `b` | `fmaxf(a)(b)` | smaller of `a` and `b` | `fminf(a)(b)` |

## Tabulated CV

`tabulated` applies a tabulated mapping to another CV. Intermediate values are
evaluated through 4th-order B-spline interpolation.

Example:

```toml
[lx]
CV_type = "box_length_x"

[example_CV]
CV_type = "tabulated"
CV = "lx"
min = 0.0
max = 100.0
parameter = [1.0, 4.0, 0.2, 7.0, 9.1, -11.0]
min_padding = 1.1
max_padding = 7.7
```

This example corresponds to the following table:

| CV value | -60 | -40 | -20 | 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 |
|----------|-----|-----|-----|---|----|----|----|----|-----|-----|-----|-----|
| mapped value | 1.1 | 1.1 | 1.1 | 1.0 | 4.0 | 0.2 | 7.0 | 9.1 | -11.0 | 7.7 | 7.7 | 7.7 |

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `CV` | CV | CV to be interpolated |
| `min` | float | Minimum CV value |
| `max` | float | Maximum CV value |
| `parameter` | float list | Tabulated mapped values |
| `parameter_in_file` | file | File containing the mapped values |
| `min_padding` | float | Fill value used below `min`, only for interpolation |
| `max_padding` | float | Fill value used above `max`, only for interpolation |

## Virtual atoms in CV files

CV files can define CV-local virtual atoms and then reuse them in later CVs.
This is useful for center-of-mass-based CVs in metadynamics.

```toml
[ML]
vatom_type = "center_of_mass"
atom_in_file = "ligand.list"

[cx]
CV_type = "position_x"
atom = "ML"
```

In this example, `ML` is not a CV. It is a named virtual atom that can be used
where an atom index would normally appear.

## Printing CV values

Use the `print` module to append CV values to SPONGE step output.

```toml
[print]
CV = ["phi", "psi"]
```

Each printed CV appears under its module name. Avoid reusing names that already
exist as built-in output columns.

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `CV` | CV list | CVs to print |

## Restrain

`restrain` applies a harmonic bias potential:

$$
U_{\rm bias} = \sum_{CV}{\rm weight} * (CV - {\rm reference})^2
$$

The restrain weight can vary with simulation step.

If `start_step` and `max_step` are non-zero while `reduce_step` and
`stop_step` are zero, the weight ramps up linearly from 0 to `weight`.

```text
           Harmonic weight
                ↑
          weight|         +-------------
                |        /
                |       /
                |      /
                0-----+---+------------>simulation step
                   start max
```

If `start_step` and `max_step` are zero while `reduce_step` and `stop_step`
are non-zero, the weight stays at `weight` first and then ramps down linearly
to 0.

```text
           Harmonic weight
                ↑
          weight|----------------+
                |                 \
                |                  \
                |                   \
                0---------------+----+-->simulation step
                             reduce stop
```

If all four step parameters are non-zero, the weight ramps up, stays at the
maximum value, and then ramps down again.

```text
           Harmonic weight
                ↑
          weight|         +------+
                |        /        \
                |       /          \
                |      /            \
                0-----+---+-----+----+-->simulation step
                   start max reduce stop
```

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `CV` | CV list | CVs to restrain |
| `weight` | float list | Bias weight |
| `reference` | float list | Reference value |
| `period` | float list | Period of the CV |
| `start_step` | int list | Step at which `weight` starts increasing linearly |
| `max_step` | int list | Step at which `weight` reaches the maximum |
| `reduce_step` | int list | Step at which `weight` starts decreasing linearly |
| `stop_step` | int list | Step at which `weight` becomes 0 |

## Steer

`steer` applies a linear bias potential:

$$
U_{\rm bias} = \sum_{CV} {\rm weight} * CV
$$

Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `CV` | CV list | CVs to bias |
| `weight` | float list | Bias weight |

## Typical examples

### Dihedral CVs for SITS or analysis

```toml
[print]
CV = ["phi", "psi"]

[phi]
CV_type = "dihedral"
atom = [4, 6, 8, 14]

[psi]
CV_type = "dihedral"
atom = [6, 8, 14, 16]
```


## Relationship to enhanced sampling modules

- `cv_in_file` only tells SPONGE where to read CV definitions.
- `meta` consumes CV names and adds metadynamics-specific parameters such as
  `CV_sigma`, `CV_grid`, and `CV_period`.
- SITS also consumes CV names, but its own control parameters live in
  `mdin.spg.toml`.

For parameter-by-parameter reference, see
[Enhanced Sampling Parameters](enhanced-sampling.md) and
[SinkMeta](sinkmeta.md).
