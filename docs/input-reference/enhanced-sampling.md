# Enhanced Sampling Parameters

## Collective Variables (CV)

| Parameter | Type | Description |
|-----------|------|-------------|
| `cv_in_file` | string | Collective variable definition file path |

CVs can also be defined inline via the `[CV]` section:

| Parameter | Type | Description |
|-----------|------|-------------|
| `CV_type` | string | CV type |
| `CV_period` | int | CV update frequency |
| `CV_minimal` / `CV_maximum` | float | CV value range |

## Metadynamics

Configured via `[META]` or `[meta]` section:

| Parameter | Type | Description |
|-----------|------|-------------|
| `sink` | string | Sink metadynamics mode |

Detailed SinkMeta parameter list: [sinkmeta.md](sinkmeta.md)

## SITS

SITS (Self-guided Integrated Tempering Sampling) parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `SITS_mode` | string | Run mode |
| `SITS_atom_numbers` | int | Number of atoms participating in SITS |
| `SITS_k_numbers` | int | Number of k-space points |
| `SITS_T_low` / `SITS_T_high` | float | Temperature range (K) |
| `SITS_record_interval` | int | Record interval |
| `SITS_update_interval` | int | Update interval |
| `SITS_nk_fix` | int | Fixed k-space point count |
| `SITS_nk_in_file` | string | k-space input file |
| `SITS_pe_a` / `SITS_pe_b` | float | Potential energy parameters |
| `SITS_fb_interval` | int | Feedback interval |

`SITS_mode` options:

| Value | Description |
|-------|-------------|
| `"observation"` | Observation phase |
| `"iteration"` | Iteration phase |
| `"production"` | Production phase |
| `"empirical"` | Empirical mode |
| `"amd"` | AMD mode |
| `"gamd"` | GaMD mode |
