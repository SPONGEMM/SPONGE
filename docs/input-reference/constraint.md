# Constraint Algorithm Parameters

## Constraint Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `constrain_mode` | string | none (constraints disabled) | Constraint algorithm |

`constrain_mode` options:

| Value | Description |
|-------|-------------|
| `"SETTLE"` | SETTLE algorithm, specialized for rigid water molecules (triangle constraints) |
| `"SHAKE"` | SHAKE algorithm, general bond length constraints |

## SETTLE Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `settle_disable` | bool | `false` | Disable SETTLE |

## SHAKE Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SHAKE_step_length` | float | - | SHAKE iteration convergence parameter |
