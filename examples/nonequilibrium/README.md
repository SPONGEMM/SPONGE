当前示例已经改为显式引擎组件，参见 [动力学组件](../../docs/dynamics-components.md)。以下旧方法说明用于理解算例物理意义。

# Two-atom torque smoke case

Copy this directory into a separate run directory and execute:

```sh
/absolute/path/to/SPONGE -mdin mdin.spg.toml
```

This is a nonperiodic pair of neutral atoms with no internal interaction.
A total 1 pN nm torque is applied about their center of mass. Its purpose is
to demonstrate the minimal group-load input. It is not a rigid molecule or a
viscosity model. See `docs/nonequilibrium-coupling.md` for units and limitations.
