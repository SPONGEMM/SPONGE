"""Workflow presets render explicit engine capabilities; never interpreted by MD."""

import argparse
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def value(x):
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, str):
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, (int, float)):
        return str(x).lower()
    if isinstance(x, list):
        return "[" + ", ".join(map(value, x)) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}={value(v)}" for k, v in x.items()) + "}"
    raise TypeError(type(x))


def dumps(config):
    lines = []
    components = ("sampling", "velocity_bias", "momentum_exchange")

    def flatten(prefix, fields):
        for key, val in fields.items():
            full = f"{prefix}.{key}" if prefix else key
            if isinstance(val, dict):
                flatten(full, val)
            else:
                lines.append(f"{full}={value(val)}")

    flatten(
        "",
        {
            k: v
            for k, v in config.items()
            if k not in components and k != "external_force"
        },
    )
    for key in components:
        if key in config:
            lines.extend(["", f"[{key}]"])
            lines.extend(
                f"{name}={value(val)}" for name, val in config[key].items()
            )
    for drive in config.get("external_force", []):
        lines.extend(["", "[[external_force]]"])
        lines.extend(f"{name}={value(val)}" for name, val in drive.items())
    return "\n".join(lines) + "\n"


def expand(config):
    config = dict(config)
    if "coupling" in config:
        if "external_force" in config:
            raise ValueError("duplicate force configuration")
        config["external_force"] = config.pop("coupling")
    preset = config.pop("viscosity", None)
    if preset is None:
        return config
    method = preset.get("method", "none")
    allowed = {
        "method",
        "flow",
        "gradient",
        "mode",
        "bins",
        "sample_interval",
        "output",
        "acceleration",
        "exchange_interval",
    }
    method_allowed = {
        "green_kubo": {"method", "sample_interval", "output"},
        "rnemd": allowed - {"acceleration", "mode"},
        "cosine": allowed - {"exchange_interval"},
    }
    unknown = set(preset) - method_allowed.get(method, allowed)
    if unknown:
        raise ValueError(
            "unknown "
            + ("RNEMD" if method == "rnemd" else "workflow")
            + " option: "
            + str(sorted(unknown))
        )
    if method == "none":
        return config
    if method not in ("green_kubo", "cosine", "rnemd"):
        raise ValueError("workflow method not implemented: " + str(method))
    if "sampling" in config:
        raise ValueError("duplicate sampling configuration")
    sampling = dict(
        interval=preset.get("sample_interval", 1),
        output=preset.get("output", "viscosity.h5"),
    )
    config["sampling"] = sampling
    if method == "green_kubo":
        sampling["stress"] = True
        return config
    if method == "cosine" and "acceleration" not in preset:
        raise ValueError("cosine acceleration required")
    if method == "rnemd" and "exchange_interval" not in preset:
        raise ValueError("exchange_interval is required")
    flow, gradient = preset.get("flow", "x"), preset.get("gradient", "z")
    sampling.update(
        profile="cosine" if method == "cosine" else "slab",
        flow=flow,
        gradient=gradient,
        bins=preset.get("bins", 20),
    )
    if method == "rnemd":
        if "momentum_exchange" in config:
            raise ValueError("duplicate exchange configuration")
        config["momentum_exchange"] = dict(
            flow=flow,
            gradient=gradient,
            bins=sampling["bins"],
            interval=preset.get("exchange_interval", 100),
        )
    else:
        harmonic = preset.get("mode", 1)
        sampling["mode"] = harmonic
        if "velocity_bias" in config:
            raise ValueError("duplicate bias configuration")
        config["velocity_bias"] = dict(
            mode="cosine", flow=flow, gradient=gradient, harmonic=harmonic
        )
        acceleration = [0.0, 0.0, 0.0]
        acceleration["xyz".index(flow)] = preset.get("acceleration", 0.0)
        config.setdefault("external_force", []).append(
            dict(
                type="acceleration",
                acceleration=acceleration,
                profile=dict(type="cosine", axis=gradient, mode=harmonic),
            )
        )
    return config


def render(text):
    return dumps(expand(tomllib.loads(text)))


def write_engine_input(path, text):
    path.write_text(render(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("use a separate output file")
    args.output.write_text(render(args.input.read_text()))


if __name__ == "__main__":
    main()
