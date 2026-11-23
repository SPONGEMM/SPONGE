"""Compile the production CPU remapper fixture using an existing CPU build.

Requires an ELF toolchain supporting function/data sections and --gc-sections.
Unused engine entry points are discarded; the exercised packing function is
included directly from production nb14.cpp, not reimplemented in this test.
"""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(
        item for item in entries if item["file"].endswith("/nb14/nb14.cpp")
    )
    flags = shlex.split(entry["command"])
    if "-DUSE_CPU" not in flags:
        parser.error("this fixture requires a CPU compile_commands.json")
    command = []
    skip = False
    for flag in flags:
        if skip:
            skip = False
            continue
        if flag == "-o":
            skip = True
            continue
        if flag in ("-c", entry["file"], "-O3"):
            continue
        command.append(flag)
    with tempfile.TemporaryDirectory(prefix="nb14-remap-") as temporary:
        executable = Path(temporary) / "check"
        command += [
            "-O0",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            str(Path(__file__).with_suffix(".cpp").resolve()),
            "-o",
            str(executable),
        ]
        subprocess.run(command, cwd=build, check=True)
        subprocess.run([str(executable)], check=True)
    print(
        "production nb14 local pair packing passed (CPU; not MPI communication)"
    )


if __name__ == "__main__":
    main()
