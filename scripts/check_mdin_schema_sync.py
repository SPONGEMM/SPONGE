#!/usr/bin/env python3
"""Compare SPONGE's authoritative schema files against Mokda copies."""

from __future__ import annotations

import argparse
import filecmp
import sys
from pathlib import Path

SCHEMA_FILES = (
    "mdin.schema.json",
    "cv.schema.json",
    "catalog.json",
    "tombi.toml",
)

MOKDA_SCHEMA_DIRS = (
    "schemas",
    "backend/src/spongui/schemas",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mokda_root",
        type=Path,
        help="Path to a Mokda checkout to compare against.",
    )
    parser.add_argument(
        "--sponge-schema-dir",
        type=Path,
        default=Path("schemas"),
        help="Path to SPONGE's authoritative schema directory.",
    )
    args = parser.parse_args()

    failed = False
    sponge_dir = args.sponge_schema_dir.resolve()
    mokda_root = args.mokda_root.resolve()

    for relative_dir in MOKDA_SCHEMA_DIRS:
        target_dir = mokda_root / relative_dir
        print(f"[CHECK] {target_dir}")
        for name in SCHEMA_FILES:
            source = sponge_dir / name
            target = target_dir / name
            if not source.exists():
                print(f"  [FAIL] missing SPONGE source: {source}")
                failed = True
                continue
            if not target.exists():
                print(f"  [FAIL] missing Mokda copy: {target}")
                print(f"         sync: cp {source} {target}")
                failed = True
                continue
            if filecmp.cmp(source, target, shallow=False):
                print(f"  [OK]   {name}")
            else:
                print(f"  [DIFF] {name}")
                print(f"         sync: cp {source} {target}")
                failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
