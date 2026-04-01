#!/usr/bin/env python3
"""Sync version from pixi.toml to README and SKILL.md files."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get_version() -> str:
    pixi_toml = ROOT / "pixi.toml"
    for line in pixi_toml.read_text().splitlines():
        line = line.strip()
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("version not found in pixi.toml")


def check_or_sync(
    path: Path, pattern: str, replacement: str, write: bool
) -> bool:
    """Return True if file needed (or received) an update."""
    text = path.read_text()
    new_text = re.sub(pattern, replacement, text)
    if new_text == text:
        return False
    if write:
        path.write_text(new_text)
    return True


def version_targets(version: str):
    """Return files and replacement rules covered by version sync."""
    targets = []

    prips_init = ROOT / "plugins" / "prips" / "prips" / "__init__.py"
    targets.append(
        (prips_init, r'(__version__ = ")[^"]+(")', rf"\g<1>{version}\2")
    )

    for name in ("README.md", "README_en.md"):
        targets.append((ROOT / name, r"`v\d[^`]*`", f"`v{version}`"))

    for skill_md in sorted((ROOT / "skills").rglob("SKILL.md")):
        targets.append(
            (skill_md, r"(本技能适配 SPONGE 版本号：)\S+", rf"\g<1>{version}")
        )

    return targets


def main() -> int:
    check_only = "--check" in sys.argv
    list_targets_only = "--targets" in sys.argv
    list_changed_only = "--changed-paths" in sys.argv
    version = get_version()
    mismatched = []

    targets = version_targets(version)

    if list_targets_only:
        for path, _, _ in targets:
            print(path.relative_to(ROOT))
        return 0

    for path, pattern, replacement in targets:
        if check_or_sync(path, pattern, replacement, write=not check_only):
            mismatched.append(str(path.relative_to(ROOT)))

    if check_only:
        if mismatched:
            print("❌ 以下文件版本号与 pixi.toml 不一致：")
            for f in mismatched:
                print(f"  {f}")
            return 1
        return 0

    if list_changed_only:
        for path in mismatched:
            print(path)
        return 0

    if mismatched:
        print(f"Synced version {version} to: {', '.join(mismatched)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
