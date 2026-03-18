"""
PRIPS: Python Runtime Interface Plugin of SPONGE
"""

from . import __version__


def _main():
    import sys
    from pathlib import Path

    sys.path.append(Path(__file__).parent)
    plugin_path = Path(__file__).parent / "_prips.so"
    if not plugin_path.exists():
        for entry in map(Path, sys.path):
            candidate = entry / "prips" / "_prips.so"
            if candidate.exists():
                plugin_path = candidate.resolve()
                break
    pylib_path = Path(__file__).parent / "pylib.txt"
    if pylib_path.exists():
        pylib = pylib_path.read_text(encoding="utf-8").strip()
    else:
        pylib = "unknown (editable install or pylib.txt missing)"

    message = """
    Usage:
        1. Copy the plugin path printed above
        2. When writing TOML on Windows, prefer the POSIX path
        3. Paste it to the value of the command "plugin" of SPONGE
    """

    print(f"""
      PRIPS: Python Runtime Interface Plugin of SPONGE

    Version: {__version__}
    Python Dynamic Library: {pylib}
    Plugin Path: {plugin_path}
    Plugin Path (POSIX): {plugin_path.as_posix()}
    {message}
    """)


if __name__ == "__main__":
    _main()
