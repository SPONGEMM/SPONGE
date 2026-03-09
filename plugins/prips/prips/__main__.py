"""
PRIPS: Python Runtime Interface Plugin of SPONGE
"""

from . import __version__


def _main():
    import sys
    from pathlib import Path

    sys.path.append(Path(__file__).parent)

    with open(Path(__file__).parent / "pylib.txt") as f:
        pylib = f.read().strip()

    message = """
    Usage:
        1. Copy the plugin path printed above
        2. Paste it to the value of the command "plugin" of SPONGE
    """

    print(f"""
      PRIPS: Python Runtime Interface Plugin of SPONGE

    Version: {__version__}
    Python Dynamic Library: {pylib}
    Plugin Path: {Path(__file__).parent / "_prips.so"}
    {message}
    """)


if __name__ == "__main__":
    _main()
