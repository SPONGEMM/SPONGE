#!/usr/bin/env python3
"""Compatibility entry point for tools.viscosity.analysis.green_kubo."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.viscosity.analysis.green_kubo import *
if __name__ == "__main__": main()
