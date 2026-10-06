"""Inspect the wide training height map on a plane with one Ant."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv.insert(1, "--wide")
    runpy.run_path(str(Path(__file__).with_name("ant_height_map.py")), run_name="__main__")
