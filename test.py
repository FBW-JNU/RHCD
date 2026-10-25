#!/usr/bin/env python
"""Run the image embedding and extraction command-line interface."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "test/test_3loop.py"), run_name="__main__")
