"""Small setup check for the research repository."""

from __future__ import annotations

import sys

import numpy as np
import matplotlib
import networkx as nx
import yaml

try:
    import doces
except ImportError as exc:
    raise SystemExit(
        "Could not import DOCES.\n\n"
        "Expected repository layout:\n"
        "  DMAS/doces-slop\n"
        "  DMAS/dmas-ai-slop\n\n"
        "From dmas-ai-slop run:\n"
        "  pip install -e ../doces-slop\n"
    ) from exc


def main() -> None:
    print("=== DMAS AI-Slop environment check ===")
    print(f"Python:      {sys.version.split()[0]}")
    print(f"NumPy:       {np.__version__}")
    print(f"Matplotlib:  {matplotlib.__version__}")
    print(f"NetworkX:    {nx.__version__}")
    print(f"PyYAML:      {yaml.__version__}")
    print("DOCES:       import OK")
    print()
    print("✓ Research environment is ready.")


if __name__ == "__main__":
    main()
