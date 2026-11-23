"""Build the resized JPEG -> memmap caches used by training and inference."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cxr.cache import build_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="448")
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for size in [int(s) for s in a.sizes.split(",")]:
        for split in a.splits.split(","):
            build_cache(split, size=size, workers=a.workers, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
