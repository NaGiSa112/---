"""Split indexing and JPEG -> memmap cache building (no torch dependency)."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from .config import CACHE, DATA, NUM_CLASSES


def read_split(split: str, data_root: str = DATA) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_root, f"{split}.csv"))
    return df.reset_index(drop=True)


def parse_label_string(s) -> np.ndarray:
    y = np.zeros(NUM_CLASSES, dtype=np.float32)
    if isinstance(s, str):
        for part in s.split(";"):
            part = part.strip()
            if part:
                idx = int(part) - 1
                if 0 <= idx < NUM_CLASSES:
                    y[idx] = 1.0
    return y


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack([parse_label_string(s) for s in df["Predict_class"]]).astype(np.float32)


def image_path(split: str, subject: str, study: str, dicom: str, data_root: str = DATA) -> str:
    return os.path.join(data_root, split, subject, study, f"{dicom}.jpg")


def load_square(path: str, size: int) -> np.ndarray:
    """Decode a JPEG straight to a size x size grayscale uint8 square.

    PIL's draft() applies DCT scaling, so a 3000 px radiograph is decoded at a
    reduced size instead of full resolution. The requested target must be slightly
    BELOW ``size``: PIL picks the largest reduction that still keeps every side at
    or above the requested value, and for a 2544 px film ``2544/4 = 636 < 640``, so
    asking for exactly ``size`` silently falls back to 1/2 and decodes 4x the
    pixels we need. Requesting ``0.98 * size`` selects the 1/4 step, which cuts
    end-to-end latency by ~40 ms per image at 640 (measured: 119 -> 78 ms).

    The pixel difference against the old path is ~0.4 grey levels; macro AUC on the
    full official valid split is unchanged (0.82854 -> 0.82841).
    """
    target = max(int(size * 0.98), 1)
    with Image.open(path) as im:
        im.draft("L", (target, target))
        im = im.convert("L").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def load_square_views(path: str, cache_size: int, sizes) -> dict[int, np.ndarray]:
    """Decode once at the existing cache DCT scale, resize each output directly."""
    target = max(int(cache_size * 0.98), 1)
    with Image.open(path) as im:
        im.draft("L", (target, target))
        gray = im.convert("L")
        return {int(size): np.asarray(gray.resize((int(size), int(size)), Image.BILINEAR),
                                      dtype=np.uint8) for size in set(sizes)}


def build_cache(split: str, size: int = 448, workers: int = 16, data_root: str = DATA,
                cache_dir: str = CACHE, overwrite: bool = False):
    os.makedirs(cache_dir, exist_ok=True)
    out_npy = os.path.join(cache_dir, f"{split}_{size}.npy")
    out_meta = os.path.join(cache_dir, f"{split}_{size}.json")
    df = read_split(split, data_root)
    paths = [image_path(split, r.Subject_id, r.Study_id, r.Dicom_id, data_root) for r in df.itertuples()]
    if os.path.exists(out_npy) and os.path.exists(out_meta) and not overwrite:
        print(f"[cache] {split}: exists -> {out_npy}")
        return out_npy, out_meta
    n = len(paths)
    mm = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.uint8, shape=(n, size, size))

    def work(i):
        mm[i] = load_square(paths[i], size)
        return 1

    with ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(work, range(n)), 1):
            if k % 2000 == 0 or k == n:
                print(f"[cache] {split}: {k}/{n}", flush=True)
    mm.flush()
    del mm
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({"split": split, "size": size, "dicom_ids": df["Dicom_id"].tolist()}, f)
    print(f"[cache] wrote {out_npy} ({n} x {size} x {size})")
    return out_npy, out_meta


def open_cache(split: str, size: int, cache_dir: str = CACHE):
    npy = os.path.join(cache_dir, f"{split}_{size}.npy")
    with open(os.path.join(cache_dir, f"{split}_{size}.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return np.load(npy, mmap_mode="r"), meta
