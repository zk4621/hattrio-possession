"""Repository paths and access to the vendored challenge modules."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party" / "mtah2027"
EXTERNAL = ROOT / "external"
CACHE = ROOT / "cache"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
PUBLIC_GT = VENDOR / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"

_VENDOR_MODULES = ("games", "possession_metrics", "load_smart_tagging", "load_homography")


def data_dir() -> Path:
    """Competition data bundle: $MTAH_DATA_DIR, else ROOT/data."""
    env = os.environ.get("MTAH_DATA_DIR")
    return Path(env) if env else ROOT / "data"


def vendor_import(name: str):
    """Import one of the vendored challenge modules by name."""
    if name not in _VENDOR_MODULES:
        raise KeyError(f"not a vendored module: {name!r}")
    for sub in (VENDOR, VENDOR / "evaluation", VENDOR / "starter"):
        if str(sub) not in sys.path:
            sys.path.insert(0, str(sub))
    return importlib.import_module(name)
