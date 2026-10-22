"""Portable robot assets and conversion cache locations."""
import os
from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parent
_default_root = PACKAGE_ROOT.parents[2]
PROJECT_ROOT = Path(os.environ.get("G1_PROJECT_ROOT", _default_root)).expanduser().resolve()
ASSET_DIR = Path(os.environ.get("G1_ASSET_DIR", PROJECT_ROOT / "assets/g1")).expanduser().resolve()
URDF = Path(os.environ.get("G1_23DOF_URDF", ASSET_DIR / "g1_23dof_rev_1_0.urdf")).expanduser().resolve()
MJCF = Path(os.environ.get("G1_23DOF_MJCF", ASSET_DIR / "g1_23dof_rev_1_0.xml")).expanduser().resolve()
CACHE_DIR = Path(os.environ.get("G1_USD_CACHE", PROJECT_ROOT / "resources/usd_cache")).expanduser().resolve()
