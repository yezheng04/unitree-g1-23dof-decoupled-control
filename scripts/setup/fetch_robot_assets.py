"""Fetch the exact licensed Unitree asset revision used by the experiments."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--destination", type=Path, default=ROOT / "assets/g1")
p.add_argument("--verify-only", action="store_true")
args = p.parse_args()
manifest = json.loads((ROOT / "robot_assets.json").read_text())
for row in manifest["files"]:
    dest = args.destination / row["destination"]
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == row["sha256"]:
        continue
    if args.verify_only:
        raise SystemExit(f"Missing or mismatched asset: {dest}")
    url = f"https://raw.githubusercontent.com/unitreerobotics/unitree_ros/{manifest['commit']}/{row['source']}"
    with urlopen(Request(url, headers={"User-Agent": "g1-isaaclab-assets"}), timeout=90) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise SystemExit(f"SHA256 mismatch: {row['destination']}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".download")
    temp.write_bytes(data)
    temp.replace(dest)
    print("Verified", row["destination"], flush=True)
print(f"ASSETS_PASS: {len(manifest['files'])} files in {args.destination.resolve()}")
