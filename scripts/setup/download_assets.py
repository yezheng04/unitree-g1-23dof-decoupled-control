"""Download robot assets and checkpoint 22000; optionally include full videos."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", action="store_true", help="Also download all three full videos")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts/setup/fetch_robot_assets.py")], check=True)
    manifest = json.loads((ROOT / "scripts/setup/release_assets.json").read_text())
    for row in manifest["files"]:
        if row["path"].startswith("videos/") and not args.videos:
            continue
        dest = ROOT / "artifacts" / row["path"]
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == row["sha256"]:
            print("Verified", row["path"], flush=True)
            continue
        url = f"{manifest['release_url']}/{dest.name}"
        with urlopen(Request(url, headers={"User-Agent": "g1-project-assets"}), timeout=90) as response:
            data = response.read()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise SystemExit(f"Checksum mismatch: {row['path']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(dest.suffix + ".download")
        temp.write_bytes(data)
        temp.replace(dest)
        print("Downloaded", row["path"], flush=True)
    print("DOWNLOAD_PASS")


if __name__ == "__main__":
    main()
