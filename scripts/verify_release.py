"""Verify bundled data hashes, portable NPZ schemas and public configuration files."""
import hashlib
import json
from pathlib import Path
import numpy as np
from step.io import load_snapshots
from step.workflows import read_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    checked = 0
    for path in sorted((ROOT / "data").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        for entry in manifest["files"]:
            data_path = path.parent / entry["file"]
            assert data_path.stat().st_size == entry["bytes"], data_path
            assert hashlib.sha256(data_path.read_bytes()).hexdigest() == entry["sha256"], data_path
            with np.load(data_path, allow_pickle=False) as data:
                schema = {key: dict(shape=list(data[key].shape), dtype=str(data[key].dtype)) for key in data.files}
                assert schema == entry["arrays"], data_path
            if data_path.parent.name == "limb_xenium":
                load_snapshots(data_path)
            checked += 1
    assert checked == 9, f"expected 9 bundled data files, got {checked}"
    for path in sorted((ROOT / "configs").glob("*.json")):
        read_config(path, "cpu", static=path.stem.startswith("static_"))
    print(f"Verified {checked} bundled NPZ files and all public configurations.")


if __name__ == "__main__":
    main()
