#!/usr/bin/env python3
"""Download AWD-AGP checkpoints listed in configs/weights_manifest.json.

Fill `gdrive_url` or `gdrive_file_id` fields in the manifest after uploading
checkpoints to Google Drive. This script intentionally does not hard-code private
Drive IDs.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_path(repo_root, weights_root, entry):
    path = Path(entry["local_path"])
    if path.parts and path.parts[0] == "weights":
        return weights_root / Path(*path.parts[1:])
    return repo_root / path


def ensure_gdown():
    try:
        import gdown  # noqa: F401
        return
    except Exception:
        print("This script requires gdown. Install it with: pip install gdown", file=sys.stderr)
        raise SystemExit(2)


def run_gdown(entry, output_path):
    url = entry.get("gdrive_url", "")
    file_id = entry.get("gdrive_file_id", "")
    if not url and not file_id:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "gdown"]
    if file_id:
        cmd += ["--id", file_id]
    else:
        cmd += [url]
    cmd += ["-O", str(output_path)]
    subprocess.check_call(cmd)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download AWD-AGP Google Drive checkpoints.")
    parser.add_argument("--manifest", default="configs/weights_manifest.json")
    parser.add_argument("--weights-dir", default=os.environ.get("AWD_AGP_WEIGHTS_DIR", "weights"))
    parser.add_argument("--only", action="append", default=[], help="download only this manifest key; can be repeated")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()
    weights_root = Path(args.weights_dir).expanduser()
    if not weights_root.is_absolute():
        weights_root = repo_root / weights_root
    manifest = load_manifest(manifest_path)
    entries = manifest["entries"]
    keys = args.only or list(entries.keys())

    ensure_gdown()
    unresolved = []
    for key in keys:
        entry = entries[key]
        if entry.get("bundled"):
            print(f"[SKIP] {key}: bundled in repository")
            continue
        output_path = resolve_path(repo_root, weights_root, entry)
        if entry.get("type") == "directory":
            print(f"[INFO] {key}: directory entries usually require uploading/downloading an archive manually: {output_path}")
            if not entry.get("gdrive_url") and not entry.get("gdrive_file_id"):
                unresolved.append(key)
            continue
        if args.skip_existing and output_path.exists():
            print(f"[OK] {key}: already exists at {output_path}")
            continue
        if run_gdown(entry, output_path):
            print(f"[DOWNLOADED] {key}: {output_path}")
        else:
            unresolved.append(key)
            print(f"[NO LINK] {key}: fill gdrive_url or gdrive_file_id in {manifest_path}")

    if unresolved:
        print("\nEntries without usable Google Drive links:")
        for key in unresolved:
            print(f"  - {key}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
