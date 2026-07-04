#!/usr/bin/env python3
"""Check local AWD-AGP checkpoints against configs/weights_manifest.json."""

import argparse
import json
import os
from pathlib import Path


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_path(repo_root, weights_root, entry):
    path = Path(entry["local_path"])
    if path.parts and path.parts[0] == "weights":
        return weights_root / Path(*path.parts[1:])
    return repo_root / path


def main():
    parser = argparse.ArgumentParser(description="Check AWD-AGP local checkpoint files.")
    parser.add_argument("--manifest", default="configs/weights_manifest.json")
    parser.add_argument("--weights-dir", default=os.environ.get("AWD_AGP_WEIGHTS_DIR", "weights"))
    parser.add_argument("--require", action="append", default=[], help="entry key to require; defaults to all non-bundled entries")
    parser.add_argument("--include-bundled", action="store_true", help="also check bundled entries such as WDNet")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()
    weights_root = Path(args.weights_dir).expanduser()
    if not weights_root.is_absolute():
        weights_root = repo_root / weights_root

    manifest = load_manifest(manifest_path)
    entries = manifest["entries"]
    selected = args.require or [k for k, v in entries.items() if args.include_bundled or not v.get("bundled")]

    missing = []
    present = []
    for key in selected:
        if key not in entries:
            raise KeyError(f"Unknown manifest entry: {key}")
        entry = entries[key]
        path = resolve_path(repo_root, weights_root, entry)
        ok = path.is_dir() if entry.get("type") == "directory" else path.is_file()
        target = "directory" if entry.get("type") == "directory" else "file"
        if ok:
            present.append((key, path))
        else:
            missing.append((key, path, target, entry))

    print(f"Manifest: {manifest_path}")
    print(f"Weights root: {weights_root}")
    for key, path in present:
        print(f"[OK]      {key}: {path}")
    for key, path, target, entry in missing:
        hint = []
        if entry.get("cli_arg"):
            hint.append(entry["cli_arg"])
        if entry.get("env"):
            hint.append(entry["env"])
        suffix = f" ({', '.join(hint)})" if hint else ""
        print(f"[MISSING] {key}: expected {target} at {path}{suffix}")

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
