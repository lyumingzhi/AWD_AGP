"""Checkpoint path helpers for local/GDrive-downloaded AWD-AGP weights."""

import os
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[2]


def get_weights_root(opt=None):
    configured = getattr(opt, "weights_dir", None) if opt is not None else None
    configured = configured or os.environ.get("AWD_AGP_WEIGHTS_DIR")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else repo_root() / path
    return repo_root() / "weights"


def materialize_path(path, opt=None):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "weights":
        return get_weights_root(opt) / Path(*parts[1:])
    return repo_root() / path


def resolve_weight_path(default_path, opt=None, opt_attr=None, env_var=None, fallback_paths=(), required_name="checkpoint"):
    candidates = []
    if opt_attr and opt is not None:
        value = getattr(opt, opt_attr, None)
        if value:
            candidates.append(Path(value).expanduser())
    if env_var and os.environ.get(env_var):
        candidates.append(Path(os.environ[env_var]).expanduser())
    if default_path:
        candidates.append(materialize_path(default_path, opt))
    for fallback in fallback_paths:
        candidates.append(Path(fallback).expanduser())

    normalized = []
    for candidate in candidates:
        if not candidate.is_absolute() and candidate.parts and candidate.parts[0] != "weights":
            candidate = repo_root() / candidate
        normalized.append(candidate)
        if candidate.exists():
            return str(candidate)

    tried = "\n  - ".join(str(p) for p in normalized)
    raise FileNotFoundError(f"{required_name} not found. Tried:\n  - {tried}")
