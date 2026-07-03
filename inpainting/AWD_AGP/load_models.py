"""Backward-compatible wrapper for the model-loading API."""

from inpainting.AWD_AGP.official_load_models import load_models

__all__ = ["load_models"]
