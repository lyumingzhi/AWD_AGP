"""Backward-compatible imports for older AWD-AGP scripts."""

from inpainting.AWD_AGP.official_Attack import Attacker
from inpainting.AWD_AGP.official_utils import generate_rect_mask

__all__ = ["Attacker", "generate_rect_mask"]
