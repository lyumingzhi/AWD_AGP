"""Namespace package for AWD-AGP and optional third-party inpainting backends.

Keeping this package extendable lets a clean AWD-AGP checkout import bundled
backends such as ``third_party/inpainting/WDNet`` and sibling projects such as
``inpainting.MAT`` when their parent directory is on ``PYTHONPATH``.
"""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_THIRD_PARTY_INPAINTING = Path(__file__).resolve().parents[1] / "third_party" / "inpainting"
if _THIRD_PARTY_INPAINTING.exists():
    third_party_path = str(_THIRD_PARTY_INPAINTING)
    if third_party_path not in __path__:
        __path__.append(third_party_path)
