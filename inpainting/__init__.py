"""Namespace package for AWD-AGP and optional third-party inpainting backends.

Keeping this package extendable lets a clean AWD-AGP checkout import sibling
projects such as ``inpainting.WDNet`` when their parent directory is on
``PYTHONPATH`` or supplied by a demo script.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
