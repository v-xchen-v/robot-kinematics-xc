# -*- coding: utf-8 -*-
"""
URDFPy Compatibility Module

This module provides compatibility patches for URDFPy + Trimesh on modern Python/NumPy versions.
Import this module before importing urdfpy to ensure compatibility.

Usage:
    from urdfpy_compat import apply_urdfpy_compatibility_patches
    apply_urdfpy_compatibility_patches()
    
    # Or use the convenience function:
    import urdfpy_compat  # Automatically applies patches
    
Compatibility Coverage:
- Python 3.8 -> 3.13
- NumPy 1.18 -> 2.1
- URDFPy latest release (still unmaintained but functional)
"""

def apply_urdfpy_compatibility_patches():
    """Apply compatibility patches for URDFPy + Trimesh on modern Python/Numpy
    
    This function patches deprecated attributes and modules that URDFPy/Trimesh
    still rely on but have been removed or moved in modern Python/NumPy versions.
    """
    # Fix collections module deprecations (Python 3.9+)
    import collections
    import collections.abc
    collections.Mapping = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping
    collections.Sequence = collections.abc.Sequence
    collections.Set = collections.abc.Set
    collections.Iterable = collections.abc.Iterable

    # Fix fractions.gcd removal (Python 3.9+)
    import math
    import fractions
    fractions.gcd = math.gcd

    # Fix NumPy type alias deprecations (NumPy 1.20+)
    import numpy as np
    np.int = int
    # np.bool = bool  # Uncomment if needed
    np.float = float
    np.float_ = float
    np.infty = np.inf

# Automatically apply patches when this module is imported
apply_urdfpy_compatibility_patches()
