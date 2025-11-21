"""
DANSur Scripts Package

This package contains the main scripts for the DANSur project,
organized into submodules for better structure.
"""

import os, sys; 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from . import surrogate
from . import training  
from . import utils

__all__ = ['surrogate', 'training', 'utils']