# -*- coding: utf-8 -*-
"""
A package for generating hydrologically conditioned DEMs from LiDAR and bathymetry contours
give the instructions contained in a JSON instruction file.

Modules:
    * processor - A module for running the overall DEM generation pipeline
    * geometry - A module associated with manipulating vector data.
    * dem - A module associated with reading, generating, and combining DEMs
    * bathymetry_estimation - A module associated with estimating river characteristics
"""

from . import version

__version__ = version.__version__
