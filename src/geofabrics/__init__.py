# -*- coding: utf-8 -*-
"""
Generate hydrologically conditioned DEMs from LiDAR and other information
using instructions supplied in a JSON instruction file.

Modules:
    * processor - Defines processors for DEMs, bathymetry, roughness, waterways,
      and stopbanks.
    * geometry - Loads and processes catchment and vector data.
    * dem - Reads, generates, and combines DEMs and roughness layers.
    * bathymetry_estimation - Estimates channel geometry and bathymetry.
    * runner - Merges instruction defaults and orchestrates processing stages.
    * __main__ - Provides the command-line entry point.
    * version - Defines the package version.
"""

from . import version

__version__ = version.__version__
