# -*- coding: utf-8 -*-
"""
pytest configuration file for GeoFabrics tests.

IMPORTANT: Import pdal FIRST on Windows to avoid DLL loading conflicts
with GDAL-dependent libraries. This must run before any test modules are loaded.
See: https://github.com/PDAL/PDAL/issues/4408
"""

import pdal  # noqa: F401
