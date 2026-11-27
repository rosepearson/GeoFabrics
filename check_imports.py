#!/usr/bin/env python
"""Test script to verify all major package imports."""

import sys

if __name__ == "__main__":
    print("Python version:", sys.version)
    print()

    packages = [
        "pandas",
        "rioxarray",
        "geoapis",
        ("osmpythontools", "OSMPythonTools"),  # Needs capital letters
        "distributed",
        "scipy",
        "dask",
        "numpy",
        "geopandas",
        "rasterio",
        "pyproj",
        "shapely",
        "pyogrio",
        "xarray",
        "requests",
        "boto3",
        "matplotlib",
    ]

    print("Testing imports:")
    errors = []
    for pkg_info in packages:
        # Handle tuples (package_name, import_name) for packages with different names
        if isinstance(pkg_info, tuple):
            pkg_name, import_name = pkg_info
        else:
            pkg_name = import_name = pkg_info

        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "version not available")
            print(f"✓ {pkg_name}: {version}")
        except Exception as e:
            print(f"✗ {pkg_name}: {e}")
            errors.append((pkg_name, str(e)))

    if errors:
        print(f"\n{len(errors)} package(s) failed to import")
        sys.exit(1)
    else:
        print(f"\nAll {len(packages)} packages imported successfully!")
        sys.exit(0)
