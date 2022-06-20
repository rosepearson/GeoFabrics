# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:22:08 2021

@author: pearsonra
"""
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name="geofabrics",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    author="Rose pearson",
    author_email="rose.pearson@niwa.co.nz",
    description="A package for creating geofabrics for flood modelling "
    "(i.e. Hydrologically conditioned DEMs and roughness maps)",
    keywords="GeoFabrics, DEM, Roughness, Hydrologically conditioned",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rosepearson/GeoFabrics",
    project_urls={
        "Documentation": "https://rosepearson.github.io/GeoFabrics/",
        "Wiki": "https://github.com/rosepearson/GeoFabrics/wiki",
        "Bug Reports": "https://github.com/rosepearson/GeoFabrics/issues",
        "Source Code": "https://github.com/rosepearson/GeoFabrics",
    },
    classifiers=[
        # see https://pypi.org/classifiers/
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "dask",
        "distributed",
        "gdal",
        "geopandas>=0.10",
        "pdal",
        "pytest",
        "python-dotenv",
        "python-pdal",
        "rioxarray",
    ],
    extras_require={
        "dev": ["check-manifest"],
        "doc": ["Sphinx", "sphinx-autoapi", "sphinx-rtd-theme"],
    },
)
