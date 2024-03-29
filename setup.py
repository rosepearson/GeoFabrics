# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:22:08 2021

@author: pearsonra
"""
import setuptools

setuptools.setup(
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "geofabrics_from_file=geofabrics.__main__:cli_run_from_file",
            "geofabrics_from_dict=geofabrics.__main__:cli_run_from_dict",
        ],
    },
)
