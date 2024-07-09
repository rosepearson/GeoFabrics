# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import json
import pathlib
import shutil
import rioxarray
import xarray
import numpy
import shapely
import geopandas
import pdal
import logging
import gc

from src.geofabrics import processor


class Test(unittest.TestCase):
    """Test the PatchDemGenerator processor class for a simple example with
    two patches specified in the instruction.json

    Tests run include:
        1. test_result_dem  Check the generated DEM matches the benchmark DEM within a
        tolerance
    """

    LAS_GROUND = 2

    @classmethod
    def setUpClass(cls):
        """Create a cache directory and CatchmentGeometry object for use in the tests
        and also download the files used in the tests."""

        # test_path = pathlib.Path(__file__).parent.resolve()

        # Setup logging
        logging.basicConfig(
            filename=cls.test_path / "test.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        logging.info("In {self.test_path.stem}")

        # Load in the test instructions
        instruction_file_path = cls.test_path / "instruction.json"
        with open(instruction_file_path, "r") as file_pointer:
            cls.instructions = json.load(file_pointer)
        # Remove any files from last test, then create a results directory
        cls.cache_dir = cls.test_path / "data"
        cls.results_dir = cls.cache_dir / "results"
        cls.tearDownClass()
        cls.results_dir.mkdir()

    @classmethod
    def tearDownClass(cls):
        """Remove created files in the cache directory as part of the testing process at
        the end of the test."""

        gc.collect()
        cls.clean_data_folder()

    @classmethod
    def clean_data_folder(cls):
        """Remove all generated or downloaded files from the data directory,
        but with only warnings if files can't be removed."""

        assert cls.cache_dir.exists(), (
            "The data directory that should include the comparison benchmark dem file "
            "doesn't exist"
        )

        # Cycle through all folders within the cache dir deleting their contents
        for path in cls.cache_dir.iterdir():
            if path.is_dir():
                for file in path.glob("*"):  # only files
                    if file.is_file():
                        try:
                            file.unlink()
                        except (Exception, PermissionError) as caught_exception:
                            logging.warning(
                                f"Caught error {caught_exception} during "
                                f"rmtree of {file}. Supressing error. You "
                                "will have to manually delete."
                            )
                    elif file.is_dir():
                        try:
                            shutil.rmtree(file)
                        except (Exception, PermissionError) as caught_exception:
                            logging.warning(
                                f"Caught error {caught_exception} during "
                                f"rmtree of {file}. Supressing error. You "
                                "will have to manually delete."
                            )
                try:
                    shutil.rmtree(path)
                except (Exception, PermissionError) as caught_exception:
                    logging.warning(
                        f"Caught error {caught_exception} during rmtree of "
                        f"{path}. Supressing error. You will have to manually "
                        "delete."
                    )


if __name__ == "__main__":
    unittest.main()
