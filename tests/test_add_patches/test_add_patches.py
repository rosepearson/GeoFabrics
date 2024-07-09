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

        test_path = pathlib.Path(__file__).parent.resolve()

        # Setup logging
        logging.basicConfig(
            filename=test_path / "test.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        logging.info("In test_add_patches")

        # Load in the test instructions
        instruction_file_path = test_path / "instruction.json"
        with open(instruction_file_path, "r") as file_pointer:
            cls.instructions = json.load(file_pointer)
        # Remove any files from last test, then create a results directory
        cls.cache_dir = test_path / "data"
        cls.results_dir = cls.cache_dir / "results"
        cls.tearDownClass()
        cls.results_dir.mkdir()

        # Generate catchment data
        catchment_file = cls.results_dir / "catchment_boundary.geojson"
        x0 = 1919400
        y0 = 5610000
        x1 = 1920703
        y1 = 5611400
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(cls.instructions["output"]["crs"]["horizontal"])
        catchment.to_file(catchment_file)

        # Run geofabrics processing pipeline
        runner = processor.PatchDemGenerator(cls.instructions)
        runner.run()

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

    def test_result_dem(self):
        """A basic comparison between the generated and benchmark DEM"""

        # Load in benchmark DEM
        file_path = self.cache_dir / self.instructions["data_paths"]["benchmark_dem"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark_dem:
            benchmark_dem.load()
        # Load in result DEM
        file_path = self.results_dir / self.instructions["data_paths"]["result_dem"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test_dem:
            test_dem.load()
        # Compare DEMs elevations
        diff_array = (
            test_dem.z.data[~numpy.isnan(test_dem.z.data)]
            - benchmark_dem.z.data[~numpy.isnan(benchmark_dem.z.data)]
        )
        logging.info(f"DEM z diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test_dem.z.data,
            benchmark_dem.z.data,
            err_msg="The generated result_dem has different data from the "
            + "benchmark_dem",
        )

        # Compare DEMs data source classification
        diff_array = test_dem.data_source.data - benchmark_dem.data_source.data
        logging.info(f"DEM z array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test_dem.data_source.data,
            benchmark_dem.data_source.data,
            err_msg="The generated test data_source layer has different data "
            "from the benchmark",
        )

        # Compare DEMs lidar source classification
        diff_array = test_dem.lidar_source.data - benchmark_dem.lidar_source.data
        logging.info(f"DEM z array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test_dem.lidar_source.data,
            benchmark_dem.lidar_source.data,
            err_msg="The generated test lidar_source layer has different data "
            "from the benchmark",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test_dem
        del benchmark_dem


if __name__ == "__main__":
    unittest.main()
