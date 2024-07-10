# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import pathlib
import rioxarray
import shapely
import geopandas
import dotenv
import os
import sys
import pytest
import logging
import numpy

from src.geofabrics import runner
from tests import base_test


@pytest.mark.skipif(sys.platform != "linux", reason="Skip test if not linux")
class Test(base_test.Test):
    """A class to test the basic river bathymetry estimation functionality
    contained in processor.RiverBathymetryGenerator.

    Tests run include:
        1. test_result_geofabric_linux - Test the geofabric layers are as expected
        2. test_result_geofabric_windows - Test the geofabric layers are as expected
    """

    @classmethod
    def setUpClass(cls):
        """Setup for test."""

        cls.test_path = pathlib.Path(__file__).parent.resolve()
        super(Test, cls).setUpClass()

        # Load in environment variables to get and set the private API keys
        dotenv.load_dotenv()
        linz_key = os.environ.get("LINZ_API", None)
        cls.instructions["default"] = {
            "datasets": {"vector": {"linz": {"key": linz_key}}}
        }

        # create fake catchment boundary
        x0 = 1768072
        y0 = 5473816
        x1 = 1769545
        y1 = 5472824
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(
            cls.instructions["dem"]["output"]["crs"]["horizontal"]
        )

        # save faked catchment boundary - used as land boundary as well
        catchment_file = cls.results_dir / "catchment.geojson"
        catchment.to_file(catchment_file)

        # Run pipeline - download files and generated DEM
        runner.from_instructions_dict(cls.instructions)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows test - this is strict")
    def test_result_windows(self):
        """A basic comparison between the generated and benchmark DEM"""

        # Load in benchmark
        file_path = self.cache_dir / self.instructions["dem"]["data_paths"]["benchmark"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark:
            benchmark.load()
        # Load in test
        file_path = (
            self.results_dir / self.instructions["dem"]["data_paths"]["result_dem"]
        )
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test:
            test.load()
        # Compare the generated and benchmark elevations
        diff_array = (
            test.z.data[~numpy.isnan(test.z.data)]
            - benchmark.z.data[~numpy.isnan(benchmark.z.data)]
        )
        logging.info(f"DEM elevation diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test.z.data,
            benchmark.z.data,
            err_msg="The generated result_geofabric has different elevation data from "
            "the benchmark_dem",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test
        del benchmark

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux test - this is less strict"
    )
    def test_result_linux(self):
        """A basic comparison between the generated and benchmark DEM"""

        # load in benchmark
        file_path = self.cache_dir / self.instructions["dem"]["data_paths"]["benchmark"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark:
            benchmark.load()
        # Load in test
        file_path = (
            self.results_dir / self.instructions["dem"]["data_paths"]["result_dem"]
        )
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test:
            test.load()
        # Get data generated from LiDAR
        lidar_mask = (test.data_source.data == 1) & (benchmark.data_source.data == 1)

        # Compare the generated and benchmark elevations
        lidar_diff = test.z.data[lidar_mask] - benchmark.z.data[lidar_mask]
        numpy.testing.assert_array_almost_equal(
            test.z.data[lidar_mask],
            benchmark.z.data[lidar_mask],
            decimal=6,
            err_msg="The generated test has significantly different elevation from the "
            f"benchmark where there is LiDAR: {lidar_diff}",
        )

        diff_array = (
            test.z.data[~numpy.isnan(test.z.data)]
            - benchmark.z.data[~numpy.isnan(test.z.data)]
        )
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")
        threshold = 10e-6
        percent = 2.5
        number_above_threshold = len(diff_array[numpy.abs(diff_array) > threshold])
        self.assertTrue(
            number_above_threshold < len(diff_array) * percent / 100,
            f"More than {percent}% of DEM values differ by more than {threshold} on Linux test"
            f" run: {diff_array[numpy.abs(diff_array) > threshold]} or "
            f"{number_above_threshold / len(diff_array.flatten()) * 100}%",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test
        del benchmark


if __name__ == "__main__":
    unittest.main()
