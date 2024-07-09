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
from tests import base_test


class Test(base_test.Test):
    """Test the PatchDemGenerator processor class for a simple example with
    two patches specified in the instruction.json

    Tests run include:
        1. test_result_dem  Check the generated DEM matches the benchmark DEM within a
        tolerance
    """

    @classmethod
    def setUpClass(cls):
        """Create a cache directory and CatchmentGeometry object for use in the tests
        and also download the files used in the tests."""

        cls.test_path = pathlib.Path(__file__).parent.resolve()
        super(Test, cls).setUpClass()

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
