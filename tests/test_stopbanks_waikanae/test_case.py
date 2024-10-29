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

from geofabrics import runner
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

        crs = cls.instructions["dem"]["output"]["crs"]["horizontal"]

        # create and save fake catchment boundary
        x0 = 1771850
        y0 = 5472250
        x1 = 1772150
        y1 = 5472600
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoDataFrame(geometry=[catchment], crs=crs)
        catchment_file = cls.results_dir / "catchment.geojson"
        catchment.to_file(catchment_file)

        # create and save stopbank masking polygon
        x0 = 1771866
        y0 = 5472488
        x1 = 1771882
        y1 = 5472568
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoDataFrame(geometry=[catchment], crs=crs)
        catchment_file = cls.results_dir / "stopbank_masking_polygon.geojson"
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

        # Comparisons
        numpy.testing.assert_array_almost_equal(
            test.z.data,
            benchmark.z.data,
            decimal=1,
            err_msg="The generated test has significantly different elevation from the "
            f"benchmark.",
        )
        
        numpy.testing.assert_array_almost_equal(
            test.data_source.data,
            benchmark.data_source.data,
            err_msg="The generated test data_source layer differs from the "
            f"benchmark.",
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

        # Comparisons
        numpy.testing.assert_array_almost_equal(
            test.z.data,
            benchmark.z.data,
            decimal=1,
            err_msg="The generated test has significantly different elevation from the "
            f"benchmark.",
        )
        numpy.testing.assert_array_almost_equal(
            test.data_source.data,
            benchmark.data_source.data,
            err_msg="The generated test data_source layer differs from the "
            f"benchmark.",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test
        del benchmark


if __name__ == "__main__":
    unittest.main()
