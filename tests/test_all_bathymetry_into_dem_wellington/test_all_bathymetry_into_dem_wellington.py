# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import json
import pathlib
import gc
import rioxarray
import shapely
import geopandas
import shutil
import dotenv
import os
import sys
import pytest
import logging
import numpy

from src.geofabrics import processor


class ProcessorRiverBathymetryTest(unittest.TestCase):
    """A class to test the basic river bathymetry estimation functionality
    contained in processor.RiverBathymetryGenerator.

    Tests run include:
        1. test_river_polygon - Test that the expected river polygon is created
        2. test_river_bathymetry - Test that the expected river bathymetry is created
        3. test_fan - Test that the expected fan polygon and geometry are created
    """

    @classmethod
    def setUpClass(cls):
        """Create a CatchmentGeometry object and then run the DemGenerator processing
        chain to download remote files and produce a DEM prior to testing."""

        test_path = pathlib.Path().cwd() / pathlib.Path(
            "tests/test_all_bathymetry_into_dem_wellington"
        )

        # Setup logging
        logging.basicConfig(
            filename=test_path / "test.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        logging.info("In test_all_bathymetry_into_dem_wellington.py")

        # load in the test instructions
        instruction_file_path = test_path / "instruction.json"
        with open(instruction_file_path, "r") as file_pointer:
            cls.instructions = json.load(file_pointer)
        # Load in environment variables to get and set the private API keys
        dotenv.load_dotenv()
        linz_key = os.environ.get("LINZ_API", None)
        cls.instructions["rivers"]["apis"]["linz"]["key"] = linz_key
        cls.instructions["drains"]["apis"]["linz"]["key"] = linz_key
        cls.instructions["dem"]["apis"]["linz"]["key"] = linz_key

        # Remove any files from last test, then create a results directory
        cls.cache_dir = test_path / "data"
        cls.results_dir = cls.cache_dir / "results"
        cls.tearDownClass()
        cls.results_dir.mkdir()

        # create fake catchment boundary
        x0 = 1768072
        y0 = 5473816
        x1 = 1769545
        y1 = 5472824
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(
            cls.instructions["rivers"]["output"]["crs"]["horizontal"]
        )

        # save faked catchment boundary - used as land boundary as well
        catchment_file = cls.results_dir / "catchment.geojson"
        catchment.to_file(catchment_file)

        # Run pipeline - download files and generated DEM
        runner = processor.RiverBathymetryGenerator(
            cls.instructions["rivers"], debug=False
        )
        runner.run()
        runner = processor.DrainBathymetryGenerator(
            cls.instructions["drains"], debug=False
        )
        runner.run()
        runner = processor.RawLidarDemGenerator(cls.instructions["dem"])
        runner.run()
        runner = processor.HydrologicDemGenerator(cls.instructions["dem"])
        runner.run()

    @classmethod
    def tearDownClass(cls):
        """Remove created cache directory and included created and downloaded files at
        the end of the test."""

        cls.clean_data_folder()

    @classmethod
    def clean_data_folder(cls):
        """Remove all generated or downloaded files from the data directory"""

        assert cls.cache_dir.exists(), (
            "The data directory that should include the comparison benchmark dem file "
            "doesn't exist"
        )

        for file in cls.results_dir.glob("*"):  # only files
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
        if cls.results_dir.exists():
            shutil.rmtree(cls.results_dir)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows test - this is strict")
    def test_result_dem_windows(self):
        """A basic comparison between the generated and benchmark DEM"""

        file_path = (
            self.cache_dir / self.instructions["dem"]["data_paths"]["benchmark_dem"]
        )
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark_dem:
            benchmark_dem.load()
        # Load in test DEM
        file_path = (
            self.results_dir / self.instructions["dem"]["data_paths"]["result_dem"]
        )
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test_dem:
            test_dem.load()
        # compare the generated and benchmark DEMs
        diff_array = (
            test_dem.z.data[~numpy.isnan(test_dem.z.data)]
            - benchmark_dem.z.data[~numpy.isnan(benchmark_dem.z.data)]
        )
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test_dem.z.data,
            benchmark_dem.z.data,
            err_msg="The generated result_dem has different data from the "
            + "benchmark_dem",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test_dem
        del benchmark_dem
        gc.collect()

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux test - this is less strict"
    )
    def test_result_dem_linux(self):
        """A basic comparison between the generated and benchmark DEM"""

        # load in benchmark DEM
        file_path = (
            self.cache_dir / self.instructions["dem"]["data_paths"]["benchmark_dem"]
        )
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark_dem:
            benchmark_dem.load()
        # Load in test DEM
        file_path = (
            self.results_dir / self.instructions["dem"]["data_paths"]["result_dem"]
        )
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test_dem:
            test_dem.load()
        # compare the generated and benchmark DEMs
        lidar_diff = (
            test_dem.z.data[test_dem.source_class.data == 1]
            - benchmark_dem.z.data[benchmark_dem.source_class.data == 1]
        )
        numpy.testing.assert_array_almost_equal(
            test_dem.z.data[test_dem.source_class.data == 1],
            benchmark_dem.z.data[test_dem.source_class.data == 1],
            decimal=6,
            err_msg="The generated test_dem has significantly different data from the "
            f"benchmark_dem where there is LiDAR: {lidar_diff}",
        )

        diff_array = (
            test_dem.z.data[~numpy.isnan(test_dem.z.data)]
            - benchmark_dem.z.data[~numpy.isnan(benchmark_dem.z.data)]
        )
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")
        threshold = 10e-6
        number_above_threshold = len(diff_array[numpy.abs(diff_array) > threshold])
        self.assertTrue(
            number_above_threshold < len(diff_array) * 0.25,
            f"More than 2.5% of DEM values differ by more than {threshold} on Linux test"
            f" run: {diff_array[numpy.abs(diff_array) > threshold]} or "
            f"{number_above_threshold / len(diff_array.flatten()) * 100}%",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test_dem
        del benchmark_dem
        gc.collect()


if __name__ == "__main__":
    unittest.main()
