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

from src.geofabrics import runner


class Test(unittest.TestCase):
    """A class to test the basic river bathymetry estimation functionality
    contained in processor.RiverBathymetryGenerator.

    Tests run include:
        1. test_result_geofabric_linux - Test the geofabric layers are as expected
        2. test_result_geofabric_windows - Test the geofabric layers are as expected
    """

    @classmethod
    def setUpClass(cls):
        """Create a CatchmentGeometry object and then run the DemGenerator processing
        chain to download remote files and produce a DEM prior to testing."""
        name = "test_many_stages_wellington"
        test_path = pathlib.Path().cwd() / "tests" / name

        # Setup logging
        logging.basicConfig(
            filename=test_path / "test.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        logging.info(f"In {name}")

        # load in the test instructions
        instruction_file_path = test_path / "instruction.json"
        with open(instruction_file_path, "r") as file_pointer:
            cls.instructions = json.load(file_pointer)
        # Load in environment variables to get and set the private API keys
        dotenv.load_dotenv()
        linz_key = os.environ.get("LINZ_API", None)
        cls.instructions["default"] = {
            "datasets": {"vector": {"linz": {"key": linz_key}}}
        }

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
            cls.instructions["dem"]["output"]["crs"]["horizontal"]
        )

        # save faked catchment boundary - used as land boundary as well
        catchment_file = cls.results_dir / "catchment.geojson"
        catchment.to_file(catchment_file)

        # Run pipeline - download files and generated DEM
        runner.from_instructions_dict(cls.instructions)

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

        # Cycle through all folders within the cache dir deleting their contents
        for path in cls.cache_dir.iterdir():
            if path.is_dir():
                for file in path.glob("*"):  # only files
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        shutil.rmtree(file)
                shutil.rmtree(path)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows test - this is strict")
    def test_result_geofabric_windows(self):
        """A basic comparison between the generated and benchmark DEM"""

        # Load in benchmark
        file_path = (
            self.cache_dir / self.instructions["roughness"]["data_paths"]["benchmark"]
        )
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark:
            benchmark.load()
        # Load in test
        file_path = (
            self.results_dir
            / self.instructions["roughness"]["data_paths"]["result_geofabric"]
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
        # Compare the generated and benchmark roughness
        diff_array = (
            test.zo.data[~numpy.isnan(test.zo.data)]
            - benchmark.zo.data[~numpy.isnan(benchmark.zo.data)]
        )
        logging.info(f"Roughness diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test.zo.data,
            benchmark.zo.data,
            err_msg="The generated result_geofabric has different roughness data from "
            "the benchmark_dem",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test
        del benchmark
        gc.collect()

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux test - this is less strict"
    )
    def test_result_geofabric_linux(self):
        """A basic comparison between the generated and benchmark DEM"""

        # load in benchmark
        file_path = (
            self.cache_dir / self.instructions["roughness"]["data_paths"]["benchmark"]
        )
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark:
            benchmark.load()
        # Load in test
        file_path = (
            self.results_dir
            / self.instructions["roughness"]["data_paths"]["result_geofabric"]
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
        # Compare the generated and benchmark roughnesses
        diff_array = test.zo.data - benchmark.zo.data
        numpy.testing.assert_array_almost_equal(
            test.zo.data,
            benchmark.zo.data,
            decimal=3,
            err_msg="The generated test has significantly different roughness from the "
            f"benchmark where there is LiDAR: {diff_array}",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test
        del benchmark
        gc.collect()


if __name__ == "__main__":
    unittest.main()
