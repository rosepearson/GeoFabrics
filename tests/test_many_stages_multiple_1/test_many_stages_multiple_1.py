# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import json
import pathlib
import shapely
import geopandas
import shutil
import numpy
import rioxarray
import pytest
import sys
import logging
import gc

from src.geofabrics import runner


class Test(unittest.TestCase):
    """A class to test the basic processor class Processor functionality for remote
    tiles by downloading files from
    OpenTopography within a small region and then generating a DEM. All files are
    deleted after checking the DEM.

    Note this spans land and sea in comparison to the companion
    `test_processor_remote_tiles_wellington` that contains
    only ocean data. The benchmark shoes buildings are removed. No offshore information
    is provided.

    Tests run include:
        1. test_result_dem_windows/linux - Check the generated DEM matches the benchmark
           DEM, where the rigor of the test depends on the operating system (windows or
                                                                             Linux)
    """

    @classmethod
    def setUpClass(cls):
        """Create a CatchmentGeometry object and then run the DemGenerator processing
        chain to download remote files and produce a DEM prior to testing."""

        name = "test_many_stages_multiple_1"
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
        # Remove any files from last test, then create a results directory
        cls.cache_dir = test_path / "data"
        cls.results_dir = cls.cache_dir / "results"
        cls.tearDownClass()
        cls.results_dir.mkdir()

        # create fake catchment boundary
        x0 = 1778250
        y0 = 5470800
        x1 = 1778550
        y1 = 5470500
        catchment = shapely.geometry.Polygon(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        )
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(
            cls.instructions["shared"]["output"]["crs"]["horizontal"]
        )

        # save faked catchment boundary - used as land boundary as well
        catchment_file = cls.results_dir / "catchment.geojson"
        catchment.to_file(catchment_file)

        # Run pipeline - download files and generated DEM
        runner.from_instructions_dict(cls.instructions)

    @classmethod
    def tearDownClass(cls):
        """Remove created and downloaded files at the end of the test."""

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

    @pytest.mark.skipif(
        sys.platform != "win32", reason="Windows test - this is strict"
    )
    def test_result_dem_windows(self):
        """A basic comparison between the generated and benchmark DEM"""

        file_path = (
            self.cache_dir
            / self.instructions["shared"]["data_paths"]["benchmark"]
        )
        with rioxarray.rioxarray.open_rasterio(
            file_path, masked=True
        ) as benchmark:
            benchmark.load()
        # Load in test DEM
        file_path = (
            self.results_dir
            / self.instructions["roughness"]["data_paths"]["result_geofabric"]
        )
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test:
            test.load()
        # compare the generated and benchmark DEMs
        diff_array = (
            test.z.data[~numpy.isnan(test.z.data)]
            - benchmark.z.data[~numpy.isnan(benchmark.z.data)]
        )
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test.z.data,
            benchmark.z.data,
            err_msg="The generated result_dem has different data from the "
            + "benchmark_dem",
        )

        # compare the generated LiDAR data source maps
        diff_array = test.lidar_source.data - benchmark.lidar_source.data
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test.lidar_source.data,
            benchmark.lidar_source.data,
            err_msg="The generated lidar_source test has different data from "
            "the benchmark",
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
    def test_result_dem_linux(self):
        """A basic comparison between the generated and benchmark DEM"""

        file_path = (
            self.cache_dir
            / self.instructions["shared"]["data_paths"]["benchmark"]
        )
        with rioxarray.rioxarray.open_rasterio(
            file_path, masked=True
        ) as benchmark:
            benchmark.load()
        # Load in test DEM
        file_path = (
            self.results_dir
            / self.instructions["roughness"]["data_paths"]["result_geofabric"]
        )
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test:
            test.load()
        # compare the generated and benchmark DEMs
        diff_array = (
            test.z.data[~numpy.isnan(test.z.data)]
            - benchmark.z.data[~numpy.isnan(benchmark.z.data)]
        )
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")

        threshold = 10e-6
        self.assertTrue(
            len(diff_array[diff_array != 0]) < len(diff_array) / 100,
            f"{len(diff_array[diff_array != 0])} or more than 1% of DEM values differ "
            f"on Linux test run: {diff_array[diff_array != 0]}",
        )
        self.assertTrue(
            len(diff_array[numpy.abs(diff_array) > threshold])
            < len(diff_array) / 250,
            "More than 0.4% of DEM values differ by more than {threshold} on Linux test"
            f" run: {diff_array[numpy.abs(diff_array) > threshold]} or "
            f"{len(diff_array[numpy.abs(diff_array) > threshold]) / len(diff_array.flatten()) * 100}%",
        )

        # compare the generated and benchmark lidar source
        diff_array = test.lidar_source.data - benchmark.lidar_source.data
        numpy.testing.assert_array_almost_equal(
            test.lidar_source.data,
            benchmark.lidar_source.data,
            decimal=3,
            err_msg="The generated test has significantly different lidar "
            "source values from the benchmark where there is LiDAR: "
            f"{diff_array}",
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
