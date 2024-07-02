# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import json
import pathlib
import geopandas
import shapely
import shutil
import sys
import pytest
import logging
import numpy
import rioxarray
import gc

from src.geofabrics import runner


class Test(unittest.TestCase):
    """A class to test the basic measured river interpolation functionality
    contained in processor.MeasuredRiverGenerator.

    Tests run include:
        1. test_river_polygon(linux/windows) - Test that the expected river polygon is
        created
        2. test_river_elevations(linux/windows) - Test that the expected river
        bathymetry is created
    """

    @classmethod
    def setUpClass(cls):
        """Create a CatchmentGeometry object and then run the DemGenerator processing
        chain to download remote files and produce a DEM prior to testing."""

        test_path = pathlib.Path().cwd() / pathlib.Path(
            "tests/test_many_stages_westport"
        )

        # Setup logging
        logging.basicConfig(
            filename=test_path / "test.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        logging.info("In test_many_stages_westport")

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
        x0 = 1482951
        y0 = 5375247
        x1 = 1484180
        y1 = 5373303
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(
            cls.instructions["default"]["output"]["crs"]["horizontal"]
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

        gc.collect()
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
            test_dem.z.data[~numpy.isnan(test_dem.z.data)],
            benchmark_dem.z.data[~numpy.isnan(benchmark_dem.z.data)],
            err_msg="The generated result_dem has different data from the "
            "benchmark_dem",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test_dem
        del benchmark_dem

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux test - this is less strict"
    )
    def test_result_dem_linux(self):
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

        threshold = 10e-2
        allowable_number_above = 3
        self.assertTrue(
            len(diff_array[numpy.abs(diff_array) > threshold])
            <= allowable_number_above,
            "more than "
            f"{allowable_number_above} DEM values differ by more than {threshold} on"
            f" Linux test run: {diff_array[numpy.abs(diff_array) > threshold]}",
        )
        threshold = 10e-6
        self.assertTrue(
            len(diff_array[numpy.abs(diff_array) > threshold]) < len(diff_array) / 100,
            f"{len(diff_array[numpy.abs(diff_array) > threshold])} or more than 1% of "
            f"DEM values differ by more than {threshold} on Linux test run: "
            f"{diff_array[numpy.abs(diff_array) > threshold]}",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test_dem
        del benchmark_dem


if __name__ == "__main__":
    unittest.main()
