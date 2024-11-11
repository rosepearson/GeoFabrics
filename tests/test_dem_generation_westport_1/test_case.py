# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import pathlib
import shapely
import geopandas
import numpy
import rioxarray
import pytest
import sys
import dotenv
import os
import logging

from geofabrics import processor
from tests import base_test


class Test(base_test.Test):
    """A class to test the basic processor class DemGenerator functionality for remote
    LiDAR tiles and remote Bathymetry contours and coast contours by downloading files
    from OpenTopography and the LINZ data portal within a small region and then
    generating a DEM. All files are deleted after checking the DEM.

    Note in comparison to the companion `test_processor_remote_tiles_westport` test
    hedges and the like are removed on land. Offshore values match the inbounds contours
    provided by bathymetry layer 50448.

    Tests run include:
        1. test_correct_datasets - Test that the expected datasets are downloaded from
           OpenTopography and LINZ
        2. test_correct_lidar_files_downloaded - Test the downloaded LIDAR files have
           the expected names
        3. test_correct_lidar_file_size - Test the downloaded LIDAR files have the
           expected file sizes
        4. test_result_dem_windows/linux - Check the generated DEM matches the benchmark
           DEM, where the rigor of the test depends on the operating system (windows or
                                                                             Linux)
    """

    # The expected datasets and files to be downloaded - used for comparison in the
    # later tests
    DATATYPES = ["lidar", "vector"]
    DATASET = "NZ20_Westport"
    LIDAR_SIZES = {
        "CL2_BR20_2020_1000_4012.laz": 2636961,
        "CL2_BR20_2020_1000_4013.laz": 3653378,
        "CL2_BR20_2020_1000_4014.laz": 4470413,
        "CL2_BR20_2020_1000_4112.laz": 9036407,
        "CL2_BR20_2020_1000_4212.laz": 8340310,
        "CL2_BR20_2020_1000_4213.laz": 6094309,
        "CL2_BR20_2020_1000_4214.laz": 8492543,
        DATASET + "_TileIndex.zip": 1848391,
    }

    @classmethod
    def setUpClass(cls):
        """Setup for test."""

        cls.test_path = pathlib.Path(__file__).parent.resolve()
        super(Test, cls).setUpClass()

        # Load in environment variables to get and set the private API keys
        dotenv.load_dotenv()
        linz_key = os.environ.get("LINZ_API", None)
        cls.instructions["datasets"]["vector"]["linz"]["key"] = linz_key

        # Create fake catchment boundary
        x0 = 1473354
        x1 = 1473704
        x2 = 1474598
        y0 = 5377655
        y1 = 5377335
        y2 = 5376291
        y3 = 5375824
        catchment = shapely.geometry.Polygon(
            [
                (x0, y0),
                (x0, y3),
                (x2, y3),
                (x2, y2),
                (x1, y2),
                (x1, y1),
                (x2, y1),
                (x2, y0),
            ]
        )
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(cls.instructions["output"]["crs"]["horizontal"])

        # Save faked catchment boundary - used as land boundary as well
        catchment_file = cls.results_dir / "catchment.geojson"
        catchment.to_file(catchment_file)

        # Run pipeline - download files and generated DEM
        runner = processor.RawLidarDemGenerator(cls.instructions)
        runner.run()
        runner = processor.HydrologicDemGenerator(cls.instructions)
        runner.run()

    def test_correct_datasets(self):
        """A test to see if the correct datasets were downloaded"""

        downloads_dir = self.cache_dir / "downloads"
        dataset_dirs = [downloads_dir / dataset for dataset in self.DATATYPES]
        # Check the right dataset is downloaded - self.DATASET
        self.assertEqual(
            len(
                [
                    directory
                    for directory in downloads_dir.glob("*")
                    if directory.is_dir()
                ]
            ),
            len(dataset_dirs),
            f"There should only be {len(dataset_dirs)} datasets named {dataset_dirs}, "
            f"but instead there are {len(list(downloads_dir.glob('*')))} list "
            f"{list(downloads_dir.glob('*'))}",
        )

        self.assertEqual(
            len(
                [
                    file
                    for file in downloads_dir.iterdir()
                    if file.is_dir() and file in dataset_dirs
                ]
            ),
            len(dataset_dirs),
            f"Only the {dataset_dirs} directories should have been downloaded. Instead "
            f"we have: {[file for file in downloads_dir.iterdir() if file.is_dir()]}",
        )

    def test_correct_lidar_files_downloaded(self):
        """A test to see if all expected LiDAR dataset files are downloaded"""

        dataset_dir = self.cache_dir / "downloads" / "lidar" / self.DATASET
        downloaded_files = [dataset_dir / file for file in self.LIDAR_SIZES.keys()]
        for file in downloaded_files:
            print(f"{file.name} of size {file.stat().st_size}")
        # Check files are correct
        self.assertEqual(
            len(list(dataset_dir.glob("*"))),
            len(downloaded_files),
            f"There should have been {len(downloaded_files)} files downloaded into the "
            f"{self.DATASET} directory, instead there are "
            f" {len(list(dataset_dir.glob('*')))} files/dirs in the directory",
        )

        self.assertTrue(
            numpy.all([file in downloaded_files for file in dataset_dir.glob("*")]),
            f"The downloaded files {list(dataset_dir.glob('*'))} do not match the "
            f"expected files {downloaded_files}",
        )

    def test_correct_lidar_file_size(self):
        """A test to see if all expected LiDAR dataset files are of the right size"""

        dataset_dir = self.cache_dir / "downloads" / "lidar" / self.DATASET
        downloaded_files = [dataset_dir / file for file in self.LIDAR_SIZES.keys()]

        # Check sizes are correct
        self.assertTrue(
            numpy.all(
                [
                    downloaded_file.stat().st_size
                    == self.LIDAR_SIZES[downloaded_file.name]
                    for downloaded_file in downloaded_files
                ]
            ),
            "There is a miss-match between the size of the downloaded files "
            f"{[file.stat().st_size for file in downloaded_files]} and the expected "
            f"sizes of {self.LIDAR_SIZES.values()}",
        )

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows test - this is strict")
    def test_result_dem_windows(self):
        """A basic comparison between the generated and benchmark DEM"""
        decimal_threshold = 5
        # Load in benchmark DEM
        file_path = self.cache_dir / self.instructions["data_paths"]["benchmark"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark:
            benchmark.load()
        # Load in test DEM
        file_path = self.results_dir / self.instructions["data_paths"]["result_dem"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test_dem:
            test_dem.load()
        # Compare the generated and benchmark DEMs
        diff_array = (
            test_dem.z.data[~numpy.isnan(test_dem.z.data)]
            - benchmark.z.data[~numpy.isnan(benchmark.z.data)]
        )
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test_dem.z.data[~numpy.isnan(test_dem.z.data)],
            benchmark.z.data[~numpy.isnan(benchmark.z.data)],
            decimal=decimal_threshold,
            err_msg="The generated result_dem has different data from the " "benchmark",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test_dem
        del benchmark

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux test - this is less strict"
    )
    def test_result_dem_linux(self):
        """A basic comparison between the generated and benchmark DEM"""

        # Load in benchmark DEM
        file_path = self.cache_dir / self.instructions["data_paths"]["benchmark"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark:
            benchmark.load()
        # Load in test DEM
        file_path = self.results_dir / self.instructions["data_paths"]["result_dem"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test_dem:
            test_dem.load()
        # Compare the generated and benchmark DEMs
        diff_array = (
            test_dem.z.data[~numpy.isnan(test_dem.z.data)]
            - benchmark.z.data[~numpy.isnan(benchmark.z.data)]
        )
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")

        threshold = 10e-2
        allowable_number_above = 5
        self.assertTrue(
            len(diff_array[numpy.abs(diff_array) > threshold])
            <= allowable_number_above,
            "Some DEM values "
            + f"differ by more than {threshold} on Linux test run: "
            + f"{diff_array[numpy.abs(diff_array) > threshold]}",
        )
        threshold = 10e-6
        self.assertTrue(
            len(diff_array[numpy.abs(diff_array) > threshold]) < len(diff_array) / 100,
            f"{len(diff_array[numpy.abs(diff_array) > threshold])} or more than 1% of "
            f"DEM values differ by more than {threshold} on Linux test run: "
            f" {diff_array[numpy.abs(diff_array) > threshold]}",
        )

        # explicitly free memory as xarray seems to be hanging onto memory
        del test_dem
        del benchmark


if __name__ == "__main__":
    unittest.main()
