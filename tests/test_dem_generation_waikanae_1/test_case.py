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
import logging

from geofabrics import processor
from tests import base_test


class Test(base_test.Test):
    """A class to test the basic processor class Processor functionality for remote
    tiles by downloading files from OpenTopography within a small region and then
    generating a DEM. All files are deleted after checking the DEM.

    Tests run include:
        1. test_correct_dataset - Test that the expected dataset is downloaded from
           OpenTopography
        2. test_correct_lidar_files_downloaded - Test the downloaded LIDAR files have
           the expected names
        3. test_correct_lidar_file_size - Test the downloaded LIDAR files have the
           expected file sizes
        4. test_result_dem_windows/linux - Check the generated DEM matches the benchmark
           DEM, where the rigor of the test depends on the operating system (windows or
                                                                             Linux)
    """

    # The expected datasets and files to be downloaded - used for comparison in tests
    DATASET = "Wellington_2013"
    FILE_SIZES = {
        "ot_CL1_WLG_2013_1km_085033.laz": 6795072,
        "ot_CL1_WLG_2013_1km_086033.laz": 5712485,
        "ot_CL1_WLG_2013_1km_085032.laz": 1670549,
        "ot_CL1_WLG_2013_1km_086032.laz": 72787,
        DATASET + "_TileIndex.zip": 598532,
    }

    @classmethod
    def setUpClass(cls):
        """Setup for test."""

        cls.test_path = pathlib.Path(__file__).parent.resolve()
        super(Test, cls).setUpClass()

        # create fake catchment boundary
        x0 = 1764864
        y0 = 5470382
        x1 = 1765656
        y1 = 5471304
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(cls.instructions["output"]["crs"]["horizontal"])

        # save faked catchment boundary - used as land boundary as well
        catchment_file = cls.results_dir / "catchment.geojson"
        catchment.to_file(catchment_file)

        # Run pipeline - download files and generated DEM
        runner = processor.RawLidarDemGenerator(cls.instructions)
        runner.run()
        runner = processor.HydrologicDemGenerator(cls.instructions)
        runner.run()

    def test_correct_dataset(self):
        """A test to see if the correct dataset is downloaded"""
        downloads_dir = self.cache_dir / "downloads" / "lidar"
        dataset_dir = downloads_dir / self.DATASET

        # check the right dataset is downloaded - self.DATASET
        self.assertEqual(
            len(list(downloads_dir.glob("*/**"))),
            1,
            f"There should only be one dataset named {self.DATASET} instead "
            f"there are {len(list(downloads_dir.glob('*/**')))} list "
            f"{list(downloads_dir.glob('*/**'))}",
        )

        self.assertEqual(
            len(
                [
                    file
                    for file in downloads_dir.iterdir()
                    if file.is_dir() and file == dataset_dir
                ]
            ),
            1,
            f"Only the {self.DATASET} directory should have been downloaded. Instead we"
            f" have: {[file for file in downloads_dir.iterdir() if file.is_dir()]}",
        )

    def test_correct_files_downloaded(self):
        """A test to see if all expected dataset files are downloaded"""

        dataset_dir = self.cache_dir / "downloads" / "lidar" / self.DATASET
        downloaded_files = [dataset_dir / file for file in self.FILE_SIZES.keys()]

        # check files are correct
        self.assertEqual(
            len(list(dataset_dir.glob("*"))),
            len(downloaded_files),
            "There should have been "
            f"{len(downloaded_files)} files downloaded into the {self.DATASET} "
            f"directory, instead there are {len(list(dataset_dir.glob('*')))} "
            "files/dirs in the directory",
        )

        self.assertTrue(
            numpy.all([file in downloaded_files for file in dataset_dir.glob("*")]),
            f"The downloaded files {list(dataset_dir.glob('*'))} do not match the "
            f"expected files {downloaded_files}",
        )

    def test_correct_file_size(self):
        """A test to see if all expected dataset files are of the right size"""

        dataset_dir = self.cache_dir / "downloads" / "lidar" / self.DATASET
        downloaded_files = [dataset_dir / file for file in self.FILE_SIZES.keys()]

        # check sizes are correct
        self.assertTrue(
            numpy.all(
                [
                    downloaded_file.stat().st_size
                    == self.FILE_SIZES[downloaded_file.name]
                    for downloaded_file in downloaded_files
                ]
            ),
            "There is a miss-match between the size of the downloaded files "
            + f"{[file.stat().st_size for file in downloaded_files]} and the expected "
            + f"sizes of {self.FILE_SIZES.values()}",
        )

    @pytest.mark.skipif(
        sys.platform != "win32", reason="Windows test - this is less strict"
    )
    def test_result_dem_windows(self):
        """A basic comparison between the generated and benchmark DEM"""

        file_path = self.cache_dir / self.instructions["data_paths"]["benchmark_dem"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark:
            benchmark.load()
        # Load in test DEM
        file_path = self.results_dir / self.instructions["data_paths"]["result_dem"]
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
            f"{len(diff_array[diff_array != 0])} or more than 1% of DEM values differ"
            f" on Linux test run: {diff_array[diff_array != 0]}",
        )
        number_under_threshold = len(diff_array[numpy.abs(diff_array) > threshold])
        self.assertTrue(
            number_under_threshold < len(diff_array) / 100,
            f"More than 0.4% of DEM values differ by more than {threshold} on Linux"
            f" test run: {diff_array[numpy.abs(diff_array) > threshold]} or "
            f"{number_under_threshold / len(diff_array.flatten()) * 100}%",
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

        # explicitly free memory as xarray seems to be hanging onto memory
        del test
        del benchmark

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Linux test - this is more strict"
    )
    def test_result_dem_linux(self):
        """A basic comparison between the generated and benchmark DEM"""

        # load in benchmark DEM
        file_path = self.cache_dir / self.instructions["data_paths"]["benchmark_dem"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark:
            benchmark.load()
        # Load in test DEM
        file_path = self.results_dir / self.instructions["data_paths"]["result_dem"]
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

        # explicitly free memory as xarray seems to be hanging onto memory
        del test
        del benchmark


if __name__ == "__main__":
    unittest.main()
