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
import dotenv
import os
import logging
import gc

from src.geofabrics import processor


class ProcessorRemoteAllWestportTest(unittest.TestCase):
    """Test the DemGenerator class functionality for remote LiDAR tiles and remote
    Bathymetry contours, a remote reference DEM and a remote land outline by downloading
    files from OpenTopography and the LINZ data portal within a small region and then
    generating a DEM. All files are deleted after checking the DEM.

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
    DATASETS = ["NZ20_Westport", "51153", "51768"]
    LIDAR_SIZES = {
        "CL2_BR21_2020_1000_4704.laz": 20851153,
        "CL2_BR21_2020_1000_4705.laz": 19749374,
        "CL2_BR21_2020_1000_4706.laz": 17977826,
        "CL2_BR21_2020_1000_4804.laz": 18379794,
        DATASETS[0] + "_TileIndex.zip": 1125874,
    }

    @classmethod
    def setUpClass(cls):
        """Create a CatchmentGeometry object and then run the DemGenerator processing
        chain to download remote files and produce a DEM prior to testing."""

        test_path = pathlib.Path().cwd() / pathlib.Path(
            "tests/test_processor_remote_all_westport"
        )

        # Setup logging
        logging.basicConfig(
            filename=test_path / "test.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        logging.info("In test_processor_remote_all_westport.py")

        # Load in the test instructions
        instruction_file_path = test_path / "instruction.json"
        with open(instruction_file_path, "r") as file_pointer:
            cls.instructions = json.load(file_pointer)
        # Load in environment variables to get and set the private API keys
        dotenv.load_dotenv()
        linz_key = os.environ.get("LINZ_API", None)
        cls.instructions["apis"]["vector"]["linz"]["key"] = linz_key
        cls.instructions["apis"]["raster"]["linz"]["key"] = linz_key

        # Remove any files from last test, then create a results directory
        cls.cache_dir = test_path / "data"
        cls.results_dir = cls.cache_dir / "results"
        cls.tearDownClass()
        cls.results_dir.mkdir()

        # Create fake catchment boundary
        x0 = 1493600
        x1 = 1494400
        y0 = 5372300
        y1 = 5371700
        catchment = shapely.geometry.Polygon(
            [
                (x0, y0),
                (x0, y1),
                (x1, y1),
                (x1, y0),
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

    def test_correct_datasets(self):
        """A test to see if the correct datasets were downloaded"""

        dataset_dirs = [self.cache_dir / dataset for dataset in self.DATASETS]

        # Check the right dataset is downloaded - self.DATASET
        self.assertEqual(
            len(list(self.cache_dir.glob("*/**"))),
            len(dataset_dirs) + 1,
            f"There should only be {len(dataset_dirs)} datasets named {dataset_dirs} "
            f"and the results dir {self.results_dir}, but instead there are "
            f" {len(list(self.cache_dir.glob('*/**')))} list "
            f"{list(self.cache_dir.glob('*/**'))}",
        )

        self.assertEqual(
            len(
                [
                    file
                    for file in self.cache_dir.iterdir()
                    if file.is_dir() and file in dataset_dirs
                ]
            ),
            len(dataset_dirs),
            f"Only the {dataset_dirs} directories should have been downloaded. Instead "
            f"we have: {[file for file in self.cache_dir.iterdir() if file.is_dir()]}",
        )

    def test_correct_lidar_files_downloaded(self):
        """A test to see if all expected LiDAR dataset files are downloaded"""

        dataset_dir = self.cache_dir / self.DATASETS[0]
        downloaded_files = [dataset_dir / file for file in self.LIDAR_SIZES.keys()]
        for file in downloaded_files:
            print(f"{file.name} of size {file.stat().st_size}")
        # Check files are correct
        self.assertEqual(
            len(list(dataset_dir.glob("*"))),
            len(downloaded_files),
            f"There should have been {len(downloaded_files)} files downloaded into the "
            f"{self.DATASETS[0]} directory, instead there are "
            f" {len(list(dataset_dir.glob('*')))} files/dirs in the directory",
        )

        self.assertTrue(
            numpy.all([file in downloaded_files for file in dataset_dir.glob("*")]),
            f"The downloaded files {list(dataset_dir.glob('*'))} do not match the "
            f"expected files {downloaded_files}",
        )

    def test_correct_lidar_file_size(self):
        """A test to see if all expected LiDAR dataset files are of the right size"""

        dataset_dir = self.cache_dir / self.DATASETS[0]
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

        # Load in benchmark DEM
        file_path = self.cache_dir / self.instructions["data_paths"]["benchmark_dem"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark_dem:
            benchmark_dem.load()
        # Load in test DEM
        file_path = self.results_dir / self.instructions["data_paths"]["result_dem"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test_dem:
            test_dem.load()
        # Compare DEMs - load both from file as rioxarray.rioxarray.open_rasterio
        # ignores index order
        diff_array = (
            test_dem.z.data[~numpy.isnan(test_dem.z.data)]
            - benchmark_dem.z.data[~numpy.isnan(benchmark_dem.z.data)]
        )
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(
            test_dem.z.data[~numpy.isnan(test_dem.z.data)],
            benchmark_dem.z.data[~numpy.isnan(benchmark_dem.z.data)],
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

        # Load in benchmark DEM
        file_path = self.cache_dir / self.instructions["data_paths"]["benchmark_dem"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as benchmark_dem:
            benchmark_dem.load()
        # Load in test DEM
        file_path = self.results_dir / self.instructions["data_paths"]["result_dem"]
        with rioxarray.rioxarray.open_rasterio(file_path, masked=True) as test_dem:
            test_dem.load()
        # Compare the generated and benchmark DEMs
        diff_array = (
            test_dem.z.data[~numpy.isnan(test_dem.z.data)]
            - benchmark_dem.z.data[~numpy.isnan(benchmark_dem.z.data)]
        )
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")

        threshold = 10e-2
        allowable_number_above = 2
        self.assertTrue(
            len(diff_array[numpy.abs(diff_array) > threshold])
            <= allowable_number_above,
            f"more than {allowable_number_above} differ by more than DEM values differ "
            f" by more than{threshold} on Linux test run: "
            f"{diff_array[numpy.abs(diff_array) > threshold]}",
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
        del benchmark_dem
        gc.collect()


if __name__ == "__main__":
    unittest.main()
