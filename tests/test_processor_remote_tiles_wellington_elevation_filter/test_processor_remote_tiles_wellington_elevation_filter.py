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

from src.geofabrics import processor


class ProcessorRemoteTilesWellingtonElevationFilterTest(unittest.TestCase):
    """ A class to test the LidarDemGenerator processor class elevation
    filtering functionality over remote tiles downloading files from
    OpenTopography within a small region. All files are deleted after checking
    the DEM.

    Tests run include:
        1. test_correct_dataset - Test that the expected dataset is downloaded from OpenTopography
        2. test_correct_lidar_files_downloaded - Test the downloaded LIDAR files have the expected names
        3. test_correct_lidar_file_size - Test the downloaded LIDAR files have the expected file sizes
        4. test_result_dem_windows/linux - Check the generated DEM matches the benchmark DEM, where the
            rigor of the test depends on the operating system (windows or Linux)
    """

    # The expected datasets and files to be downloaded - used for comparison in the later tests
    DATASET = "Wellington_2013"
    FILE_SIZES = {"ot_CL1_WLG_2013_1km_092044.laz": 23427023,
                  "ot_CL1_WLG_2013_1km_091044.laz": 26435976,
                  DATASET + "_TileIndex.zip": 598532}

    @classmethod
    def setUpClass(cls):
        """ Create a CatchmentGeometry object and then run the DemGenerator processing chain to download remote
        files and produce a DEM prior to testing. """

        test_path = pathlib.Path().cwd() / pathlib.Path("tests/test_processor_remote_tiles_wellington_elevation_filter")

        # Setup logging
        logging.basicConfig(filename=test_path / 'test.log', encoding='utf-8', level=logging.INFO, force=True)
        logging.info("In test_processor_remote_tiles_wellington_elevation_filter.py")

        # load in the test instructions
        instruction_file_path = test_path / "instruction.json"
        with open(instruction_file_path, 'r') as file_pointer:
            cls.instructions = json.load(file_pointer)

        # define cache location - and catchment dirs
        cls.cache_dir = pathlib.Path(cls.instructions['instructions']['data_paths']['local_cache'])

        # ensure the cache directory doesn't exist - i.e. clean up from last test occurred correctly
        cls.clean_data_folder()

        # create fake catchment boundary
        x0 = 1776358
        y0 = 5477084
        x1 = 1776629
        y1 = 5476914
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(cls.instructions['instructions']['output']['crs']['horizontal'])

        # save faked catchment boundary - used as land boundary as well
        catchment_file = cls.cache_dir / "catchment.geojson"
        catchment.to_file(catchment_file)

        # Run pipeline - download files and generated DEM
        runner = processor.LidarDemGenerator(cls.instructions)
        runner.run()

    @classmethod
    def tearDownClass(cls):
        """ Remove created and downloaded files at the end of the test. """

        cls.clean_data_folder()

    @classmethod
    def clean_data_folder(cls):
        """ Remove all generated or downloaded files from the data directory """

        assert cls.cache_dir.exists(), "The data directory that should include the comparison benchmark dem file " + \
            "doesn't exist"

        benchmark_file = cls.cache_dir / "benchmark_dem.nc"
        for file in cls.cache_dir.glob('*'):  # only files
            if file != benchmark_file and file.is_file():
                file.unlink()
            elif file != benchmark_file and file.is_dir():
                shutil.rmtree(file)

    def test_correct_dataset(self):
        """ A test to see if the correct dataset is downloaded """

        dataset_dir = self.cache_dir / self.DATASET

        # check the right dataset is downloaded - self.DATASET
        self.assertEqual(len(list(self.cache_dir.glob('*/**'))), 1,
                         f"There should only be one dataset named {self.DATASET} instead there are " +
                         f"{len(list(self.cache_dir.glob('*/**')))} list {list(self.cache_dir.glob('*/**'))}")

        self.assertEqual(len([file for file in self.cache_dir.iterdir() if file.is_dir() and file == dataset_dir]), 1,
                         f"Only the {self.DATASET} directory should have been downloaded. Instead we have: " +
                         f"{[file for file in self.cache_dir.iterdir() if file.is_dir()]}")

    def test_correct_files_downloaded(self):
        """ A test to see if all expected dataset files are downloaded """

        dataset_dir = self.cache_dir / self.DATASET
        downloaded_files = [dataset_dir / file for file in self.FILE_SIZES.keys()]

        # check files are correct
        self.assertEqual(len(list(dataset_dir.glob('*'))), len(downloaded_files), "There should have been " +
                         f"{len(downloaded_files)} files downloaded into the {self.DATASET} directory, instead there " +
                         f"are {len(list(dataset_dir.glob('*')))} files/dirs in the directory")

        self.assertTrue(numpy.all([file in downloaded_files for file in dataset_dir.glob('*')]), "The downloaded files"
                        + f" {list(dataset_dir.glob('*'))} do not match the expected files {downloaded_files}")

    def test_correct_file_size(self):
        """ A test to see if all expected dataset files are of the right size """

        dataset_dir = self.cache_dir / self.DATASET
        downloaded_files = [dataset_dir / file for file in self.FILE_SIZES.keys()]

        # check sizes are correct
        self.assertTrue(numpy.all([downloaded_file.stat().st_size == self.FILE_SIZES[downloaded_file.name] for
                                   downloaded_file in downloaded_files]), "There is a miss-match between the size" +
                        f" of the downloaded files {[file.stat().st_size for file in downloaded_files]}" +
                        f" and the expected sizes of {self.FILE_SIZES.values()}")

    @pytest.mark.skipif(sys.platform != 'win32', reason="Windows test - this is strict")
    def test_result_dem_windows(self):
        """ A basic comparison between the generated and benchmark DEM """

        file_path = self.cache_dir / self.instructions['instructions']['data_paths']['benchmark_dem']
        with rioxarray.rioxarray.open_rasterio(file_path,
                                               masked=True) as benchmark_dem:
            benchmark_dem.load()

        # Load in test DEM
        file_path = self.cache_dir / self.instructions['instructions']['data_paths']['result_dem']
        with rioxarray.rioxarray.open_rasterio(file_path,
                                               masked=True) as test_dem:
            test_dem.load()

        # compare the generated and benchmark DEMs
        diff_array = test_dem.z.data[~numpy.isnan(test_dem.z.data)] \
            - benchmark_dem.z.data[~numpy.isnan(benchmark_dem.z.data)]
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(test_dem.z.data, benchmark_dem.z.data,
                                                err_msg="The generated result_dem has different data from the " +
                                                "benchmark_dem")

        # explicitly free memory as xarray seems to be hanging onto memory
        del test_dem
        del benchmark_dem
        gc.collect()

    @pytest.mark.skipif(sys.platform != 'linux', reason="Linux test - this is less strict")
    def test_result_dem_linux(self):
        """ A basic comparison between the generated and benchmark DEM """

        # load in benchmark DEM
        file_path = self.cache_dir / self.instructions['instructions']['data_paths']['benchmark_dem']
        with rioxarray.rioxarray.open_rasterio(file_path,
                                               masked=True) as benchmark_dem:
            benchmark_dem.load()

        # Load in test DEM
        file_path = self.cache_dir / self.instructions['instructions']['data_paths']['result_dem']
        with rioxarray.rioxarray.open_rasterio(file_path,
                                               masked=True) as test_dem:
            test_dem.load()

        # compare the generated and benchmark DEMs
        diff_array = test_dem.z.data[~numpy.isnan(test_dem.z.data)] \
            - benchmark_dem.z.data[~numpy.isnan(benchmark_dem.z.data)]
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")

        threshold = 10e-6
        self.assertTrue(len(diff_array[diff_array != 0]) < len(diff_array) / 100, f"{len(diff_array[diff_array != 0])} "
                        + f"or more than 1% of DEM values differ on Linux test run: {diff_array[diff_array != 0]}")
        self.assertTrue(len(diff_array[numpy.abs(diff_array) > threshold]) < len(diff_array) / 250, "More than 0.4% of"
                        + " DEM values differ by more than {threshold} on Linux test run: " +
                        f"{diff_array[numpy.abs(diff_array) > threshold]} or " +
                        f"{len(diff_array[numpy.abs(diff_array) > threshold]) / len(diff_array.flatten()) * 100}%")

        # explicitly free memory as xarray seems to be hanging onto memory
        del test_dem
        del benchmark_dem
        gc.collect()


if __name__ == '__main__':
    unittest.main()