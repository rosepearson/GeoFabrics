# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:11:25 2021

@author: pearsonra
"""

import unittest
import json
import pathlib
import geopandas
import shutil
import numpy
import rioxarray
import pytest
import sys
import logging

from src.geofabrics import processor


class ProcessorRiverBathymetryTest(unittest.TestCase):
    """ A class to test the basic river bathymetry estimation functionality
    contained in processor.RiverBathymetryGenerator.

    Tests run include:
        1. test_catchment - Test that the expected catchment geometry created
        2. test_dem - Test that generated DEM matches the benchmark DEM, where the
            rigor of the test depends on the operating system (windows or Linux)
        3. test_transects - Test the transect samples are as expected
        4. test_slope - Test the slope is as expected
        5. test_width - Test the width is as expected
    """

    # The expected datasets and files to be downloaded - used for comparison in the later tests
    CATCHMENT = {"area": 176682.30423209636, "length": 1640.933504236023}

    @classmethod
    def setUpClass(cls):
        """ Create a CatchmentGeometry object and then run the DemGenerator processing chain to download remote
        files and produce a DEM prior to testing. """

        test_path = pathlib.Path().cwd() / pathlib.Path("tests/test_river_bathymetry")

        # Setup logging
        logging.basicConfig(filename=test_path / 'test.log', encoding='utf-8', level=logging.INFO, force=True)
        logging.info("In test_river_bathymetry.py")

        # load in the test instructions
        instruction_file_path = test_path / "instruction.json"
        with open(instruction_file_path, 'r') as file_pointer:
            cls.instructions = json.load(file_pointer)

        # define cache location - and catchment dirs
        cls.cache_dir = pathlib.Path(cls.instructions['instructions']['data_paths']['local_cache'])

        # ensure the cache directory doesn't exist - i.e. clean up from last test occurred correctly
        cls.clean_data_folder()

        # Run pipeline - download files and generated DEM
        runner = processor.RiverBathymetryGenerator(cls.instructions)
        runner.run(
            pathlib.Path(cls.instructions['instructions']['data_paths']['local_cache']) / 'instruction_parameters.json')

    @classmethod
    def tearDownClass(cls):
        """ Remove created cache directory and included created and downloaded files at the end of the test. """

        cls.clean_data_folder()

    @classmethod
    def clean_data_folder(cls):
        """ Remove all generated or downloaded files from the data directory """

        assert cls.cache_dir.exists(), "The data directory that should include the comparison benchmark dem file " \
            "doesn't exist"

        benchmark_file = cls.cache_dir / "benchmark_dem.nc"
        rec_file = pathlib.Path(
            cls.instructions['instructions']['channel_bathymetry']['rec_file'])
        for file in cls.cache_dir.glob('*'):  # only files
            if file != benchmark_file and file != rec_file:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)

    def test_catchment(self):
        """ A test to see if the correct dataset is downloaded """

        catchment_file = pathlib.Path(
            self.instructions['instructions']['data_paths']['catchment_boundary'])
        catchment = geopandas.read_file(catchment_file)

        # check the area and length
        self.assertEqual(catchment.area.sum(), self.CATCHMENT['area'], "The "
                         f"catchment area should be {self.CATCHMENT['area']}"
                         f",but is {catchment.area.sum()}")
        self.assertEqual(catchment.length.sum(), self.CATCHMENT['length'],
                         "The catchment area should be"
                         f" {self.CATCHMENT['length']},but is "
                         f"{catchment.length.sum()}")

    @pytest.mark.skipif(sys.platform != 'win32', reason="Windows test - this is strict")
    def test_dem_windows(self):
        """ A basic comparison between the generated and benchmark DEM """

        # load in benchmark DEM
        with rioxarray.rioxarray.open_rasterio(self.instructions['instructions']['data_paths']['benchmark_dem'],
                                               masked=True) as benchmark_dem:
            benchmark_dem.load()

        # load in test DEM
        with rioxarray.rioxarray.open_rasterio(self.instructions['instructions']['data_paths']['result_dem'],
                                               masked=True) as test_dem:
            test_dem.load()

        # compare the generated and benchmark DEMs
        diff_array = test_dem.data[~numpy.isnan(test_dem.data)]-benchmark_dem.data[~numpy.isnan(benchmark_dem.data)]
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")
        numpy.testing.assert_array_almost_equal(test_dem.data[~numpy.isnan(test_dem.data)],
                                                benchmark_dem.data[~numpy.isnan(benchmark_dem.data)],
                                                err_msg="The generated result_dem has different data from the " +
                                                "benchmark_dem")


    '''def test_transects(self):
        """ A test to see if all expected dataset files are downloaded """

        dataset_dir = self.cache_dir / self.DATASET
        downloaded_files = [dataset_dir / file for file in self.FILE_SIZES.keys()]

        # check files are correct
        self.assertEqual(len(list(dataset_dir.glob('*'))), len(downloaded_files), "There should have been " +
                         f"{len(downloaded_files)} files downloaded into the {self.DATASET} directory, instead there " +
                         f"are {len(list(dataset_dir.glob('*')))} files/dirs in the directory")

        self.assertTrue(numpy.all([file in downloaded_files for file in dataset_dir.glob('*')]), "The downloaded files"
                        + f" {list(dataset_dir.glob('*'))} do not match the expected files {downloaded_files}")

    def test_slope(self):
        """ A test to see if all expected dataset files are of the right size """

        dataset_dir = self.cache_dir / self.DATASET
        downloaded_files = [dataset_dir / file for file in self.FILE_SIZES.keys()]

        # check sizes are correct
        self.assertTrue(numpy.all([downloaded_file.stat().st_size == self.FILE_SIZES[downloaded_file.name] for
                                   downloaded_file in downloaded_files]), "There is a miss-match between the size" +
                        f" of the downloaded files {[file.stat().st_size for file in downloaded_files]}" +
                        f" and the expected sizes of {self.FILE_SIZES.values()}")

    @pytest.mark.skipif(sys.platform != 'linux', reason="Linux test - this is less strict")
    def test_dem_linux(self):
        """ A basic comparison between the generated and benchmark DEM """

        # load in benchmark DEM
        with rioxarray.rioxarray.open_rasterio(self.instructions['instructions']['data_paths']['benchmark_dem'],
                                               masked=True) as benchmark_dem:
            benchmark_dem.load()

        # load in test DEM
        with rioxarray.rioxarray.open_rasterio(self.instructions['instructions']['data_paths']['result_dem'],
                                               masked=True) as test_dem:
            test_dem.load()

        # compare the generated and benchmark DEMs
        diff_array = test_dem.data[~numpy.isnan(test_dem.data)]-benchmark_dem.data[~numpy.isnan(benchmark_dem.data)]
        logging.info(f"DEM array diff is: {diff_array[diff_array != 0]}")

        threshold = 10e-2
        allowable_number_above = 3
        self.assertTrue(len(diff_array[numpy.abs(diff_array) > threshold]) <= allowable_number_above, "more than "
                        + f"{allowable_number_above} DEM values differ by more than {threshold} on Linux test run: "
                        + "{diff_array[numpy.abs(diff_array) > threshold]}")
        threshold = 10e-6
        self.assertTrue(len(diff_array[numpy.abs(diff_array) > threshold]) < len(diff_array) / 100,
                        f"{len(diff_array[numpy.abs(diff_array) > threshold])} or more than 1% of DEM values differ by "
                        + f" more than {threshold} on Linux test run: {diff_array[numpy.abs(diff_array) > threshold]}")'''


if __name__ == '__main__':
    unittest.main()
