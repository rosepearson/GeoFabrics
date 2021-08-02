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

from src.geofabrics import lidar_fetch
from src.geofabrics import geometry


class OpenTopographyTest(unittest.TestCase):
    """ A class to test the basic lidar_fetch class OpenTopography functionality by downloading files from
    OpenTopography within a small region. All files are deleted after checking their names and size.

    Tests run include:
        1. test_correct_dataset - Test that the expected dataset is downloaded from OpenTopography
        2. test_correct_lidar_files_downloaded - Test the downloaded LIDAR files have the expected names
        3. test_correct_lidar_file_size - Test the downloaded LIDAR files have the expected file sizes
    """

    # The expected datasets and files to be downloaded - used for comparison in the later tests
    DATASET = "Wellington_2013"
    FILE_SIZES = {"ot_CL1_WLG_2013_1km_085033.laz": 6795072, "ot_CL1_WLG_2013_1km_086033.laz": 5712485,
                  "ot_CL1_WLG_2013_1km_085032.laz": 1670549, "ot_CL1_WLG_2013_1km_086032.laz": 72787,
                  DATASET + "_TileIndex.zip": 598532}

    @classmethod
    def setUpClass(cls):
        """ Create a cache directory and CatchmentGeometry object for use in the tests and also download the files used
        in the tests. """

        # load in the test instructions
        file_path = pathlib.Path().cwd() / pathlib.Path("tests/test_lidar_fetch/instruction.json")
        with open(file_path, 'r') as file_pointer:
            instructions = json.load(file_pointer)

        # define cache location - and catchment dirs
        cls.cache_dir = pathlib.Path(instructions['instructions']['data_paths']['local_cache'])

        # ensure the cache directory doesn't exist - i.e. clean up from last test occurred correctly
        if cls.cache_dir.exists():
            shutil.rmtree(cls.cache_dir)
        cls.cache_dir.mkdir()

        # create fake catchment boundary
        x0 = 1764410
        y0 = 5470382
        x1 = 1765656
        y1 = 5471702
        catchment = shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        catchment = geopandas.GeoSeries([catchment])
        catchment = catchment.set_crs(instructions['instructions']['projection'])

        # save faked catchment boundary
        catchment_dir = cls.cache_dir / "catchment"
        catchment.to_file(catchment_dir)
        shutil.make_archive(base_name=catchment_dir, format='zip', root_dir=catchment_dir)
        shutil.rmtree(catchment_dir)

        # create a catchment_geometry
        catchment_dir = pathlib.Path(str(catchment_dir) + ".zip")
        catchment_geometry = geometry.CatchmentGeometry(catchment_dir,
                                                        instructions['instructions']['projection'],
                                                        instructions['instructions']['grid_params']['resolution'])
        catchment_geometry.land = catchment_dir  # all land

        # Run pipeline - download files
        runner = lidar_fetch.OpenTopography(catchment_geometry, cls.cache_dir, verbose=True)
        runner.run()

    @classmethod
    def tearDownClass(cls):
        """ Remove created cache directory and included created and downloaded files at the end of the test. """

        if cls.cache_dir.exists():
            shutil.rmtree(cls.cache_dir)

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


if __name__ == '__main__':
    unittest.main()
